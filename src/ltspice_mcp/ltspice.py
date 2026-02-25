from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import SimulationDiagnostic, SimulationRun
from .textio import read_text_auto


_END_DIRECTIVE_RE = re.compile(r"(?im)^\s*\.end\b")

_CATEGORY_RULES: list[tuple[str, str, re.Pattern[str], str]] = [
    (
        "convergence",
        "error",
        re.compile(
            r"(?i)(time step too small|convergence failed|gmin stepping failed|source stepping failed|newton iteration failed)"
        ),
        "Add realistic parasitics or bleeder resistors, set reasonable initial conditions (.ic), and relax time step constraints.",
    ),
    (
        "floating_node",
        "error",
        re.compile(r"(?i)(singular matrix|floating)"),
        "Ensure every node has a DC return path (often via a large resistor to ground) and verify all pins are connected.",
    ),
    (
        "model_missing",
        "error",
        re.compile(
            r"(?i)(unknown subcircuit|can't find definition of model|could not open include file|unable to open .*\\.lib)"
        ),
        "Check .include/.lib paths and model names, and make sure referenced model files exist in accessible paths.",
    ),
    (
        "netlist_syntax",
        "error",
        re.compile(r"(?i)(syntax error|unknown parameter|missing value|expected .* token)"),
        "Inspect the failing line in the netlist and verify directive spelling, parameter order, and numeric units.",
    ),
    (
        "generic_error",
        "error",
        re.compile(r"(?i)\b(fatal|error)\b"),
        "Inspect the log context around this message and adjust netlist directives or model includes accordingly.",
    ),
    (
        "warning",
        "warning",
        re.compile(r"(?i)\bwarning\b"),
        "Review warning context to ensure simulation accuracy is acceptable.",
    ),
]


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def sanitize_project_name(name: str) -> str:
    safe = "".join(char if (char.isalnum() or char in "_-") else "_" for char in name)
    safe = safe.strip("_")
    return safe or "circuit"


def find_ltspice_executable(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.getenv("LTSPICE_BINARY")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for name in ("LTspice", "ltspice"):
        which = shutil.which(name)
        if which:
            candidates.append(Path(which))

    candidates.extend(
        [
            Path("/Applications/LTspice.app/Contents/MacOS/LTspice"),
            Path("/Applications/LTspice.app/Contents/MacOS/LTspice XVII"),
            Path("/Applications/ADI/LTspice/LTspice.app/Contents/MacOS/LTspice"),
            Path.home() / "Applications/LTspice.app/Contents/MacOS/LTspice",
            Path.home() / "Applications/ADI/LTspice/LTspice.app/Contents/MacOS/LTspice",
        ]
    )

    for root in (Path("/Applications"), Path.home() / "Applications"):
        if not root.exists():
            continue
        for app_dir in root.glob("**/LTspice*.app/Contents/MacOS/*"):
            candidates.append(app_dir)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_executable(resolved):
            return resolved
    return None


def get_ltspice_version(executable: Path) -> str | None:
    for flag in ("-version", "-v"):
        try:
            proc = subprocess.run(
                [str(executable), flag],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        output = (proc.stdout + "\n" + proc.stderr).strip()
        if output:
            return output.splitlines()[0].strip()
    return None


def analyze_log(log_path: Path | None) -> tuple[list[str], list[str], list[SimulationDiagnostic]]:
    if not log_path or not log_path.exists():
        return [], [], []

    issues: list[str] = []
    warnings: list[str] = []
    diagnostics: list[SimulationDiagnostic] = []
    seen_issue: set[str] = set()
    seen_warning: set[str] = set()
    seen_diag: set[tuple[str, str]] = set()

    for raw_line in read_text_auto(log_path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for category, severity, pattern, suggestion in _CATEGORY_RULES:
            if not pattern.search(line):
                continue

            key = (category, line)
            if key in seen_diag:
                continue
            seen_diag.add(key)
            diagnostics.append(
                SimulationDiagnostic(
                    category=category,
                    severity=severity,
                    message=line,
                    suggestion=suggestion,
                )
            )

            if severity == "error" and line not in seen_issue:
                issues.append(line)
                seen_issue.add(line)
            if severity == "warning" and line not in seen_warning:
                warnings.append(line)
                seen_warning.add(line)
            break

    return issues, warnings, diagnostics


def tail_text_file(path: Path | None, max_lines: int = 120) -> str:
    if not path or not path.exists():
        return ""
    lines = read_text_auto(path).splitlines()
    return "\n".join(lines[-max_lines:])


def is_ltspice_ui_running() -> bool:
    try:
        proc = subprocess.run(
            ["pgrep", "-x", "LTspice"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def open_in_ltspice_ui(
    path: str | Path,
    *,
    background: bool = True,
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Cannot open missing path in LTspice UI: {target}")
    if platform.system() != "Darwin":
        raise RuntimeError("LTspice UI integration is currently implemented for macOS only.")

    command = ["open"]
    effective_background = bool(background)
    if effective_background:
        # `-g` avoids foreground activation and `-j` asks LaunchServices to launch hidden.
        # Together they reduce chances of macOS Space switches during automation.
        command.extend(["-g", "-j"])
    command.extend(["-a", "LTspice", str(target)])
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "opened": proc.returncode == 0,
        "return_code": proc.returncode,
        "path": str(target),
        "background_requested": background,
        "background": effective_background,
        "command": command,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def close_ltspice_window(title_hint: str) -> dict[str, Any]:
    if not title_hint.strip():
        return {
            "closed": False,
            "return_code": 1,
            "title_hint": title_hint,
            "error": "title_hint must be non-empty to close LTspice window",
        }
    if platform.system() != "Darwin":
        raise RuntimeError("LTspice UI integration is currently implemented for macOS only.")

    escaped = title_hint.replace('"', '\\"')
    script = (
        'tell application "System Events"\n'
        '  if not (exists process "LTspice") then return\n'
        '  tell process "LTspice"\n'
        f'    set matches to (windows whose name contains "{escaped}")\n'
        '    repeat with w in matches\n'
        '      try\n'
        '        click (first button of w whose subrole is "AXCloseButton")\n'
        '      on error\n'
        '        try\n'
        '          perform action "AXCloseButton" of w\n'
        '        end try\n'
        '      end try\n'
        '    end repeat\n'
        '  end tell\n'
        'end tell'
    )
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "closed": proc.returncode == 0,
        "return_code": proc.returncode,
        "title_hint": title_hint,
        "command": ["osascript", "-e", script],
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _capture_ltspice_window_with_screencapturekit(
    *,
    output_path: Path,
    title_hint: str | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise RuntimeError("ScreenCaptureKit capture is currently implemented for macOS only.")
    if shutil.which("xcrun") is None:
        raise RuntimeError("xcrun not found; ScreenCaptureKit backend is unavailable.")

    swift_source = r'''
import Foundation
import AppKit
import ScreenCaptureKit
import CoreGraphics
import CoreMedia
import CoreVideo
import VideoToolbox
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
if args.count < 3 {
    fputs("usage: <outputPath> <titleHint>\n", stderr)
    exit(2)
}

let outputURL = URL(fileURLWithPath: args[1])
let titleHint = args[2].trimmingCharacters(in: .whitespacesAndNewlines)

let _ = NSApplication.shared

func emitJSON(_ payload: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    }
}

enum CaptureError: LocalizedError {
    case timedOut
    case conversionFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .timedOut:
            return "Timed out waiting for first stream frame."
        case .conversionFailed(let status):
            return "VTCreateCGImageFromCVPixelBuffer failed (status=\(status))."
        }
    }
}

final class FrameSink: NSObject, SCStreamOutput {
    private var continuation: CheckedContinuation<CGImage, Error>?
    private let lock = NSLock()

    func waitForFrame(timeoutSeconds: TimeInterval) async throws -> CGImage {
        try await withThrowingTaskGroup(of: CGImage.self) { group in
            group.addTask {
                try await withCheckedThrowingContinuation { (cont: CheckedContinuation<CGImage, Error>) in
                    self.lock.lock()
                    self.continuation = cont
                    self.lock.unlock()
                }
            }
            group.addTask {
                let nanos = UInt64(max(1, Int(timeoutSeconds * 1_000_000_000)))
                try await Task.sleep(nanoseconds: nanos)
                throw CaptureError.timedOut
            }
            let frame = try await group.next()!
            group.cancelAll()
            return frame
        }
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .screen else {
            return
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        var image: CGImage?
        let status = VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &image)
        if status == noErr, let image {
            resumeOnce(with: .success(image))
        } else {
            resumeOnce(with: .failure(CaptureError.conversionFailed(status)))
        }
    }

    private func resumeOnce(with result: Result<CGImage, Error>) {
        lock.lock()
        guard let cont = continuation else {
            lock.unlock()
            return
        }
        continuation = nil
        lock.unlock()

        switch result {
        case .success(let image):
            cont.resume(returning: image)
        case .failure(let error):
            cont.resume(throwing: error)
        }
    }
}

let sema = DispatchSemaphore(value: 0)
var failed = false

Task {
    defer { sema.signal() }
    do {
        let shareable = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        let candidates = shareable.windows.filter {
            ($0.owningApplication?.applicationName ?? "") == "LTspice" &&
            $0.frame.width > 80 &&
            $0.frame.height > 80
        }

        guard !candidates.isEmpty else {
            fputs("No LTspice windows found.\n", stderr)
            failed = true
            return
        }

        let selected: SCWindow
        var titleHintMatched = false
        var selectionReason = "largest_window"
        if !titleHint.isEmpty {
            if let exact = candidates.first(where: { ($0.title ?? "").localizedCaseInsensitiveContains(titleHint) }) {
                selected = exact
                titleHintMatched = true
                selectionReason = "title_hint_match"
            } else {
                selected = candidates.max(by: { ($0.frame.width * $0.frame.height) < ($1.frame.width * $1.frame.height) })!
                selectionReason = "largest_window_fallback"
            }
        } else {
            selected = candidates.max(by: { ($0.frame.width * $0.frame.height) < ($1.frame.width * $1.frame.height) })!
        }
        let candidateTitles = candidates.prefix(12).map { $0.title ?? "" }

        let filter = SCContentFilter(desktopIndependentWindow: selected)
        let configuration = SCStreamConfiguration()
        configuration.width = max(1, Int(selected.frame.width))
        configuration.height = max(1, Int(selected.frame.height))
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        configuration.queueDepth = 2
        configuration.capturesAudio = false
        configuration.excludesCurrentProcessAudio = true
        configuration.showsCursor = false

        let sink = FrameSink()
        let stream = SCStream(filter: filter, configuration: configuration, delegate: nil)
        try stream.addStreamOutput(
            sink,
            type: .screen,
            sampleHandlerQueue: DispatchQueue(label: "ltspice.sck.output")
        )
        try await stream.startCapture()
        let image = try await sink.waitForFrame(timeoutSeconds: 10.0)
        try await stream.stopCapture()

        guard let destination = CGImageDestinationCreateWithURL(
            outputURL as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            fputs("Failed to create image destination.\n", stderr)
            failed = true
            return
        }

        CGImageDestinationAddImage(destination, image, nil)
        if !CGImageDestinationFinalize(destination) {
            fputs("Failed to finalize image destination.\n", stderr)
            failed = true
            return
        }

        let payload: [String: Any] = [
            "window_id": selected.windowID,
            "window_title": selected.title ?? "",
            "window_frame": [
                "x": selected.frame.origin.x,
                "y": selected.frame.origin.y,
                "width": selected.frame.width,
                "height": selected.frame.height
            ],
            "title_hint": titleHint,
            "title_hint_matched": titleHintMatched,
            "selection_reason": selectionReason,
            "candidate_count": candidates.count,
            "candidate_titles": candidateTitles,
            "capture_mode": "screencapturekit_window",
            "capture_strategy": "desktop_independent_window",
            "captured_width": image.width,
            "captured_height": image.height
        ]
        emitJSON(payload)
    } catch {
        fputs("ScreenCaptureKit error: \(error)\n", stderr)
        failed = true
    }
}

_ = sema.wait(timeout: .now() + 25)
if failed {
    exit(1)
}
'''

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".swift",
        prefix="ltspice_sck_",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(swift_source)
        script_path = Path(handle.name)

    started_monotonic = time.monotonic()
    command = [
        "xcrun",
        "swift",
        str(script_path),
        str(output_path),
        (title_hint or ""),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    finally:
        script_path.unlink(missing_ok=True)

    duration_seconds = round(time.monotonic() - started_monotonic, 6)

    if proc.returncode != 0:
        message = (proc.stderr.strip() or proc.stdout.strip() or "unknown ScreenCaptureKit failure")
        raise RuntimeError(
            f"ScreenCaptureKit capture failed after {duration_seconds:.3f}s: {message}"
        )

    payload: dict[str, Any] = {}
    for raw in reversed(proc.stdout.splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break

    # ScreenCaptureKit finalization can lag a fraction after swift exits.
    # Use a fixed post-capture settle to avoid flaky "missing file" failures.
    if not output_path.exists():
        time.sleep(0.2)
    if not output_path.exists():
        raise FileNotFoundError(
            "ScreenCaptureKit did not produce image: "
            f"{output_path} (stdout={proc.stdout.strip()[:400]!r}, stderr={proc.stderr.strip()[:400]!r})"
        )

    diagnostics = {
        "swift_command": command,
        "duration_seconds": duration_seconds,
        "return_code": proc.returncode,
        "stdout_line_count": len([line for line in proc.stdout.splitlines() if line.strip()]),
        "stderr": proc.stderr.strip(),
    }
    if payload:
        payload["capture_diagnostics"] = diagnostics
        return payload
    return {
        "capture_mode": "screencapturekit_window",
        "capture_diagnostics": diagnostics,
    }


def _downscale_image_file(path: Path, downscale_factor: float) -> dict[str, Any]:
    if downscale_factor >= 1.0:
        return {"downscaled": False}
    if downscale_factor <= 0:
        raise ValueError("downscale_factor must be > 0")

    if platform.system() != "Darwin":
        return {"downscaled": False, "warning": "downscale currently implemented with macOS sips"}

    probe = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return {"downscaled": False, "warning": probe.stderr.strip() or "sips probe failed"}

    width_match = re.search(r"pixelWidth:\s*(\d+)", probe.stdout)
    height_match = re.search(r"pixelHeight:\s*(\d+)", probe.stdout)
    if not width_match or not height_match:
        return {"downscaled": False, "warning": "could not parse image dimensions"}

    width = int(width_match.group(1))
    height = int(height_match.group(1))
    new_width = max(1, int(round(width * downscale_factor)))
    new_height = max(1, int(round(height * downscale_factor)))

    scale = subprocess.run(
        ["sips", "--resampleHeightWidth", str(new_height), str(new_width), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if scale.returncode != 0:
        return {"downscaled": False, "warning": scale.stderr.strip() or "sips resample failed"}
    return {
        "downscaled": True,
        "original_width": width,
        "original_height": height,
        "scaled_width": new_width,
        "scaled_height": new_height,
    }


def _probe_image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if platform.system() != "Darwin":
        return None, None
    probe = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None, None
    width_match = re.search(r"pixelWidth:\s*(\d+)", probe.stdout)
    height_match = re.search(r"pixelHeight:\s*(\d+)", probe.stdout)
    if not width_match or not height_match:
        return None, None
    return int(width_match.group(1)), int(height_match.group(1))


def capture_ltspice_window_screenshot(
    *,
    output_path: str | Path,
    open_path: str | Path | None = None,
    settle_seconds: float = 1.0,
    downscale_factor: float = 1.0,
    title_hint: str | None = None,
    avoid_space_switch: bool = True,
    prefer_screencapturekit: bool = True,
    close_after_capture: bool = True,
) -> dict[str, Any]:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    preflight = {
        "platform": platform.system(),
        "prefer_screencapturekit": bool(prefer_screencapturekit),
        "avoid_space_switch": bool(avoid_space_switch),
        "open_path": str(Path(open_path).expanduser().resolve()) if open_path is not None else None,
        "title_hint_requested": title_hint,
        "xcrun_available": shutil.which("xcrun"),
        "ltspice_ui_running_before_open": is_ltspice_ui_running(),
    }

    open_event: dict[str, Any] | None = None
    close_event: dict[str, Any] | None = None
    opened_window = False
    if open_path is not None:
        open_event = open_in_ltspice_ui(
            open_path,
            background=avoid_space_switch,
        )
        if not open_event.get("opened", False):
            raise RuntimeError(f"Failed to open LTspice UI target: {open_event}")
        opened_window = True

    if settle_seconds > 0:
        time.sleep(settle_seconds)

    if title_hint is None and open_path is not None:
        title_hint = Path(open_path).name

    capture_command: list[str] | None = None
    capture_backend = "screencapturekit" if prefer_screencapturekit else "screencapture"
    capture_stderr = ""
    window_id: int | None = None
    window_info: dict[str, Any] = {}

    try:
        if prefer_screencapturekit:
            try:
                window_info = _capture_ltspice_window_with_screencapturekit(
                    output_path=target,
                    title_hint=title_hint,
                    timeout_seconds=max(10.0, settle_seconds + 20.0),
                )
                capture_backend = "screencapturekit"
                raw_window_id = window_info.get("window_id")
                if isinstance(raw_window_id, int):
                    window_id = raw_window_id
            except Exception as exc:
                elapsed = round(time.monotonic() - started_monotonic, 6)
                raise RuntimeError(
                    "ScreenCaptureKit capture failed: "
                    f"{exc}; diagnostics={json.dumps({'elapsed_seconds': elapsed, 'preflight': preflight})}"
                ) from exc

        if capture_backend != "screencapturekit":
            # Optional non-ScreenCaptureKit path when explicitly requested.
            capture_command = ["screencapture", "-x", str(target)]
            capture = subprocess.run(
                capture_command,
                capture_output=True,
                text=True,
                check=False,
            )
            if capture.returncode != 0:
                raise RuntimeError(
                    f"screencapture failed (rc={capture.returncode}): {capture.stderr.strip()}"
                )
            capture_stderr = capture.stderr.strip()
        else:
            capture = None
    finally:
        if opened_window and close_after_capture:
            close_title = title_hint or (Path(open_path).name if open_path is not None else "")
            try:
                close_event = close_ltspice_window(close_title)
            except Exception as exc:
                close_event = {
                    "closed": False,
                    "return_code": 1,
                    "title_hint": close_title,
                    "error": str(exc),
                }

    if not target.exists():
        raise FileNotFoundError(f"Screenshot capture did not produce file: {target}")

    downscale_info = _downscale_image_file(target, downscale_factor=downscale_factor)
    width, height = _probe_image_dimensions(target)
    elapsed = round(time.monotonic() - started_monotonic, 6)
    return {
        "image_path": str(target),
        "format": target.suffix.lstrip(".").lower() or "png",
        "window_id": window_id,
        "capture_command": capture_command,
        "capture_backend": capture_backend,
        "capture_window_info": window_info,
        "open_event": open_event,
        "close_event": close_event,
        "avoid_space_switch": avoid_space_switch,
        "downscale_factor": float(downscale_factor),
        "downscale": downscale_info,
        "width": width,
        "height": height,
        "capture_stderr": capture_stderr,
        "capture_diagnostics": {
            "elapsed_seconds": elapsed,
            "settle_seconds": float(settle_seconds),
            "title_hint_used": title_hint,
            "preflight": preflight,
        },
    }


class LTspiceRunner:
    def __init__(
        self,
        *,
        workdir: Path,
        executable: str | Path | None = None,
        default_timeout_seconds: int = 120,
    ) -> None:
        self.workdir = workdir.expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._executable = find_ltspice_executable(executable)
        self.default_timeout_seconds = default_timeout_seconds

    @property
    def executable(self) -> Path | None:
        return self._executable

    def ensure_executable(self) -> Path:
        executable = self._executable or find_ltspice_executable()
        if executable is None:
            raise RuntimeError(
                "Could not find LTspice executable. Set LTSPICE_BINARY or pass --ltspice-binary."
            )
        self._executable = executable
        return executable

    def write_netlist(self, netlist_content: str, circuit_name: str) -> Path:
        if not netlist_content.strip():
            raise ValueError("netlist_content cannot be empty")

        safe_name = sanitize_project_name(circuit_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.workdir / "runs" / f"{stamp}_{safe_name}"
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = self.workdir / "runs" / f"{stamp}_{safe_name}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)

        content = netlist_content.rstrip() + "\n"
        if not _END_DIRECTIVE_RE.search(content):
            content += ".end\n"

        netlist_path = run_dir / f"{safe_name}.cir"
        netlist_path.write_text(content, encoding="utf-8")
        return netlist_path

    def run_file(
        self,
        netlist_path: str | Path,
        *,
        ascii_raw: bool = False,
        timeout_seconds: int | None = None,
    ) -> SimulationRun:
        netlist = Path(netlist_path).expanduser().resolve()
        if not netlist.exists():
            raise FileNotFoundError(f"Netlist not found: {netlist}")

        executable = self.ensure_executable()
        timeout = timeout_seconds or self.default_timeout_seconds
        command = [str(executable), "-b"]
        if ascii_raw:
            command.append("-ascii")
        command.append(str(netlist))

        started_at = datetime.now().astimezone().isoformat()
        start_ts = time.time()

        try:
            proc = subprocess.run(
                command,
                cwd=netlist.parent,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            return_code = -1
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nLTspice timed out after {timeout} seconds."

        duration = time.time() - start_ts
        artifacts = sorted(path for path in netlist.parent.glob(f"{netlist.stem}*") if path.is_file())
        raw_files = [path for path in artifacts if path.suffix.lower() == ".raw"]

        log_path = netlist.with_suffix(".log")
        if not log_path.exists():
            candidates = sorted(netlist.parent.glob(f"{netlist.stem}*.log"))
            log_path = candidates[0] if candidates else None

        issues, warnings, diagnostics = analyze_log(log_path)
        if return_code != 0:
            issues.append(f"LTspice exited with return code {return_code}.")
            diagnostics.append(
                SimulationDiagnostic(
                    category="process_error",
                    severity="error",
                    message=f"LTspice exited with return code {return_code}.",
                    suggestion="Check stderr output and LTspice log details for the underlying simulation failure.",
                )
            )
        if not raw_files and return_code == 0:
            warnings.append("No .raw output file was generated.")
            diagnostics.append(
                SimulationDiagnostic(
                    category="missing_artifact",
                    severity="warning",
                    message="No .raw output file was generated.",
                    suggestion="Ensure the netlist includes a simulation directive such as .tran, .ac, .dc, or .op.",
                )
            )

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return SimulationRun(
            run_id=run_id,
            netlist_path=netlist,
            command=command,
            ltspice_executable=executable,
            started_at=started_at,
            duration_seconds=duration,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            log_path=log_path,
            raw_files=raw_files,
            artifacts=artifacts,
            issues=issues,
            warnings=warnings,
            diagnostics=diagnostics,
        )
