from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import plistlib
import re
import shutil
import subprocess
import time
from collections import Counter, deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import SimulationDiagnostic, SimulationRun
from .textio import read_text_auto


_END_DIRECTIVE_RE = re.compile(r"(?im)^\s*\.end\b")
_LOGGER = logging.getLogger(__name__)
_CAPTURE_HISTORY_LIMIT = max(50, int(os.getenv("LTSPICE_MCP_CAPTURE_HISTORY_LIMIT", "800")))
_capture_event_history: deque[dict[str, Any]] = deque(maxlen=_CAPTURE_HISTORY_LIMIT)

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


def _log_capture_event(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": logging.getLevelName(level),
        **payload,
    }
    _capture_event_history.append(history_entry)
    try:
        encoded = json.dumps(payload, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        encoded = str(payload)
    _LOGGER.log(level, "ltspice_capture %s", encoded)


def get_capture_event_history(limit: int = 200) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return list(_capture_event_history)[-limit:]


def get_capture_health_snapshot(limit: int = 400) -> dict[str, Any]:
    events = get_capture_event_history(limit=limit)
    event_counts = Counter(str(item.get("event", "")) for item in events)
    starts = int(event_counts.get("capture_start", 0))
    successes = int(event_counts.get("capture_success", 0))
    failure_events = [
        "capture_open_failed",
        "capture_screencapturekit_failed",
        "capture_screencapture_failed",
        "capture_file_missing",
        "capture_close_verification_mismatch",
    ]
    failures = sum(int(event_counts.get(name, 0)) for name in failure_events)
    close_incomplete = int(event_counts.get("capture_close_incomplete", 0)) + int(
        event_counts.get("capture_close_verification_mismatch", 0)
    )
    success_rate = round(min(successes, starts) / starts, 4) if starts > 0 else None
    latest_failure = next(
        (
            item
            for item in reversed(events)
            if str(item.get("event", "")) in set(failure_events + ["capture_close_incomplete"])
        ),
        None,
    )
    return {
        "total_events_considered": len(events),
        "capture_starts": starts,
        "capture_successes": successes,
        "capture_failures": failures,
        "capture_close_incomplete": close_incomplete,
        "success_rate": success_rate,
        "window_truncated_bias": bool(successes > starts),
        "event_counts": dict(sorted(event_counts.items())),
        "latest_event": events[-1] if events else None,
        "latest_failure": latest_failure,
    }


def _write_utf8_log_sidecar(log_path: Path | None) -> Path | None:
    if log_path is None or not log_path.exists():
        return None
    try:
        text = read_text_auto(log_path)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Failed to decode LTspice log '%s' for utf8 sidecar: %s", log_path, exc)
        return None

    sidecar = log_path.with_name(f"{log_path.name}.utf8.txt")
    try:
        if sidecar.exists() and sidecar.read_text(encoding="utf-8", errors="ignore") == text:
            return sidecar
        sidecar.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Failed to write LTspice utf8 log sidecar '%s': %s", sidecar, exc)
        return None
    return sidecar

_SCK_HELPER_SOURCE = r"""
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
    fputs("usage: <outputPath> <titleHint> [captureTimeoutSeconds]\n", stderr)
    exit(2)
}

let outputURL = URL(fileURLWithPath: args[1])
let titleHint = args[2].trimmingCharacters(in: .whitespacesAndNewlines)
let captureTimeoutSeconds = args.count >= 4 ? max(1.0, Double(args[3]) ?? 10.0) : 10.0

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
        let image = try await sink.waitForFrame(timeoutSeconds: captureTimeoutSeconds)
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

_ = sema.wait(timeout: .now() + 30)
if failed {
    exit(1)
}
"""
_SCK_HELPER_FILENAME = "ltspice-sck-helper"
_SCK_HELPER_HASH_FILENAME = ".ltspice-sck-helper.sha256"
_SCK_HELPER_SWIFT_FILENAME = "ltspice-sck-helper.swift"
_AX_CLOSE_HELPER_FILENAME = "ltspice-ax-close-helper"
_AX_CLOSE_HELPER_HASH_FILENAME = ".ltspice-ax-close-helper.sha256"
_AX_CLOSE_HELPER_SWIFT_FILENAME = "ltspice-ax-close-helper.swift"
_AX_TEXT_HELPER_FILENAME = "ltspice-ax-text-helper"
_AX_TEXT_HELPER_HASH_FILENAME = ".ltspice-ax-text-helper.sha256"
_AX_TEXT_HELPER_SWIFT_FILENAME = "ltspice-ax-text-helper.swift"

_AX_CLOSE_HELPER_SOURCE = r"""
import Foundation
import AppKit
import ApplicationServices
import CoreGraphics

let args = CommandLine.arguments
if args.count < 6 {
    fputs("usage: <titleContains> <titleExact> <windowId> <maxPasses> <settleMs>\n", stderr)
    exit(2)
}

let titleContains = args[1].trimmingCharacters(in: .whitespacesAndNewlines)
let titleExact = args[2].trimmingCharacters(in: .whitespacesAndNewlines)
let windowIdRaw = args[3].trimmingCharacters(in: .whitespacesAndNewlines)
let maxPasses = max(1, min(15, Int(args[4]) ?? 1))
let settleMs = max(0, min(2000, Int(args[5]) ?? 150))
let targetWindowId: Int64? = Int64(windowIdRaw)

func emitJSON(_ payload: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    }
}

func cfString(_ value: String) -> CFString {
    return value as CFString
}

func runningLTspicePids() -> Set<pid_t> {
    var pids: [pid_t] = []
    for app in NSWorkspace.shared.runningApplications {
        if app.isTerminated {
            continue
        }
        let name = app.localizedName ?? ""
        let bundleId = app.bundleIdentifier ?? ""
        if name == "LTspice" || bundleId.lowercased().contains("ltspice") {
            pids.append(app.processIdentifier)
        }
    }
    return Set(pids)
}

func asAXElement(_ value: Any?) -> AXUIElement? {
    guard let value else {
        return nil
    }
    let ref = value as CFTypeRef
    if CFGetTypeID(ref) == AXUIElementGetTypeID() {
        return unsafeBitCast(ref, to: AXUIElement.self)
    }
    return nil
}

func asInt64(_ value: Any?) -> Int64? {
    if let number = value as? NSNumber {
        return number.int64Value
    }
    if let intValue = value as? Int {
        return Int64(intValue)
    }
    if let text = value as? String {
        return Int64(text)
    }
    return nil
}

func asPid(_ value: Any?) -> pid_t? {
    guard let intValue = asInt64(value) else {
        return nil
    }
    return pid_t(intValue)
}

func windowTitle(_ window: AXUIElement) -> String {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &value) == .success,
       let title = value as? String {
        return title
    }
    return ""
}

func windowNumber(_ window: AXUIElement) -> Int64? {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, cfString("AXWindowNumber"), &value) != .success {
        return nil
    }
    if let number = value as? NSNumber {
        return number.int64Value
    }
    return nil
}

func closeWindow(_ window: AXUIElement) -> Bool {
    var closeButtonValue: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, kAXCloseButtonAttribute as CFString, &closeButtonValue) == .success,
       let closeButtonValue,
       let closeButton = asAXElement(closeButtonValue),
       AXUIElementPerformAction(closeButton, kAXPressAction as CFString) == .success {
        return true
    }
    if AXUIElementPerformAction(window, cfString("AXClose")) == .success {
        return true
    }
    return false
}

func windowPoint(_ window: AXUIElement, key: String) -> CGPoint? {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, key as CFString, &value) != .success {
        return nil
    }
    guard let axValue = value else {
        return nil
    }
    if CFGetTypeID(axValue) != AXValueGetTypeID() {
        return nil
    }
    let typed = unsafeBitCast(axValue, to: AXValue.self)
    var point = CGPoint.zero
    if AXValueGetType(typed) != .cgPoint {
        return nil
    }
    if AXValueGetValue(typed, .cgPoint, &point) {
        return point
    }
    return nil
}

func windowSize(_ window: AXUIElement) -> CGSize? {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, kAXSizeAttribute as CFString, &value) != .success {
        return nil
    }
    guard let axValue = value else {
        return nil
    }
    if CFGetTypeID(axValue) != AXValueGetTypeID() {
        return nil
    }
    let typed = unsafeBitCast(axValue, to: AXValue.self)
    var size = CGSize.zero
    if AXValueGetType(typed) != .cgSize {
        return nil
    }
    if AXValueGetValue(typed, .cgSize, &size) {
        return size
    }
    return nil
}

func windowKey(_ window: AXUIElement, pid: pid_t) -> String {
    let title = windowTitle(window)
    let position = windowPoint(window, key: kAXPositionAttribute as String) ?? .zero
    let size = windowSize(window) ?? .zero
    return "\(pid)|\(title)|\(Int(position.x))|\(Int(position.y))|\(Int(size.width))|\(Int(size.height))"
}

func appendWindow(_ elementRef: CFTypeRef?, pid: pid_t, into windows: inout [AXUIElement], seen: inout Set<String>) {
    guard let window = asAXElement(elementRef) else {
        return
    }
    let key = windowKey(window, pid: pid)
    if seen.contains(key) {
        return
    }
    seen.insert(key)
    windows.append(window)
}

func candidateWindows(_ appElement: AXUIElement, pid: pid_t) -> [AXUIElement] {
    var windows: [AXUIElement] = []
    var seen: Set<String> = []

    var rawWindows: CFTypeRef?
    if AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &rawWindows) == .success,
       let rawWindows {
        if let array = rawWindows as? [Any] {
            for item in array {
                appendWindow(item as CFTypeRef, pid: pid, into: &windows, seen: &seen)
            }
        } else if let array = rawWindows as? NSArray {
            for item in array {
                appendWindow(item as CFTypeRef, pid: pid, into: &windows, seen: &seen)
            }
        }
    }

    var focusedWindow: CFTypeRef?
    if AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &focusedWindow) == .success {
        appendWindow(focusedWindow, pid: pid, into: &windows, seen: &seen)
    }

    var mainWindow: CFTypeRef?
    if AXUIElementCopyAttributeValue(appElement, kAXMainWindowAttribute as CFString, &mainWindow) == .success {
        appendWindow(mainWindow, pid: pid, into: &windows, seen: &seen)
    }

    return windows
}

func matchesSelector(_ window: AXUIElement) -> Bool {
    let title = windowTitle(window)
    let number = windowNumber(window)
    var isMatch = false
    if let targetWindowId, number == targetWindowId {
        isMatch = true
    }
    if !isMatch && !titleExact.isEmpty && title == titleExact {
        isMatch = true
    }
    if !isMatch && !titleContains.isEmpty && title.localizedCaseInsensitiveContains(titleContains) {
        isMatch = true
    }
    return isMatch
}

func cgMatchingWindows() -> [[String: Any]] {
    guard let info = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] else {
        return []
    }
    var matches: [[String: Any]] = []
    for entry in info {
        let owner = entry[kCGWindowOwnerName as String] as? String ?? ""
        if owner != "LTspice" {
            continue
        }
        let title = entry[kCGWindowName as String] as? String ?? ""
        let windowNumber = asInt64(entry[kCGWindowNumber as String])
        var isMatch = false
        if let targetWindowId, let windowNumber, windowNumber == targetWindowId {
            isMatch = true
        }
        if !isMatch && !titleExact.isEmpty && title == titleExact {
            isMatch = true
        }
        if !isMatch && !titleContains.isEmpty && title.localizedCaseInsensitiveContains(titleContains) {
            isMatch = true
        }
        if isMatch {
            matches.append(entry)
        }
    }
    return matches
}

func cgMatchingPidSet() -> Set<pid_t> {
    var pids: Set<pid_t> = []
    for entry in cgMatchingWindows() {
        if let pid = asPid(entry[kCGWindowOwnerPID as String]) {
            pids.insert(pid)
        }
    }
    return pids
}

func matchingWindows(targetPids: Set<pid_t>) -> [AXUIElement] {
    var matches: [AXUIElement] = []
    for pid in targetPids {
        let appElement = AXUIElementCreateApplication(pid)
        for window in candidateWindows(appElement, pid: pid) {
            if matchesSelector(window) {
                matches.append(window)
            }
        }
    }
    return matches
}

if titleContains.isEmpty && titleExact.isEmpty && targetWindowId == nil {
    emitJSON([
        "status": "INVALID_SELECTORS",
        "closed": false,
        "matched_windows": 0,
        "closed_windows": 0
    ])
    exit(2)
}

if !AXIsProcessTrusted() {
    emitJSON([
        "status": "AX_NOT_TRUSTED",
        "closed": false,
        "matched_windows": 0,
        "closed_windows": 0
    ])
    exit(1)
}

let pids = runningLTspicePids()
if pids.isEmpty {
    emitJSON([
        "status": "PROCESS_MISSING",
        "closed": false,
        "matched_windows": 0,
        "closed_windows": 0,
        "remaining_windows": 0
    ])
    exit(0)
}

let initialCgMatches = cgMatchingWindows()
let initialMatchedCount = initialCgMatches.count
let matchingPidSet = cgMatchingPidSet()
let targetPids = matchingPidSet.isEmpty ? pids : matchingPidSet
var passEvents: [[String: Any]] = []
var totalCloseActions = 0

for passIndex in 1...maxPasses {
    let matches = matchingWindows(targetPids: targetPids)
    if matches.isEmpty {
        let remaining = cgMatchingWindows().count
        passEvents.append([
            "pass": passIndex,
            "ax_matched": 0,
            "close_actions": 0,
            "remaining": remaining
        ])
        if remaining == 0 {
            break
        }
        if settleMs > 0 {
            usleep(useconds_t(settleMs * 1000))
        }
        continue
    }
    var closeActions = 0
    for window in matches {
        if closeWindow(window) {
            closeActions += 1
        }
    }
    totalCloseActions += closeActions
    if settleMs > 0 {
        usleep(useconds_t(settleMs * 1000))
    }
    let remaining = cgMatchingWindows().count
    passEvents.append([
        "pass": passIndex,
        "ax_matched": matches.count,
        "close_actions": closeActions,
        "remaining": remaining
    ])
    if remaining == 0 {
        break
    }
}

let remainingCount = cgMatchingWindows().count
let closedCount = max(0, initialMatchedCount - remainingCount)
let closed = initialMatchedCount > 0 && remainingCount == 0
let partiallyClosed = !closed && closedCount > 0

emitJSON([
    "status": "OK",
    "closed": closed,
    "partially_closed": partiallyClosed,
    "matched_windows": initialMatchedCount,
    "closed_windows": closedCount,
    "remaining_windows": remainingCount,
    "close_strategy": "ax_helper",
    "pid_count": pids.count,
    "target_pid_count": targetPids.count,
    "attempt_count": passEvents.count,
    "attempts": passEvents
])
"""

_AX_TEXT_HELPER_SOURCE = r"""
import Foundation
import AppKit
import ApplicationServices
import CoreGraphics

let args = CommandLine.arguments
if args.count < 5 {
    fputs("usage: <titleContains> <titleExact> <windowId> <maxChars>\n", stderr)
    exit(2)
}

let titleContains = args[1].trimmingCharacters(in: .whitespacesAndNewlines)
let titleExact = args[2].trimmingCharacters(in: .whitespacesAndNewlines)
let windowIdRaw = args[3].trimmingCharacters(in: .whitespacesAndNewlines)
let targetWindowId: Int64? = Int64(windowIdRaw)
let maxChars = max(512, min(2_000_000, Int(args[4]) ?? 200_000))

func emitJSON(_ payload: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
       let text = String(data: data, encoding: .utf8) {
        print(text)
    }
}

func cfString(_ value: String) -> CFString {
    return value as CFString
}

func runningLTspicePids() -> Set<pid_t> {
    var pids: [pid_t] = []
    for app in NSWorkspace.shared.runningApplications {
        if app.isTerminated {
            continue
        }
        let name = app.localizedName ?? ""
        let bundleId = app.bundleIdentifier ?? ""
        if name == "LTspice" || bundleId.lowercased().contains("ltspice") {
            pids.append(app.processIdentifier)
        }
    }
    return Set(pids)
}

func asAXElement(_ value: Any?) -> AXUIElement? {
    guard let value else {
        return nil
    }
    let ref = value as CFTypeRef
    if CFGetTypeID(ref) == AXUIElementGetTypeID() {
        return unsafeBitCast(ref, to: AXUIElement.self)
    }
    return nil
}

func asInt64(_ value: Any?) -> Int64? {
    if let number = value as? NSNumber {
        return number.int64Value
    }
    if let intValue = value as? Int {
        return Int64(intValue)
    }
    if let text = value as? String {
        return Int64(text)
    }
    return nil
}

func asPid(_ value: Any?) -> pid_t? {
    guard let intValue = asInt64(value) else {
        return nil
    }
    return pid_t(intValue)
}

func windowTitle(_ window: AXUIElement) -> String {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &value) == .success,
       let title = value as? String {
        return title
    }
    return ""
}

func windowNumber(_ window: AXUIElement) -> Int64? {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, cfString("AXWindowNumber"), &value) != .success {
        return nil
    }
    if let number = value as? NSNumber {
        return number.int64Value
    }
    return nil
}

func windowPoint(_ window: AXUIElement, key: String) -> CGPoint? {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, key as CFString, &value) != .success {
        return nil
    }
    guard let axValue = value else {
        return nil
    }
    if CFGetTypeID(axValue) != AXValueGetTypeID() {
        return nil
    }
    let typed = unsafeBitCast(axValue, to: AXValue.self)
    var point = CGPoint.zero
    if AXValueGetType(typed) != .cgPoint {
        return nil
    }
    if AXValueGetValue(typed, .cgPoint, &point) {
        return point
    }
    return nil
}

func windowSize(_ window: AXUIElement) -> CGSize? {
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(window, kAXSizeAttribute as CFString, &value) != .success {
        return nil
    }
    guard let axValue = value else {
        return nil
    }
    if CFGetTypeID(axValue) != AXValueGetTypeID() {
        return nil
    }
    let typed = unsafeBitCast(axValue, to: AXValue.self)
    var size = CGSize.zero
    if AXValueGetType(typed) != .cgSize {
        return nil
    }
    if AXValueGetValue(typed, .cgSize, &size) {
        return size
    }
    return nil
}

func windowKey(_ window: AXUIElement, pid: pid_t) -> String {
    let title = windowTitle(window)
    let position = windowPoint(window, key: kAXPositionAttribute as String) ?? .zero
    let size = windowSize(window) ?? .zero
    return "\(pid)|\(title)|\(Int(position.x))|\(Int(position.y))|\(Int(size.width))|\(Int(size.height))"
}

func appendWindow(_ elementRef: CFTypeRef?, pid: pid_t, into windows: inout [AXUIElement], seen: inout Set<String>) {
    guard let window = asAXElement(elementRef) else {
        return
    }
    let key = windowKey(window, pid: pid)
    if seen.contains(key) {
        return
    }
    seen.insert(key)
    windows.append(window)
}

func candidateWindows(_ appElement: AXUIElement, pid: pid_t) -> [AXUIElement] {
    var windows: [AXUIElement] = []
    var seen: Set<String> = []
    var rawWindows: CFTypeRef?
    if AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &rawWindows) == .success,
       let rawWindows {
        if let array = rawWindows as? [Any] {
            for item in array {
                appendWindow(item as CFTypeRef, pid: pid, into: &windows, seen: &seen)
            }
        } else if let array = rawWindows as? NSArray {
            for item in array {
                appendWindow(item as CFTypeRef, pid: pid, into: &windows, seen: &seen)
            }
        }
    }

    var focusedWindow: CFTypeRef?
    if AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &focusedWindow) == .success {
        appendWindow(focusedWindow, pid: pid, into: &windows, seen: &seen)
    }

    var mainWindow: CFTypeRef?
    if AXUIElementCopyAttributeValue(appElement, kAXMainWindowAttribute as CFString, &mainWindow) == .success {
        appendWindow(mainWindow, pid: pid, into: &windows, seen: &seen)
    }
    return windows
}

func matchesSelector(_ window: AXUIElement) -> Bool {
    let title = windowTitle(window)
    let number = windowNumber(window)
    var isMatch = false
    if let targetWindowId, number == targetWindowId {
        isMatch = true
    }
    if !isMatch && !titleExact.isEmpty && title == titleExact {
        isMatch = true
    }
    if !isMatch && !titleContains.isEmpty && title.localizedCaseInsensitiveContains(titleContains) {
        isMatch = true
    }
    return isMatch
}

func cgMatchingWindows() -> [[String: Any]] {
    guard let info = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] else {
        return []
    }
    var matches: [[String: Any]] = []
    for entry in info {
        let owner = entry[kCGWindowOwnerName as String] as? String ?? ""
        if owner != "LTspice" {
            continue
        }
        let title = entry[kCGWindowName as String] as? String ?? ""
        let windowNumber = asInt64(entry[kCGWindowNumber as String])
        var isMatch = false
        if let targetWindowId, let windowNumber, windowNumber == targetWindowId {
            isMatch = true
        }
        if !isMatch && !titleExact.isEmpty && title == titleExact {
            isMatch = true
        }
        if !isMatch && !titleContains.isEmpty && title.localizedCaseInsensitiveContains(titleContains) {
            isMatch = true
        }
        if isMatch {
            matches.append(entry)
        }
    }
    return matches
}

func cgMatchingPidSet() -> Set<pid_t> {
    var pids: Set<pid_t> = []
    for entry in cgMatchingWindows() {
        if let pid = asPid(entry[kCGWindowOwnerPID as String]) {
            pids.insert(pid)
        }
    }
    return pids
}

func appendTextCandidate(_ value: Any?, into chunks: inout [String]) {
    guard let value else {
        return
    }
    if let text = value as? String {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty {
            chunks.append(cleaned)
        }
        return
    }
    if let attributed = value as? NSAttributedString {
        appendTextCandidate(attributed.string, into: &chunks)
        return
    }
    if let array = value as? [Any] {
        for item in array {
            appendTextCandidate(item, into: &chunks)
        }
        return
    }
    if let array = value as? NSArray {
        for item in array {
            appendTextCandidate(item, into: &chunks)
        }
    }
}

func appendChildElements(_ value: Any?, into children: inout [AXUIElement]) {
    if let element = asAXElement(value) {
        children.append(element)
        return
    }
    if let array = value as? [Any] {
        for item in array {
            appendChildElements(item, into: &children)
        }
        return
    }
    if let array = value as? NSArray {
        for item in array {
            appendChildElements(item, into: &children)
        }
    }
}

func elementKey(_ element: AXUIElement) -> String {
    let opaque = Unmanaged.passUnretained(element).toOpaque()
    return String(UInt(bitPattern: opaque))
}

func collectWindowText(_ root: AXUIElement) -> [String] {
    let childAttributes = Set<String>([
        kAXChildrenAttribute as String,
        "AXVisibleChildren",
        "AXContents",
        "AXRows",
        "AXColumns",
        "AXCells"
    ])
    let preferredTextAttributes = Set<String>([
        kAXValueAttribute as String,
        kAXTitleAttribute as String,
        kAXDescriptionAttribute as String,
        kAXHelpAttribute as String,
        "AXText"
    ])
    var queue: [(AXUIElement, Int)] = [(root, 0)]
    var visited: Set<String> = []
    var chunks: [String] = []
    let maxDepth = 24
    let maxNodes = 5000

    while !queue.isEmpty && visited.count < maxNodes {
        let (element, depth) = queue.removeFirst()
        let key = elementKey(element)
        if visited.contains(key) {
            continue
        }
        visited.insert(key)

        var namesRef: CFArray?
        if AXUIElementCopyAttributeNames(element, &namesRef) != .success {
            continue
        }
        guard let namesAny = namesRef else {
            continue
        }

        let names = (namesAny as? [String]) ?? ((namesAny as? NSArray)?.compactMap { $0 as? String } ?? [])
        if names.isEmpty {
            continue
        }
        for attr in names {
            var value: CFTypeRef?
            if AXUIElementCopyAttributeValue(element, attr as CFString, &value) != .success {
                continue
            }
            guard let raw = value else {
                continue
            }
            if preferredTextAttributes.contains(attr) || raw is String || raw is NSAttributedString {
                appendTextCandidate(raw, into: &chunks)
            }
            if depth < maxDepth && childAttributes.contains(attr) {
                var children: [AXUIElement] = []
                appendChildElements(raw, into: &children)
                for child in children {
                    queue.append((child, depth + 1))
                }
            }
        }
    }
    return chunks
}

func selectBestText(_ chunks: [String]) -> String {
    var ordered: [String] = []
    var seen: Set<String> = []
    for chunk in chunks {
        let cleaned = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty {
            continue
        }
        if seen.insert(cleaned).inserted {
            ordered.append(cleaned)
        }
    }
    if ordered.isEmpty {
        return ""
    }
    let measurementChunks = ordered.filter {
        let lowered = $0.lowercased()
        return lowered.contains("measurement:") || lowered.contains(".meas")
    }
    let selected = measurementChunks.isEmpty ? ordered : measurementChunks
    let combined = selected.joined(separator: "\n")
    if combined.count <= maxChars {
        return combined
    }
    return String(combined.prefix(maxChars))
}

if titleContains.isEmpty && titleExact.isEmpty && targetWindowId == nil {
    emitJSON([
        "status": "INVALID_SELECTORS",
        "ok": false,
        "text": "",
        "matched_windows": 0
    ])
    exit(2)
}

if !AXIsProcessTrusted() {
    emitJSON([
        "status": "AX_NOT_TRUSTED",
        "ok": false,
        "text": "",
        "matched_windows": 0
    ])
    exit(1)
}

let pids = runningLTspicePids()
if pids.isEmpty {
    emitJSON([
        "status": "PROCESS_MISSING",
        "ok": false,
        "text": "",
        "matched_windows": 0
    ])
    exit(0)
}

let matchingPidSet = cgMatchingPidSet()
let targetPids = matchingPidSet.isEmpty ? pids : matchingPidSet
var matches: [AXUIElement] = []
for pid in targetPids {
    let appElement = AXUIElementCreateApplication(pid)
    for window in candidateWindows(appElement, pid: pid) {
        if matchesSelector(window) {
            matches.append(window)
        }
    }
}
if matches.isEmpty && targetPids != pids {
    for pid in pids {
        let appElement = AXUIElementCreateApplication(pid)
        for window in candidateWindows(appElement, pid: pid) {
            if matchesSelector(window) {
                matches.append(window)
            }
        }
    }
}

guard let selected = matches.first else {
    let titles = cgMatchingWindows().compactMap { $0[kCGWindowName as String] as? String }
    emitJSON([
        "status": "WINDOW_NOT_FOUND",
        "ok": false,
        "text": "",
        "matched_windows": matches.count,
        "candidate_titles": Array(titles.prefix(12))
    ])
    exit(1)
}

let chunks = collectWindowText(selected)
let text = selectBestText(chunks)
let title = windowTitle(selected)
let number = windowNumber(selected)
emitJSON([
    "status": text.isEmpty ? "OK_NO_TEXT" : "OK",
    "ok": !text.isEmpty,
    "text": text,
    "text_length": text.count,
    "window_title": title,
    "window_id": number as Any,
    "matched_windows": matches.count,
    "chunk_count": chunks.count
])
"""


def _resolve_sck_helper_path() -> tuple[Path, bool]:
    explicit = os.getenv("LTSPICE_MCP_SCK_HELPER_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve(), True
    helper_dir_raw = os.getenv("LTSPICE_MCP_SCK_HELPER_DIR")
    helper_dir = (
        Path(helper_dir_raw).expanduser()
        if helper_dir_raw
        else (Path.home() / "Library" / "Application Support" / "ltspice-mcp" / "bin")
    )
    return (helper_dir / _SCK_HELPER_FILENAME).resolve(), False


def _resolve_ax_close_helper_path() -> tuple[Path, bool]:
    explicit = os.getenv("LTSPICE_MCP_AX_CLOSE_HELPER_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve(), True
    helper_dir_raw = os.getenv("LTSPICE_MCP_AX_CLOSE_HELPER_DIR")
    helper_dir = (
        Path(helper_dir_raw).expanduser()
        if helper_dir_raw
        else (Path.home() / "Library" / "Application Support" / "ltspice-mcp" / "bin")
    )
    return (helper_dir / _AX_CLOSE_HELPER_FILENAME).resolve(), False


def _resolve_ax_text_helper_path() -> tuple[Path, bool]:
    explicit = os.getenv("LTSPICE_MCP_AX_TEXT_HELPER_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve(), True
    helper_dir_raw = os.getenv("LTSPICE_MCP_AX_TEXT_HELPER_DIR")
    helper_dir = (
        Path(helper_dir_raw).expanduser()
        if helper_dir_raw
        else (Path.home() / "Library" / "Application Support" / "ltspice-mcp" / "bin")
    )
    return (helper_dir / _AX_TEXT_HELPER_FILENAME).resolve(), False


def _ensure_screencapturekit_helper() -> tuple[Path, dict[str, Any]]:
    helper_path, explicit = _resolve_sck_helper_path()
    if explicit:
        if not _is_executable(helper_path):
            raise RuntimeError(
                "LTSPICE_MCP_SCK_HELPER_PATH is set but not executable: "
                f"{helper_path}"
            )
        return helper_path, {
            "helper_path": str(helper_path),
            "helper_source": "env_path",
            "compiled": False,
        }

    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise RuntimeError(
            "swiftc not found; install Xcode command line tools "
            "or set LTSPICE_MCP_SCK_HELPER_PATH to a prebuilt helper executable."
        )

    helper_dir = helper_path.parent
    helper_dir.mkdir(parents=True, exist_ok=True)
    helper_source_path = helper_dir / _SCK_HELPER_SWIFT_FILENAME
    helper_hash_path = helper_dir / _SCK_HELPER_HASH_FILENAME
    helper_lock_path = helper_dir / ".ltspice-sck-helper.lock"

    source_hash = hashlib.sha256(_SCK_HELPER_SOURCE.encode("utf-8")).hexdigest()
    existing_hash = ""
    if helper_hash_path.exists():
        try:
            existing_hash = helper_hash_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing_hash = ""

    if _is_executable(helper_path) and existing_hash == source_hash:
        return helper_path, {
            "helper_path": str(helper_path),
            "helper_source": "compiled_cache",
            "compiled": False,
            "swiftc_path": swiftc,
            "source_hash": source_hash,
        }

    compile_stdout = ""
    compile_stderr = ""
    compile_command: list[str] = []
    compiled = False

    # Serialize helper compilation to avoid races when multiple server calls start concurrently.
    with helper_lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        existing_hash = ""
        if helper_hash_path.exists():
            try:
                existing_hash = helper_hash_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing_hash = ""

        if not (_is_executable(helper_path) and existing_hash == source_hash):
            helper_source_path.write_text(_SCK_HELPER_SOURCE, encoding="utf-8")
            compile_command = [
                swiftc,
                "-O",
                "-o",
                str(helper_path),
                str(helper_source_path),
            ]
            compile_proc = subprocess.run(
                compile_command,
                capture_output=True,
                text=True,
                check=False,
            )
            compile_stdout = compile_proc.stdout.strip()
            compile_stderr = compile_proc.stderr.strip()
            if compile_proc.returncode != 0:
                raise RuntimeError(
                    "swiftc failed to build ScreenCaptureKit helper "
                    f"(rc={compile_proc.returncode}): {compile_stderr or compile_stdout}"
                )
            helper_path.chmod(0o755)
            helper_hash_path.write_text(f"{source_hash}\n", encoding="utf-8")
            compiled = True

    return helper_path, {
        "helper_path": str(helper_path),
        "helper_source": "compiled_now" if compiled else "compiled_cache_after_lock",
        "compiled": compiled,
        "swiftc_path": swiftc,
        "source_hash": source_hash,
        "compile_command": compile_command or None,
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
    }


def _ensure_ax_close_helper() -> tuple[Path, dict[str, Any]]:
    helper_path, explicit = _resolve_ax_close_helper_path()
    if explicit:
        if not _is_executable(helper_path):
            raise RuntimeError(
                "LTSPICE_MCP_AX_CLOSE_HELPER_PATH is set but not executable: "
                f"{helper_path}"
            )
        return helper_path, {
            "helper_path": str(helper_path),
            "helper_source": "env_path",
            "compiled": False,
        }

    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise RuntimeError(
            "swiftc not found; install Xcode command line tools "
            "or set LTSPICE_MCP_AX_CLOSE_HELPER_PATH to a prebuilt helper executable."
        )

    helper_dir = helper_path.parent
    helper_dir.mkdir(parents=True, exist_ok=True)
    helper_source_path = helper_dir / _AX_CLOSE_HELPER_SWIFT_FILENAME
    helper_hash_path = helper_dir / _AX_CLOSE_HELPER_HASH_FILENAME
    helper_lock_path = helper_dir / ".ltspice-ax-close-helper.lock"

    source_hash = hashlib.sha256(_AX_CLOSE_HELPER_SOURCE.encode("utf-8")).hexdigest()
    existing_hash = ""
    if helper_hash_path.exists():
        try:
            existing_hash = helper_hash_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing_hash = ""

    if _is_executable(helper_path) and existing_hash == source_hash:
        return helper_path, {
            "helper_path": str(helper_path),
            "helper_source": "compiled_cache",
            "compiled": False,
            "swiftc_path": swiftc,
            "source_hash": source_hash,
        }

    compile_stdout = ""
    compile_stderr = ""
    compile_command: list[str] = []
    compiled = False

    with helper_lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        existing_hash = ""
        if helper_hash_path.exists():
            try:
                existing_hash = helper_hash_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing_hash = ""

        if not (_is_executable(helper_path) and existing_hash == source_hash):
            helper_source_path.write_text(_AX_CLOSE_HELPER_SOURCE, encoding="utf-8")
            compile_command = [
                swiftc,
                "-O",
                "-o",
                str(helper_path),
                str(helper_source_path),
            ]
            compile_proc = subprocess.run(
                compile_command,
                capture_output=True,
                text=True,
                check=False,
            )
            compile_stdout = compile_proc.stdout.strip()
            compile_stderr = compile_proc.stderr.strip()
            if compile_proc.returncode != 0:
                raise RuntimeError(
                    "swiftc failed to build LTspice AX close helper "
                    f"(rc={compile_proc.returncode}): {compile_stderr or compile_stdout}"
                )
            helper_path.chmod(0o755)
            helper_hash_path.write_text(f"{source_hash}\n", encoding="utf-8")
            compiled = True

    return helper_path, {
        "helper_path": str(helper_path),
        "helper_source": "compiled_now" if compiled else "compiled_cache_after_lock",
        "compiled": compiled,
        "swiftc_path": swiftc,
        "source_hash": source_hash,
        "compile_command": compile_command or None,
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
    }


def _ensure_ax_text_helper() -> tuple[Path, dict[str, Any]]:
    helper_path, explicit = _resolve_ax_text_helper_path()
    if explicit:
        if not _is_executable(helper_path):
            raise RuntimeError(
                "LTSPICE_MCP_AX_TEXT_HELPER_PATH is set but not executable: "
                f"{helper_path}"
            )
        return helper_path, {
            "helper_path": str(helper_path),
            "helper_source": "env_path",
            "compiled": False,
        }

    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise RuntimeError(
            "swiftc not found; install Xcode command line tools "
            "or set LTSPICE_MCP_AX_TEXT_HELPER_PATH to a prebuilt helper executable."
        )

    helper_dir = helper_path.parent
    helper_dir.mkdir(parents=True, exist_ok=True)
    helper_source_path = helper_dir / _AX_TEXT_HELPER_SWIFT_FILENAME
    helper_hash_path = helper_dir / _AX_TEXT_HELPER_HASH_FILENAME
    helper_lock_path = helper_dir / ".ltspice-ax-text-helper.lock"

    source_hash = hashlib.sha256(_AX_TEXT_HELPER_SOURCE.encode("utf-8")).hexdigest()
    existing_hash = ""
    if helper_hash_path.exists():
        try:
            existing_hash = helper_hash_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing_hash = ""

    if _is_executable(helper_path) and existing_hash == source_hash:
        return helper_path, {
            "helper_path": str(helper_path),
            "helper_source": "compiled_cache",
            "compiled": False,
            "swiftc_path": swiftc,
            "source_hash": source_hash,
        }

    compile_stdout = ""
    compile_stderr = ""
    compile_command: list[str] = []
    compiled = False

    with helper_lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

        existing_hash = ""
        if helper_hash_path.exists():
            try:
                existing_hash = helper_hash_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing_hash = ""

        if not (_is_executable(helper_path) and existing_hash == source_hash):
            helper_source_path.write_text(_AX_TEXT_HELPER_SOURCE, encoding="utf-8")
            compile_command = [
                swiftc,
                "-O",
                "-o",
                str(helper_path),
                str(helper_source_path),
            ]
            compile_proc = subprocess.run(
                compile_command,
                capture_output=True,
                text=True,
                check=False,
            )
            compile_stdout = compile_proc.stdout.strip()
            compile_stderr = compile_proc.stderr.strip()
            if compile_proc.returncode != 0:
                raise RuntimeError(
                    "swiftc failed to build LTspice AX text helper "
                    f"(rc={compile_proc.returncode}): {compile_stderr or compile_stdout}"
                )
            helper_path.chmod(0o755)
            helper_hash_path.write_text(f"{source_hash}\n", encoding="utf-8")
            compiled = True

    return helper_path, {
        "helper_path": str(helper_path),
        "helper_source": "compiled_now" if compiled else "compiled_cache_after_lock",
        "compiled": compiled,
        "swiftc_path": swiftc,
        "source_hash": source_hash,
        "compile_command": compile_command or None,
        "compile_stdout": compile_stdout,
        "compile_stderr": compile_stderr,
    }


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


def _ltspice_info_plist_path(executable: Path) -> Path | None:
    try:
        resolved = executable.expanduser().resolve()
    except Exception:  # noqa: BLE001
        return None
    parts = resolved.parts
    try:
        app_index = next(index for index, part in enumerate(parts) if part.endswith(".app"))
    except StopIteration:
        return None
    return Path(*parts[: app_index + 1]) / "Contents" / "Info.plist"


@lru_cache(maxsize=32)
def get_ltspice_version(executable: Path) -> str | None:
    plist_path = _ltspice_info_plist_path(executable)
    if plist_path and plist_path.exists():
        try:
            payload = plistlib.loads(plist_path.read_bytes())
        except Exception:  # noqa: BLE001
            payload = {}
        short_version = str(payload.get("CFBundleShortVersionString", "")).strip()
        bundle_version = str(payload.get("CFBundleVersion", "")).strip()
        if short_version and bundle_version and bundle_version not in short_version:
            return f"{short_version} ({bundle_version})"
        if short_version:
            return short_version
        if bundle_version:
            return bundle_version

    for flag in ("-version", "-v"):
        try:
            proc = subprocess.run(
                [str(executable), flag],
                capture_output=True,
                text=True,
                timeout=0.8,
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


def _expected_simulation_output_paths(netlist_path: Path) -> list[Path]:
    stem = netlist_path.stem
    parent = netlist_path.parent
    return [
        parent / f"{stem}.log",
        parent / f"{stem}.log.utf8.txt",
        parent / f"{stem}.raw",
        parent / f"{stem}.op.raw",
    ]


def _collect_related_artifacts(netlist_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if netlist_path.exists() and netlist_path.is_file():
        candidates.append(netlist_path)
    for candidate in _expected_simulation_output_paths(netlist_path):
        if candidate.exists() and candidate.is_file():
            candidates.append(candidate)
    return sorted({path.resolve() for path in candidates})


def _is_simulation_output_artifact(netlist_path: Path, candidate: Path) -> bool:
    if candidate == netlist_path:
        return False
    expected = {path.resolve() for path in _expected_simulation_output_paths(netlist_path)}
    try:
        resolved = candidate.resolve()
    except Exception:  # noqa: BLE001
        return False
    return resolved in expected


def _collect_simulation_output_artifacts(netlist_path: Path) -> list[Path]:
    return [
        path
        for path in _collect_related_artifacts(netlist_path)
        if _is_simulation_output_artifact(netlist_path, path)
    ]


def _purge_previous_simulation_outputs(netlist_path: Path) -> list[Path]:
    removed: list[Path] = []
    for path in _collect_simulation_output_artifacts(netlist_path):
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            _LOGGER.warning("Failed to remove stale LTspice artifact '%s': %s", path, exc)
    return removed


def _is_recent_artifact(path: Path, *, started_ts: float, grace_seconds: float = 1.0) -> bool:
    try:
        return path.stat().st_mtime >= (started_ts - max(0.0, grace_seconds))
    except OSError:
        return False


def _resolve_log_path(netlist_path: Path) -> Path | None:
    primary = netlist_path.with_suffix(".log")
    if primary.exists():
        return primary
    return None


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
        # `-g` avoids foreground activation. `-j` (launch hidden) caused window-management
        # side effects on some setups, so keep it opt-in.
        command.append("-g")
        launch_hidden = os.getenv("LTSPICE_MCP_OPEN_LAUNCH_HIDDEN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if launch_hidden:
            command.append("-j")
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


def _close_ltspice_window_with_ax_helper(
    *,
    title_hint: str,
    exact_title: str | None,
    window_id: int | None,
    attempts: int,
    retry_delay: float,
) -> dict[str, Any] | None:
    helper_disabled = os.getenv("LTSPICE_MCP_AX_CLOSE_HELPER_ENABLED", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }
    if helper_disabled:
        return None

    try:
        helper_path, helper_details = _ensure_ax_close_helper()
    except Exception as exc:
        return {
            "closed": False,
            "partially_closed": False,
            "matched_windows": 0,
            "closed_windows": 0,
            "close_strategy": "ax_helper_unavailable",
            "status": "AX_HELPER_UNAVAILABLE",
            "return_code": 1,
            "title_hint": title_hint,
            "exact_title": exact_title,
            "window_id": window_id,
            "error": str(exc),
        }

    settle_ms = max(0, min(2000, int(round(max(0.0, retry_delay) * 1000.0))))
    command = [
        str(helper_path),
        title_hint,
        exact_title or "",
        str(window_id) if isinstance(window_id, int) and window_id > 0 else "",
        str(max(1, min(15, int(attempts)))),
        str(settle_ms),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    helper_payload: dict[str, Any] = {}
    for raw in reversed(proc.stdout.splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            helper_payload = candidate
            break

    status = str(helper_payload.get("status") or ("OK" if proc.returncode == 0 else "AX_HELPER_FAILED"))
    matched_windows = int(helper_payload.get("matched_windows") or 0)
    closed_windows = int(helper_payload.get("closed_windows") or 0)
    remaining_windows = int(helper_payload.get("remaining_windows") or 0)
    closed = bool(helper_payload.get("closed"))
    partially_closed = bool(helper_payload.get("partially_closed"))
    if not partially_closed:
        partially_closed = (not closed) and closed_windows > 0

    return {
        "closed": closed,
        "partially_closed": partially_closed,
        "matched_windows": matched_windows,
        "closed_windows": closed_windows,
        "remaining_windows": remaining_windows,
        "close_strategy": str(helper_payload.get("close_strategy") or "ax_helper"),
        "status": status,
        "return_code": proc.returncode,
        "title_hint": title_hint,
        "exact_title": exact_title,
        "window_id": window_id,
        "command": command,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "attempt_count": helper_payload.get("attempt_count"),
        "attempts": helper_payload.get("attempts"),
        "helper_details": helper_details,
    }


def _probe_ltspice_window_matches(
    *,
    title_hint: str,
    exact_title: str | None,
    window_id: int | None,
) -> dict[str, Any]:
    def _escape_applescript_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    escaped_contains = _escape_applescript_string(title_hint.strip())
    escaped_exact = _escape_applescript_string((exact_title or "").strip())
    escaped_window_id = str(window_id) if isinstance(window_id, int) and window_id > 0 else ""
    script = (
        'set AppleScript\'s text item delimiters to ""\n'
        "on joinItems(theList, delim)\n"
        '  set AppleScript\'s text item delimiters to delim\n'
        '  set outText to theList as text\n'
        '  set AppleScript\'s text item delimiters to ""\n'
        "  return outText\n"
        "end joinItems\n"
        f'set targetContainsName to "{escaped_contains}"\n'
        f'set targetExactName to "{escaped_exact}"\n'
        f'set targetWindowId to "{escaped_window_id}"\n'
        "try\n"
        '  tell application "System Events"\n'
        '    if not (exists process "LTspice") then return "PROCESS_MISSING|0|"\n'
        '    tell process "LTspice"\n'
        '      set matchIds to {}\n'
        "      repeat with w in windows\n"
        '        set windowName to ""\n'
        "        try\n"
        "          set windowName to (name of w) as text\n"
        "        end try\n"
        '        set windowNumber to ""\n'
        "        try\n"
        '          set windowNumber to (value of attribute "AXWindowNumber" of w) as text\n'
        "        end try\n"
        "        set isMatch to false\n"
        '        if targetWindowId is not "" and windowNumber is targetWindowId then set isMatch to true\n'
        '        if (not isMatch) and targetExactName is not "" and windowName is targetExactName then set isMatch to true\n'
        '        if (not isMatch) and targetContainsName is not "" and windowName contains targetContainsName then set isMatch to true\n'
        "        if isMatch then\n"
        '          if windowNumber is "" then\n'
        '            set end of matchIds to "unknown"\n'
        "          else\n"
        "            set end of matchIds to windowNumber\n"
        "          end if\n"
        "        end if\n"
        "      end repeat\n"
        '      return "OK|" & ((count of matchIds) as text) & "|" & my joinItems(matchIds, ",")\n'
        "    end tell\n"
        "  end tell\n"
        "on error errMsg number errNum\n"
        '  return "ERROR|" & (errNum as text) & "|" & errMsg\n'
        "end try"
    )

    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )

    status = "UNKNOWN"
    matched_windows = 0
    matching_ids: list[int | str] = []
    message = ""
    for raw_line in reversed(proc.stdout.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) >= 2 and parts[0] in {"OK", "PROCESS_MISSING", "ERROR"}:
            status = parts[0]
            try:
                matched_windows = int(parts[1])
            except Exception:  # noqa: BLE001
                matched_windows = 0
            if len(parts) >= 3 and parts[2]:
                if status == "OK":
                    for token in parts[2].split(","):
                        item = token.strip()
                        if not item:
                            continue
                        if item.isdigit():
                            matching_ids.append(int(item))
                        else:
                            matching_ids.append(item)
                else:
                    message = parts[2]
            break

    if status == "UNKNOWN" and proc.returncode == 0:
        status = "OK"
    if not message and status == "ERROR":
        message = proc.stderr.strip() or "Window verification failed."
    stderr_lower = proc.stderr.lower()
    access_denied = "assistive access" in stderr_lower or "(-25211)" in stderr_lower
    verification_available = status in {"OK", "PROCESS_MISSING"} and proc.returncode == 0
    verified_closed = status == "PROCESS_MISSING" or (verification_available and matched_windows == 0)
    return {
        "status": status,
        "verified_closed": bool(verified_closed),
        "verification_available": bool(verification_available),
        "matched_windows": matched_windows,
        "matching_window_ids": matching_ids,
        "window_id": window_id,
        "title_hint": title_hint,
        "exact_title": exact_title,
        "return_code": proc.returncode,
        "command": ["osascript", "-e", script],
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "access_denied": access_denied,
        "message": message,
    }


def _apply_post_close_verification(
    event: dict[str, Any],
    *,
    title_hint: str,
    exact_title: str | None,
    window_id: int | None,
) -> dict[str, Any]:
    verify_enabled = os.getenv("LTSPICE_MCP_VERIFY_WINDOW_CLOSE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not verify_enabled:
        event["post_verify"] = {"verification_available": False, "status": "DISABLED"}
        return event
    try:
        post_verify = _probe_ltspice_window_matches(
            title_hint=title_hint,
            exact_title=exact_title,
            window_id=window_id,
        )
    except Exception as exc:  # noqa: BLE001
        event["post_verify"] = {
            "verification_available": False,
            "status": "ERROR",
            "error": str(exc),
        }
        return event

    event["post_verify"] = post_verify
    if not bool(post_verify.get("verification_available")):
        return event

    closed_windows = int(event.get("closed_windows") or 0)
    matched_windows = int(event.get("matched_windows") or 0)
    closed_before_verify = bool(event.get("closed", False)) and closed_windows > 0
    verified_closed = bool(post_verify.get("verified_closed"))
    if matched_windows <= 0 and closed_windows <= 0:
        verified_closed = False
    event["closed_before_verify"] = closed_before_verify
    event["verified_closed"] = verified_closed
    event["verification_mismatch"] = closed_before_verify != verified_closed
    event["closed"] = bool(closed_windows > 0 and verified_closed)
    if event["closed"]:
        event["partially_closed"] = False
    elif closed_windows > 0:
        event["partially_closed"] = True
    else:
        event["partially_closed"] = False
    return event


def close_ltspice_window(
    title_hint: str,
    *,
    window_id: int | None = None,
    exact_title: str | None = None,
    attempts: int = 1,
    retry_delay: float = 0.15,
) -> dict[str, Any]:
    title_contains = title_hint.strip()
    title_exact = (exact_title or "").strip()
    target_window_id = window_id if isinstance(window_id, int) and window_id > 0 else None
    if not title_contains and not title_exact and target_window_id is None:
        return {
            "closed": False,
            "return_code": 1,
            "title_hint": title_hint,
            "exact_title": exact_title,
            "window_id": window_id,
            "error": "Provide at least one selector (title_hint, exact_title, or positive window_id)",
        }
    if platform.system() != "Darwin":
        raise RuntimeError("LTspice UI integration is currently implemented for macOS only.")

    def _escape_applescript_string(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    helper_event = _close_ltspice_window_with_ax_helper(
        title_hint=title_contains,
        exact_title=title_exact or None,
        window_id=target_window_id,
        attempts=attempts,
        retry_delay=retry_delay,
    )
    if helper_event is not None:
        helper_status = str(helper_event.get("status") or "")
        # Keep AX helper as the default path to avoid Space-dependent System Events behavior.
        if helper_status != "AX_HELPER_UNAVAILABLE":
            return _apply_post_close_verification(
                helper_event,
                title_hint=title_contains,
                exact_title=title_exact or None,
                window_id=target_window_id,
            )

    escaped_contains = _escape_applescript_string(title_contains)
    escaped_exact = _escape_applescript_string(title_exact)
    escaped_window_id = str(target_window_id) if target_window_id is not None else ""
    script = (
        'tell application "System Events"\n'
        '  if not (exists process "LTspice") then return "PROCESS_MISSING|0|0"\n'
        f'  set targetContainsName to "{escaped_contains}"\n'
        f'  set targetExactName to "{escaped_exact}"\n'
        f'  set targetWindowId to "{escaped_window_id}"\n'
        '  tell process "LTspice"\n'
        '    set matches to {}\n'
        '    repeat with w in windows\n'
        '      set windowName to ""\n'
        '      try\n'
        '        set windowName to (name of w) as text\n'
        '      end try\n'
        '      set windowNumber to ""\n'
        '      try\n'
        '        set windowNumber to (value of attribute "AXWindowNumber" of w) as text\n'
        '      end try\n'
        '      set isMatch to false\n'
        '      if targetWindowId is not "" and windowNumber is targetWindowId then set isMatch to true\n'
        '      if (not isMatch) and targetExactName is not "" and windowName is targetExactName then set isMatch to true\n'
        '      if (not isMatch) and targetContainsName is not "" and windowName contains targetContainsName then set isMatch to true\n'
        '      if isMatch then set end of matches to w\n'
        '    end repeat\n'
        '    set matchCount to (count of matches)\n'
        '    set closeCount to 0\n'
        '    repeat with w in matches\n'
        '      set didClose to false\n'
        '      try\n'
        '        click (first button of w whose subrole is "AXCloseButton")\n'
        '        set didClose to true\n'
        '      on error\n'
        '        try\n'
        '          perform action "AXPress" of (first button of w whose subrole is "AXCloseButton")\n'
        '          set didClose to true\n'
        '        on error\n'
        '          try\n'
        '            perform action "AXClose" of w\n'
        '            set didClose to true\n'
        '          end try\n'
        '        end try\n'
        '      end try\n'
        '      if didClose then set closeCount to closeCount + 1\n'
        '    end repeat\n'
        '    return "OK|" & (matchCount as text) & "|" & (closeCount as text) & "|ax"\n'
        '  end tell\n'
        'end tell'
    )
    max_attempts = max(1, min(10, int(attempts)))
    attempt_delay = max(0.0, float(retry_delay))
    attempt_events: list[dict[str, Any]] = []
    for attempt_index in range(max_attempts):
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        status = "UNKNOWN"
        matched_windows = 0
        closed_windows = 0
        close_strategy = None
        for raw_line in reversed(proc.stdout.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                status = parts[0]
                matched_windows = int(parts[1])
                closed_windows = int(parts[2])
                if len(parts) >= 4 and parts[3]:
                    close_strategy = parts[3]
                break
        if status == "UNKNOWN" and proc.returncode == 0:
            status = "OK"

        all_matches_closed = matched_windows > 0 and closed_windows == matched_windows
        event: dict[str, Any] = {
            "closed": proc.returncode == 0 and all_matches_closed,
            "partially_closed": proc.returncode == 0 and 0 < closed_windows < matched_windows,
            "matched_windows": matched_windows,
            "closed_windows": closed_windows,
            "close_strategy": close_strategy,
            "status": status,
            "return_code": proc.returncode,
            "title_hint": title_hint,
            "exact_title": title_exact or None,
            "window_id": target_window_id,
            "command": ["osascript", "-e", script],
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "attempt": attempt_index + 1,
        }
        attempt_events.append(event)

        if event["closed"]:
            break

        stderr_lower = str(event.get("stderr", "")).lower()
        if "assistive access" in stderr_lower or "(-25211)" in stderr_lower:
            break

        if attempt_index + 1 < max_attempts and attempt_delay > 0:
            time.sleep(attempt_delay)

    result = attempt_events[-1]
    result["attempt_count"] = len(attempt_events)
    if len(attempt_events) > 1:
        result["attempts"] = attempt_events
    if helper_event is not None:
        result["ax_helper_event"] = helper_event
    return _apply_post_close_verification(
        result,
        title_hint=title_contains,
        exact_title=title_exact or None,
        window_id=target_window_id,
    )


def read_ltspice_window_text(
    *,
    title_hint: str = "",
    exact_title: str | None = None,
    window_id: int | None = None,
    max_chars: int = 200000,
) -> dict[str, Any]:
    title_contains = title_hint.strip()
    title_exact = (exact_title or "").strip()
    target_window_id = window_id if isinstance(window_id, int) and window_id > 0 else None
    if not title_contains and not title_exact and target_window_id is None:
        return {
            "ok": False,
            "status": "INVALID_SELECTORS",
            "error": "Provide at least one selector (title_hint, exact_title, or positive window_id)",
            "text": "",
            "matched_windows": 0,
        }
    if platform.system() != "Darwin":
        raise RuntimeError("LTspice UI integration is currently implemented for macOS only.")

    safe_max_chars = max(512, min(2_000_000, int(max_chars)))
    try:
        helper_path, helper_details = _ensure_ax_text_helper()
    except Exception as exc:
        return {
            "ok": False,
            "status": "AX_HELPER_UNAVAILABLE",
            "error": str(exc),
            "text": "",
            "matched_windows": 0,
        }

    command = [
        str(helper_path),
        title_contains,
        title_exact,
        str(target_window_id) if target_window_id is not None else "",
        str(safe_max_chars),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    helper_payload: dict[str, Any] = {}
    for raw in reversed(proc.stdout.splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            helper_payload = candidate
            break

    status = str(helper_payload.get("status") or ("OK" if proc.returncode == 0 else "AX_TEXT_HELPER_FAILED"))
    text_value = str(helper_payload.get("text") or "")
    if len(text_value) > safe_max_chars:
        text_value = text_value[:safe_max_chars]

    return {
        "ok": bool(proc.returncode == 0 and status == "OK" and text_value),
        "status": status,
        "text": text_value,
        "text_length": len(text_value),
        "matched_windows": int(helper_payload.get("matched_windows") or 0),
        "window_title": helper_payload.get("window_title"),
        "window_id": helper_payload.get("window_id"),
        "chunk_count": helper_payload.get("chunk_count"),
        "candidate_titles": helper_payload.get("candidate_titles"),
        "title_hint": title_contains,
        "exact_title": title_exact or None,
        "max_chars": safe_max_chars,
        "return_code": proc.returncode,
        "command": command,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "helper_details": helper_details,
    }


def _capture_ltspice_window_with_screencapturekit(
    *,
    output_path: Path,
    title_hint: str | None = None,
    timeout_seconds: float = 20.0,
    attempts: int = 3,
    retry_delay: float = 0.4,
    capture_id: str | None = None,
) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise RuntimeError("ScreenCaptureKit capture is currently implemented for macOS only.")
    helper_path, helper_details = _ensure_screencapturekit_helper()
    max_attempts = max(1, min(5, int(attempts)))
    delay_seconds = max(0.0, float(retry_delay))
    total_timeout = max(3.0, float(timeout_seconds))
    per_attempt_timeout = max(3.0, min(total_timeout, (total_timeout / max_attempts) + 2.0))
    helper_timeout = max(1.5, min(15.0, per_attempt_timeout - 1.0))
    attempt_events: list[dict[str, Any]] = []
    last_error_message = "unknown ScreenCaptureKit failure"

    def _extract_payload(stdout: str) -> dict[str, Any]:
        for raw in reversed(stdout.splitlines()):
            line = raw.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
        return {}

    def _is_retryable_sck_error(message: str) -> bool:
        lowered = message.lower()
        markers = (
            "timed out waiting for first stream frame",
            "no ltspice windows found",
            "selected content was unavailable",
            "stream failed with",
        )
        return any(marker in lowered for marker in markers)

    for attempt_index in range(max_attempts):
        started_monotonic = time.monotonic()
        command = [
            str(helper_path),
            str(output_path),
            (title_hint or ""),
            f"{helper_timeout:.3f}",
        ]
        _log_capture_event(
            logging.INFO,
            "sck_attempt_start",
            capture_id=capture_id,
            attempt=attempt_index + 1,
            max_attempts=max_attempts,
            helper_path=str(helper_path),
            title_hint=title_hint,
            output_path=str(output_path),
            per_attempt_timeout=per_attempt_timeout,
            helper_timeout=helper_timeout,
        )
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=per_attempt_timeout,
            )
        except subprocess.TimeoutExpired:
            duration_seconds = round(time.monotonic() - started_monotonic, 6)
            last_error_message = (
                f"helper process timed out after {per_attempt_timeout:.3f}s "
                f"(attempt {attempt_index + 1}/{max_attempts})"
            )
            _log_capture_event(
                logging.WARNING,
                "sck_attempt_timeout",
                capture_id=capture_id,
                attempt=attempt_index + 1,
                max_attempts=max_attempts,
                duration_seconds=duration_seconds,
                error=last_error_message,
            )
            attempt_events.append(
                {
                    "attempt": attempt_index + 1,
                    "return_code": None,
                    "duration_seconds": duration_seconds,
                    "error": last_error_message,
                    "timed_out": True,
                }
            )
            if attempt_index + 1 < max_attempts and delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        duration_seconds = round(time.monotonic() - started_monotonic, 6)
        payload = _extract_payload(proc.stdout)
        message = (proc.stderr.strip() or proc.stdout.strip() or "unknown ScreenCaptureKit failure")
        last_error_message = message

        attempt_events.append(
            {
                "attempt": attempt_index + 1,
                "return_code": proc.returncode,
                "duration_seconds": duration_seconds,
                "stdout_line_count": len([line for line in proc.stdout.splitlines() if line.strip()]),
                "stderr": proc.stderr.strip(),
                "error": "" if proc.returncode == 0 else message,
            }
        )

        if proc.returncode == 0:
            # ScreenCaptureKit finalization can lag slightly after helper exit.
            file_ready = output_path.exists()
            if not file_ready:
                time.sleep(0.2)
                file_ready = output_path.exists()
            if file_ready:
                diagnostics = {
                    "helper_command": command,
                    "duration_seconds": duration_seconds,
                    "return_code": proc.returncode,
                    "stdout_line_count": len([line for line in proc.stdout.splitlines() if line.strip()]),
                    "stderr": proc.stderr.strip(),
                    "helper_details": helper_details,
                    "attempt_count": len(attempt_events),
                    "attempts": attempt_events,
                }
                if payload:
                    _log_capture_event(
                        logging.INFO,
                        "sck_capture_success",
                        capture_id=capture_id,
                        attempt=attempt_index + 1,
                        duration_seconds=duration_seconds,
                        output_path=str(output_path),
                        window_id=payload.get("window_id"),
                        window_title=payload.get("window_title"),
                    )
                    payload["capture_diagnostics"] = diagnostics
                    return payload
                _log_capture_event(
                    logging.INFO,
                    "sck_capture_success",
                    capture_id=capture_id,
                    attempt=attempt_index + 1,
                    duration_seconds=duration_seconds,
                    output_path=str(output_path),
                )
                return {
                    "capture_mode": "screencapturekit_window",
                    "capture_diagnostics": diagnostics,
                }

            last_error_message = (
                "ScreenCaptureKit helper reported success but did not produce an image file: "
                f"{output_path}"
            )
            attempt_events[-1]["error"] = last_error_message
            _log_capture_event(
                logging.WARNING,
                "sck_attempt_missing_file",
                capture_id=capture_id,
                attempt=attempt_index + 1,
                duration_seconds=duration_seconds,
                error=last_error_message,
            )
            if attempt_index + 1 < max_attempts and delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        retryable = _is_retryable_sck_error(message)
        _log_capture_event(
            logging.WARNING,
            "sck_attempt_failed",
            capture_id=capture_id,
            attempt=attempt_index + 1,
            max_attempts=max_attempts,
            duration_seconds=duration_seconds,
            return_code=proc.returncode,
            retryable=retryable,
            error=message,
        )
        if attempt_index + 1 < max_attempts and retryable:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            continue
        break

    _log_capture_event(
        logging.ERROR,
        "sck_capture_failed",
        capture_id=capture_id,
        attempts=len(attempt_events),
        max_attempts=max_attempts,
        error=last_error_message,
        helper_details=helper_details,
    )
    raise RuntimeError(
        "ScreenCaptureKit capture failed after "
        f"{len(attempt_events)} attempt(s): {last_error_message}; "
        f"diagnostics={json.dumps({'capture_id': capture_id, 'attempts': attempt_events, 'helper_details': helper_details})}"
    )


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
    capture_id = uuid4().hex[:12]
    preflight = {
        "platform": platform.system(),
        "prefer_screencapturekit": bool(prefer_screencapturekit),
        "avoid_space_switch": bool(avoid_space_switch),
        "open_path": str(Path(open_path).expanduser().resolve()) if open_path is not None else None,
        "title_hint_requested": title_hint,
        "swiftc_available": shutil.which("swiftc"),
        "sck_helper_path_env": os.getenv("LTSPICE_MCP_SCK_HELPER_PATH"),
        "sck_helper_dir_env": os.getenv("LTSPICE_MCP_SCK_HELPER_DIR"),
        "ltspice_ui_running_before_open": is_ltspice_ui_running(),
    }
    _log_capture_event(
        logging.INFO,
        "capture_start",
        capture_id=capture_id,
        output_path=str(target),
        open_path=preflight["open_path"],
        title_hint=title_hint,
        prefer_screencapturekit=bool(prefer_screencapturekit),
        avoid_space_switch=bool(avoid_space_switch),
        settle_seconds=float(settle_seconds),
    )

    open_event: dict[str, Any] | None = None
    close_event: dict[str, Any] | None = None
    opened_window = False
    if open_path is not None:
        open_event = open_in_ltspice_ui(
            open_path,
            background=avoid_space_switch,
        )
        if not open_event.get("opened", False):
            _log_capture_event(
                logging.ERROR,
                "capture_open_failed",
                capture_id=capture_id,
                open_event=open_event,
            )
            raise RuntimeError(
                f"Failed to open LTspice UI target (capture_id={capture_id}): {open_event}"
            )
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
                    capture_id=capture_id,
                )
                capture_backend = "screencapturekit"
                raw_window_id = window_info.get("window_id")
                if isinstance(raw_window_id, int):
                    window_id = raw_window_id
            except Exception as exc:
                elapsed = round(time.monotonic() - started_monotonic, 6)
                _log_capture_event(
                    logging.ERROR,
                    "capture_screencapturekit_failed",
                    capture_id=capture_id,
                    error=str(exc),
                    elapsed_seconds=elapsed,
                    preflight=preflight,
                )
                raise RuntimeError(
                    "ScreenCaptureKit capture failed: "
                    f"{exc}; diagnostics={json.dumps({'capture_id': capture_id, 'elapsed_seconds': elapsed, 'preflight': preflight})}"
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
                _log_capture_event(
                    logging.ERROR,
                    "capture_screencapture_failed",
                    capture_id=capture_id,
                    return_code=capture.returncode,
                    stderr=capture.stderr.strip(),
                )
                raise RuntimeError(
                    f"screencapture failed (capture_id={capture_id}, rc={capture.returncode}): {capture.stderr.strip()}"
                )
            capture_stderr = capture.stderr.strip()
        else:
            capture = None
    finally:
        if opened_window and close_after_capture:
            close_title = title_hint or (Path(open_path).name if open_path is not None else "")
            close_kwargs: dict[str, Any] = {}
            if window_id is not None:
                close_kwargs["window_id"] = window_id
            capture_window_title = window_info.get("window_title")
            if isinstance(capture_window_title, str) and capture_window_title.strip():
                close_kwargs["exact_title"] = capture_window_title.strip()
            try:
                close_event = close_ltspice_window(
                    close_title,
                    attempts=5,
                    retry_delay=0.2,
                    **close_kwargs,
                )
            except Exception as exc:
                close_event = {
                    "closed": False,
                    "return_code": 1,
                    "title_hint": close_title,
                    "exact_title": close_kwargs.get("exact_title"),
                    "window_id": close_kwargs.get("window_id"),
                    "error": str(exc),
                }
            if close_event and not close_event.get("closed", False):
                _log_capture_event(
                    logging.WARNING,
                    "capture_close_incomplete",
                    capture_id=capture_id,
                    close_event=close_event,
                )
            if close_event and close_event.get("verification_mismatch"):
                _log_capture_event(
                    logging.WARNING,
                    "capture_close_verification_mismatch",
                    capture_id=capture_id,
                    close_event=close_event,
                )

    if not target.exists():
        _log_capture_event(
            logging.ERROR,
            "capture_file_missing",
            capture_id=capture_id,
            output_path=str(target),
            backend=capture_backend,
            window_info=window_info,
            open_event=open_event,
            close_event=close_event,
        )
        raise FileNotFoundError(
            f"Screenshot capture did not produce file (capture_id={capture_id}): {target}"
        )

    downscale_info = _downscale_image_file(target, downscale_factor=downscale_factor)
    width, height = _probe_image_dimensions(target)
    elapsed = round(time.monotonic() - started_monotonic, 6)
    _log_capture_event(
        logging.INFO,
        "capture_success",
        capture_id=capture_id,
        backend=capture_backend,
        output_path=str(target),
        elapsed_seconds=elapsed,
        width=width,
        height=height,
        window_id=window_id,
    )
    return {
        "capture_id": capture_id,
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
            "capture_id": capture_id,
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
        _purge_previous_simulation_outputs(netlist)
        command = [str(executable), "-b", str(netlist)]
        if ascii_raw:
            command.append("-ascii")

        started_at = datetime.now().astimezone().isoformat()
        start_ts = time.time()
        timed_out = False

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
            timed_out = True
            return_code = -1
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nLTspice timed out after {timeout} seconds."

        duration = time.time() - start_ts
        artifacts = _collect_related_artifacts(netlist)
        output_artifacts = [path for path in artifacts if _is_simulation_output_artifact(netlist, path)]
        fresh_output_artifacts = [
            path for path in output_artifacts if _is_recent_artifact(path, started_ts=start_ts)
        ]
        raw_files = [path for path in fresh_output_artifacts if path.suffix.lower() == ".raw"]

        log_path = _resolve_log_path(netlist)
        if log_path is not None and not _is_recent_artifact(log_path, started_ts=start_ts):
            log_path = None
        log_utf8_path = _write_utf8_log_sidecar(log_path)
        if log_utf8_path is not None:
            artifacts = sorted({*artifacts, log_utf8_path})

        issues, warnings, diagnostics = analyze_log(log_path)
        if timed_out:
            issues.append(f"LTspice timed out after {timeout} seconds.")
            diagnostics.append(
                SimulationDiagnostic(
                    category="timeout",
                    severity="error",
                    message=f"LTspice timed out after {timeout} seconds.",
                    suggestion="Increase timeout_seconds or simplify the simulation setup.",
                )
            )
        elif return_code != 0:
            issues.append(f"LTspice exited with return code {return_code}.")
            diagnostics.append(
                SimulationDiagnostic(
                    category="process_error",
                    severity="error",
                    message=f"LTspice exited with return code {return_code}.",
                    suggestion="Check stderr output and LTspice log details for the underlying simulation failure.",
                )
            )
            if not fresh_output_artifacts:
                stale_message = (
                    "Simulation artifacts were not regenerated for this run; refusing to reuse stale .log/.raw files."
                )
                issues.append(stale_message)
                diagnostics.append(
                    SimulationDiagnostic(
                        category="artifact_stale_or_missing",
                        severity="error",
                        message=stale_message,
                        suggestion=(
                            "Check LTspice command-line arguments and verify the netlist path is valid. "
                            "No fresh .log/.raw outputs were detected."
                        ),
                    )
                )
            if ascii_raw and log_path is None:
                issues.append(
                    "No .log file was generated in -ascii mode; retry with ascii_raw=false to obtain diagnostics."
                )
                diagnostics.append(
                    SimulationDiagnostic(
                        category="ascii_raw_mode",
                        severity="warning",
                        message="Simulation failed in -ascii mode without a log file.",
                        suggestion="Retry with ascii_raw=false. Some LTspice/macOS runs fail early in -ascii mode.",
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
            log_utf8_path=log_utf8_path,
            raw_files=raw_files,
            artifacts=artifacts,
            issues=issues,
            warnings=warnings,
            diagnostics=diagnostics,
        )
