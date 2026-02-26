from __future__ import annotations

from bisect import bisect_left
from math import atan2, degrees, log10
from typing import Any


def _is_non_decreasing(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    for idx in range(1, len(values)):
        prev = values[idx - 1]
        curr = values[idx]
        tol = max(1e-30, abs(prev) * 1e-12, abs(curr) * 1e-12)
        if curr < prev - tol:
            return False
    return True


def _ensure_non_decreasing(values: list[float], *, name: str) -> None:
    if not _is_non_decreasing(values):
        raise ValueError(f"{name} must be non-decreasing")


def sample_indices(total_points: int, max_points: int) -> list[int]:
    if total_points <= 0:
        return []
    if max_points <= 0:
        raise ValueError("max_points must be > 0")
    if total_points <= max_points:
        return list(range(total_points))
    if max_points == 1:
        return [0]

    step = (total_points - 1) / (max_points - 1)
    sampled = {round(index * step) for index in range(max_points)}
    sampled.add(0)
    sampled.add(total_points - 1)
    return sorted(sampled)


def interpolate_series(scale: list[float], series: list[complex], points: list[float]) -> list[complex]:
    if len(scale) != len(series):
        raise ValueError("scale and series lengths must match")
    if not scale:
        raise ValueError("cannot interpolate an empty series")
    if len(scale) == 1:
        return [series[0] for _ in points]

    ascending = scale[-1] >= scale[0]
    if not ascending:
        scale = list(reversed(scale))
        series = list(reversed(series))

    min_scale = scale[0]
    max_scale = scale[-1]
    result: list[complex] = []
    for point in points:
        if point < min_scale or point > max_scale:
            raise ValueError(
                f"Point {point} is outside scale range [{min_scale}, {max_scale}]"
            )
        right = bisect_left(scale, point)
        if right == 0:
            result.append(series[0])
            continue
        if right >= len(scale):
            result.append(series[-1])
            continue
        left = right - 1
        x0 = scale[left]
        x1 = scale[right]
        y0 = series[left]
        y1 = series[right]
        if x1 == x0:
            result.append(y0)
            continue
        ratio = (point - x0) / (x1 - x0)
        result.append(y0 + (y1 - y0) * ratio)
    return result


def find_local_extrema(
    scale: list[float],
    series: list[complex],
    *,
    include_minima: bool = True,
    include_maxima: bool = True,
    threshold: float = 0.0,
    max_results: int = 200,
) -> list[dict[str, Any]]:
    if len(scale) != len(series):
        raise ValueError("scale and series lengths must match")
    if len(series) < 3:
        return []
    if not include_minima and not include_maxima:
        return []
    if max_results <= 0:
        return []

    signal = [abs(value) if value.imag else value.real for value in series]
    extrema: list[dict[str, Any]] = []

    for idx in range(1, len(signal) - 1):
        left = signal[idx - 1]
        mid = signal[idx]
        right = signal[idx + 1]

        if include_maxima and mid > left and mid > right:
            prominence = mid - max(left, right)
            if prominence >= threshold:
                extrema.append(
                    {
                        "type": "max",
                        "index": idx,
                        "x": scale[idx],
                        "value": mid,
                        "prominence": prominence,
                    }
                )
                if len(extrema) >= max_results:
                    break

        if include_minima and mid < left and mid < right:
            prominence = min(left, right) - mid
            if prominence >= threshold:
                extrema.append(
                    {
                        "type": "min",
                        "index": idx,
                        "x": scale[idx],
                        "value": mid,
                        "prominence": prominence,
                    }
                )
                if len(extrema) >= max_results:
                    break

    return extrema


def format_series(series: list[complex], representation: str, prefer_real: bool) -> dict[str, Any]:
    if representation not in {"auto", "real", "rectangular", "magnitude-phase", "both"}:
        raise ValueError(
            "representation must be one of: auto, real, rectangular, magnitude-phase, both"
        )

    is_complex = any(abs(value.imag) > 0.0 for value in series)
    if representation == "auto":
        if prefer_real and not is_complex:
            return {"representation": "real", "values": [value.real for value in series]}
        representation = "rectangular" if is_complex else "real"

    if representation == "real":
        return {"representation": "real", "values": [value.real for value in series]}
    if representation == "rectangular":
        return {
            "representation": "rectangular",
            "real": [value.real for value in series],
            "imag": [value.imag for value in series],
        }
    if representation == "magnitude-phase":
        return {
            "representation": "magnitude-phase",
            "magnitude": [abs(value) for value in series],
            "phase_deg": [degrees(atan2(value.imag, value.real)) for value in series],
        }
    return {
        "representation": "both",
        "real": [value.real for value in series],
        "imag": [value.imag for value in series],
        "magnitude": [abs(value) for value in series],
        "phase_deg": [degrees(atan2(value.imag, value.real)) for value in series],
    }


def _interpolate_x_at_y(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return x0
    ratio = (target - y0) / (y1 - y0)
    return x0 + (x1 - x0) * ratio


def _interpolate_y_at_x(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if x1 == x0:
        return y0
    ratio = (target - x0) / (x1 - x0)
    return y0 + (y1 - y0) * ratio


def _find_crossings(
    x: list[float],
    y: list[float],
    target: float,
    *,
    rising: bool | None = None,
) -> list[float]:
    if len(x) != len(y):
        raise ValueError("x and y lengths must match")
    if len(x) < 2:
        return []

    crossings: list[float] = []
    for idx in range(1, len(x)):
        x0 = x[idx - 1]
        x1 = x[idx]
        y0 = y[idx - 1]
        y1 = y[idx]

        crossed = (y0 <= target <= y1) or (y1 <= target <= y0)
        if not crossed:
            continue
        if y1 == y0:
            continue
        if rising is True and not (y1 > y0):
            continue
        if rising is False and not (y1 < y0):
            continue
        crossings.append(_interpolate_x_at_y(x0, y0, x1, y1, target))
    return crossings


def _unwrap_phase_deg(phase: list[float]) -> list[float]:
    if not phase:
        return []
    unwrapped = [phase[0]]
    offset = 0.0
    for idx in range(1, len(phase)):
        delta = phase[idx] - phase[idx - 1]
        if delta > 180.0:
            offset -= 360.0
        elif delta < -180.0:
            offset += 360.0
        unwrapped.append(phase[idx] + offset)
    return unwrapped


def compute_bandwidth(
    frequency_hz: list[float],
    response: list[complex],
    *,
    reference: str = "first",
    drop_db: float = 3.0,
) -> dict[str, Any]:
    if len(frequency_hz) != len(response):
        raise ValueError("frequency_hz and response lengths must match")
    if len(frequency_hz) < 2:
        raise ValueError("Need at least 2 points to compute bandwidth")
    _ensure_non_decreasing(frequency_hz, name="frequency_hz")
    if reference not in {"first", "max"}:
        raise ValueError("reference must be 'first' or 'max'")
    if drop_db <= 0:
        raise ValueError("drop_db must be > 0")

    magnitudes = [abs(value) for value in response]
    reference_mag = magnitudes[0] if reference == "first" else max(magnitudes)
    target_mag = reference_mag / (10 ** (drop_db / 20.0))

    lowpass_crossings = _find_crossings(
        frequency_hz,
        magnitudes,
        target_mag,
        rising=False,
    )
    highpass_crossings = _find_crossings(
        frequency_hz,
        magnitudes,
        target_mag,
        rising=True,
    )

    return {
        "reference_mode": reference,
        "reference_magnitude": reference_mag,
        "drop_db": drop_db,
        "target_magnitude": target_mag,
        "lowpass_bandwidth_hz": lowpass_crossings[0] if lowpass_crossings else None,
        "highpass_bandwidth_hz": highpass_crossings[0] if highpass_crossings else None,
        "all_lowpass_crossings_hz": lowpass_crossings,
        "all_highpass_crossings_hz": highpass_crossings,
    }


def compute_gain_phase_margin(
    frequency_hz: list[float],
    response: list[complex],
) -> dict[str, Any]:
    if len(frequency_hz) != len(response):
        raise ValueError("frequency_hz and response lengths must match")
    if len(frequency_hz) < 2:
        raise ValueError("Need at least 2 points to compute margins")
    _ensure_non_decreasing(frequency_hz, name="frequency_hz")

    magnitudes = [abs(value) for value in response]
    magnitude_db = [20.0 * log10(max(mag, 1e-300)) for mag in magnitudes]
    phase_deg = [degrees(atan2(value.imag, value.real)) for value in response]
    phase_unwrapped = _unwrap_phase_deg(phase_deg)

    gain_crossings = _find_crossings(frequency_hz, magnitude_db, 0.0, rising=None)
    phase_crossings = _find_crossings(frequency_hz, phase_unwrapped, -180.0, rising=None)

    gain_crossover_hz = gain_crossings[0] if gain_crossings else None
    phase_crossover_hz = phase_crossings[0] if phase_crossings else None

    phase_margin_deg = None
    if gain_crossover_hz is not None:
        phase_at_gain = None
        for idx in range(1, len(frequency_hz)):
            x0 = frequency_hz[idx - 1]
            x1 = frequency_hz[idx]
            if (x0 <= gain_crossover_hz <= x1) or (x1 <= gain_crossover_hz <= x0):
                phase_at_gain = _interpolate_y_at_x(
                    x0,
                    phase_unwrapped[idx - 1],
                    x1,
                    phase_unwrapped[idx],
                    gain_crossover_hz,
                )
                break
        if phase_at_gain is not None:
            phase_margin_deg = 180.0 + phase_at_gain

    gain_margin_db = None
    if phase_crossover_hz is not None:
        magnitude_at_phase = None
        for idx in range(1, len(frequency_hz)):
            x0 = frequency_hz[idx - 1]
            x1 = frequency_hz[idx]
            if (x0 <= phase_crossover_hz <= x1) or (x1 <= phase_crossover_hz <= x0):
                magnitude_at_phase = _interpolate_y_at_x(
                    x0,
                    magnitude_db[idx - 1],
                    x1,
                    magnitude_db[idx],
                    phase_crossover_hz,
                )
                break
        if magnitude_at_phase is not None:
            gain_margin_db = -magnitude_at_phase

    return {
        "gain_crossover_hz": gain_crossover_hz,
        "phase_crossover_hz": phase_crossover_hz,
        "phase_margin_deg": phase_margin_deg,
        "gain_margin_db": gain_margin_db,
        "all_gain_crossovers_hz": gain_crossings,
        "all_phase_crossovers_hz": phase_crossings,
    }


def compute_rise_fall_time(
    time_s: list[float],
    signal: list[complex],
    *,
    low_threshold_pct: float = 10.0,
    high_threshold_pct: float = 90.0,
) -> dict[str, Any]:
    if len(time_s) != len(signal):
        raise ValueError("time_s and signal lengths must match")
    if len(time_s) < 2:
        raise ValueError("Need at least 2 points to compute rise/fall times")
    _ensure_non_decreasing(time_s, name="time_s")
    if not (0.0 <= low_threshold_pct < high_threshold_pct <= 100.0):
        raise ValueError("thresholds must satisfy 0 <= low < high <= 100")

    values = [value.real for value in signal]
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    low_threshold = min_value + span * (low_threshold_pct / 100.0)
    high_threshold = min_value + span * (high_threshold_pct / 100.0)

    rising_low = _find_crossings(time_s, values, low_threshold, rising=True)
    rising_high = _find_crossings(time_s, values, high_threshold, rising=True)
    falling_high = _find_crossings(time_s, values, high_threshold, rising=False)
    falling_low = _find_crossings(time_s, values, low_threshold, rising=False)

    rise_time = None
    rise_start = rising_low[0] if rising_low else None
    rise_end = None
    if rise_start is not None:
        rise_end = next((x for x in rising_high if x >= rise_start), None)
        if rise_end is not None:
            rise_time = rise_end - rise_start

    fall_time = None
    fall_start = falling_high[0] if falling_high else None
    fall_end = None
    if fall_start is not None:
        fall_end = next((x for x in falling_low if x >= fall_start), None)
        if fall_end is not None:
            fall_time = fall_end - fall_start

    return {
        "low_threshold_pct": low_threshold_pct,
        "high_threshold_pct": high_threshold_pct,
        "low_threshold_value": low_threshold,
        "high_threshold_value": high_threshold,
        "rise_start_s": rise_start,
        "rise_end_s": rise_end,
        "rise_time_s": rise_time,
        "fall_start_s": fall_start,
        "fall_end_s": fall_end,
        "fall_time_s": fall_time,
    }


def compute_settling_time(
    time_s: list[float],
    signal: list[complex],
    *,
    tolerance_percent: float = 2.0,
    target_value: float | None = None,
) -> dict[str, Any]:
    if len(time_s) != len(signal):
        raise ValueError("time_s and signal lengths must match")
    if len(time_s) < 2:
        raise ValueError("Need at least 2 points to compute settling time")
    _ensure_non_decreasing(time_s, name="time_s")
    if tolerance_percent <= 0:
        raise ValueError("tolerance_percent must be > 0")

    values = [value.real for value in signal]
    final_value = target_value if target_value is not None else values[-1]
    full_scale = max(max(values) - min(values), abs(final_value), 1e-15)
    band = full_scale * (tolerance_percent / 100.0)

    band_error = [abs(value - final_value) - band for value in values]
    outside = [error > 0.0 for error in band_error]
    last_outside_idx = None
    for idx, is_outside in enumerate(outside):
        if is_outside:
            last_outside_idx = idx

    first_inside_idx = next((idx for idx, is_outside in enumerate(outside) if not is_outside), None)

    settling_time = None
    if last_outside_idx is None:
        settling_time = time_s[0]
    elif last_outside_idx + 1 < len(time_s):
        left = last_outside_idx
        right = last_outside_idx + 1
        if outside[left] and not outside[right]:
            settling_time = _interpolate_x_at_y(
                time_s[left],
                band_error[left],
                time_s[right],
                band_error[right],
                0.0,
            )
        else:
            settling_time = time_s[right]

    first_entry_time = None
    if first_inside_idx is not None:
        if first_inside_idx == 0:
            first_entry_time = time_s[0]
        else:
            left = first_inside_idx - 1
            right = first_inside_idx
            if outside[left] and not outside[right]:
                first_entry_time = _interpolate_x_at_y(
                    time_s[left],
                    band_error[left],
                    time_s[right],
                    band_error[right],
                    0.0,
                )
            else:
                first_entry_time = time_s[first_inside_idx]
    last_exit_time = time_s[last_outside_idx] if last_outside_idx is not None else None

    return {
        "tolerance_percent": tolerance_percent,
        "target_value": final_value,
        "tolerance_band": band,
        "settling_time_s": settling_time,
        "first_entry_time_s": first_entry_time,
        "last_exit_time_s": last_exit_time,
        "final_value": values[-1],
    }
