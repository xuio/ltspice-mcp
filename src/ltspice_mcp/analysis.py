from __future__ import annotations

from bisect import bisect_left
from math import atan2, degrees
from typing import Any


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
