"""No-sudo macOS thermal-pressure preflight for measurement blocks.

Apple Silicon offers no frequency lock or core pinning a local evaluator can
reach without elevated privileges.  The next-best honest control is to *know*
when a block was measured under thermal throttling and say so in the
receipt, rather than silently pooling a throttled block's samples with
everything else.

This module never raises on a probing failure: a missing ``pmset`` binary, a
timeout, or output this parser does not recognize all degrade to
``state="unknown"`` -- distinct from ``"nominal"`` (probed, not throttled)
and ``"throttled"`` (probed, throttled).  ``"unknown"`` is never treated as
throttled and never gates a block; it is recorded so the honesty is visible,
not hidden.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Final

_DEFAULT_PMSET_PATH: Final = "pmset"
_DEFAULT_PROBE_TIMEOUT_SECONDS: Final = 2.0
_DEFAULT_RETRY_PAUSE_SECONDS: Final = 30.0
_MAX_DETAIL_CHARS: Final = 2000

_SPEED_LIMIT_RE: Final = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
_SCHEDULER_LIMIT_RE: Final = re.compile(r"CPU_Scheduler_Limit\s*=\s*(\d+)")
_PRESSURE_LEVEL_RE: Final = re.compile(r"Thermal_Pressure_Level\s*=\s*([A-Za-z_]+)")

_NOMINAL: Final = "nominal"
_THROTTLED: Final = "throttled"
_UNKNOWN: Final = "unknown"
_STATES: Final = frozenset({_NOMINAL, _THROTTLED, _UNKNOWN})


@dataclass(frozen=True, slots=True)
class ThermalReading:
    """One point-in-time ``pmset -g therm`` reading, parsed defensively."""

    state: str
    cpu_speed_limit_percent: int | None
    cpu_scheduler_limit_percent: int | None
    thermal_pressure_level: str | None
    detail: str

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise ValueError(f"thermal state must be one of {sorted(_STATES)}, got {self.state!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "cpu_speed_limit_percent": self.cpu_speed_limit_percent,
            "cpu_scheduler_limit_percent": self.cpu_scheduler_limit_percent,
            "thermal_pressure_level": self.thermal_pressure_level,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ThermalReading":
        if type(value) is not dict:
            raise ValueError("thermal reading must be a JSON object")
        expected = {"state", "cpu_speed_limit_percent", "cpu_scheduler_limit_percent", "thermal_pressure_level", "detail"}
        if set(value) != expected:
            raise ValueError(f"thermal reading has unexpected fields: {sorted(set(value) ^ expected)}")
        return cls(
            value["state"],
            value["cpu_speed_limit_percent"],
            value["cpu_scheduler_limit_percent"],
            value["thermal_pressure_level"],
            value["detail"],
        )


def parse_pmset_therm(output: str) -> ThermalReading:
    """Parse ``pmset -g therm`` output into a :class:`ThermalReading`.

    Pure and total: any input, including empty strings, garbage, or a future
    ``pmset`` output format this parser does not recognize, produces a
    well-formed reading (``state="unknown"`` when nothing recognizable is
    present) rather than raising.  Recognizes the two integer percentage
    fields (``CPU_Speed_Limit``, ``CPU_Scheduler_Limit`` -- throttled below
    100) and the categorical ``Thermal_Pressure_Level`` field (throttled
    when present and not ``Nominal``); different Apple Silicon generations
    and macOS versions expose different subsets of these.
    """

    if type(output) is not str:
        output = str(output)
    detail = output.strip()[:_MAX_DETAIL_CHARS]

    speed_match = _SPEED_LIMIT_RE.search(output)
    scheduler_match = _SCHEDULER_LIMIT_RE.search(output)
    pressure_match = _PRESSURE_LEVEL_RE.search(output)

    speed_limit = int(speed_match.group(1)) if speed_match else None
    scheduler_limit = int(scheduler_match.group(1)) if scheduler_match else None
    pressure_level = pressure_match.group(1) if pressure_match else None

    if speed_limit is None and scheduler_limit is None and pressure_level is None:
        return ThermalReading(_UNKNOWN, None, None, None, detail)

    throttled = (
        (speed_limit is not None and speed_limit < 100)
        or (scheduler_limit is not None and scheduler_limit < 100)
        or (pressure_level is not None and pressure_level.lower() != "nominal")
    )
    return ThermalReading(
        _THROTTLED if throttled else _NOMINAL,
        speed_limit,
        scheduler_limit,
        pressure_level,
        detail,
    )


def probe_thermal_pressure(
    *,
    pmset_path: str = _DEFAULT_PMSET_PATH,
    timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> ThermalReading:
    """Run ``pmset -g therm`` and parse it.  Never raises.

    A missing binary, a timeout, or any other ``OSError`` all degrade to a
    ``state="unknown"`` reading rather than propagating -- consistent with
    this module's contract that a thermal preflight failure is diagnostic
    information, never a crash.
    """

    try:
        completed = subprocess.run(
            [pmset_path, "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ThermalReading(_UNKNOWN, None, None, None, f"pmset probe failed: {type(exc).__name__}: {exc}"[:_MAX_DETAIL_CHARS])
    combined = f"{completed.stdout}\n{completed.stderr}"
    return parse_pmset_therm(combined)


def thermal_preflight(
    *,
    prober: Callable[[], ThermalReading] = probe_thermal_pressure,
    sleep: Callable[[float], None] = time.sleep,
    retry_pause_seconds: float = _DEFAULT_RETRY_PAUSE_SECONDS,
) -> dict[str, Any]:
    """Read thermal pressure once; on throttled, wait once and re-read.

    The FINAL reading (post-retry when a retry happened) decides
    ``thermally_suspect``.  ``state="unknown"`` (missing tool, parse
    failure) is never treated as throttled and never triggers a retry --
    "unknown" is honestly recorded, not silently escalated.  Returns a
    plain JSON-safe dict (never a crash) so it can be embedded directly in
    an evaluator bundle / receipt.
    """

    initial = prober()
    retried = False
    final = initial
    if initial.state == _THROTTLED:
        sleep(retry_pause_seconds)
        retried = True
        final = prober()
    return {
        "initial": initial.to_dict(),
        "final": final.to_dict(),
        "retried": retried,
        "thermally_suspect": final.state == _THROTTLED,
    }


__all__: Final = [
    "ThermalReading",
    "parse_pmset_therm",
    "probe_thermal_pressure",
    "thermal_preflight",
]
