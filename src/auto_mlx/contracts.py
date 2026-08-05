"""Immutable G0 contracts shared by candidate generation and later lanes."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping

from .canonical import MAX_JSON_DEPTH, canonical_json, sha256_hex, strict_json_loads, validate_json_value
from .errors import ContractError, FailureCode, UnknownFieldError
from .paths import validate_non_negative_int, validate_relative_posix_path, validate_sha256


ConfigValue = str | int | bool
JSONValue = None | str | int | bool | list["JSONValue"] | dict[str, "JSONValue"]
_RESERVED_CONFIG_NAMES = frozenset({"candidate_id"})
MAX_CONFIG_ENTRIES: Final = 64
MAX_WARMUP_RUNS: Final = 100
MAX_MEASUREMENT_RUNS: Final = 100
MAX_POLICY_OUTPUT_BYTES: Final = 8 * 1024 * 1024
MAX_K_REPETITIONS: Final = 10_000
MAX_BOOTSTRAP_RESAMPLES: Final = 1_000_000
MIN_BOOTSTRAP_RESAMPLES: Final = 100
MAX_MIN_EFFECT_BPS: Final = 10_000


def _object(value: Any, *, label: str) -> dict[str, Any]:
    validate_json_value(value)
    if type(value) is not dict:
        raise ContractError(f"{label} must be a JSON object", code=FailureCode.WRONG_TYPE)
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ContractError(f"{label} field names must be strings", code=FailureCode.WRONG_TYPE)
    actual = set(value)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ContractError(
            f"{label} is missing field(s): {', '.join(sorted(missing))}",
            code=FailureCode.INVALID_VALUE,
        )


def _string(value: Any, *, label: str, non_empty: bool = True) -> str:
    if type(value) is not str or (non_empty and not value):
        raise ContractError(f"{label} must be a {'non-empty ' if non_empty else ''}string", code=FailureCode.WRONG_TYPE)
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractError(
            f"{label} must not contain unpaired surrogates",
            code=FailureCode.INVALID_UNICODE,
        )
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
    bound_code: FailureCode = FailureCode.WRONG_TYPE,
) -> int:
    if type(value) is not int:
        raise ContractError(f"{label} must be an integer", code=FailureCode.WRONG_TYPE)
    if minimum is not None and value < minimum:
        raise ContractError(f"{label} must be >= {minimum}", code=bound_code)
    if maximum is not None and value > maximum:
        raise ContractError(f"{label} must be <= {maximum}", code=bound_code)
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be a boolean", code=FailureCode.WRONG_TYPE)
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if type(value) not in {list, tuple}:
        raise ContractError(f"{label} must be an array", code=FailureCode.WRONG_TYPE)
    return tuple(value)


def _freeze_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise ContractError(
            f"JSON exceeds the maximum nesting depth of {MAX_JSON_DEPTH} at {path}",
            code=FailureCode.JSON_TOO_DEEP,
        )
    value_type = type(value)
    if value is None or value_type is int or value_type is bool:
        return value
    if value_type is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ContractError(
                f"JSON contains an unpaired UTF-16 surrogate at {path}",
                code=FailureCode.INVALID_UNICODE,
            )
        return value
    if value_type is float:
        raise ContractError(f"floating-point value at {path} is not allowed", code=FailureCode.FLOAT_NOT_ALLOWED)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ContractError(f"object key at {path} must be a string", code=FailureCode.WRONG_TYPE)
            if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                raise ContractError(
                    f"JSON contains an unpaired UTF-16 surrogate in an object key at {path}",
                    code=FailureCode.INVALID_UNICODE,
                )
            frozen[key] = _freeze_json(item, path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise ContractError(f"unsupported value at {path}: {type(value).__name__}", code=FailureCode.WRONG_TYPE)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_config(value: Any) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise ContractError("config must be an object", code=FailureCode.WRONG_TYPE)
    if len(value) > MAX_CONFIG_ENTRIES:
        raise ContractError(
            f"config has more than {MAX_CONFIG_ENTRIES} entries",
            code=FailureCode.CONFIG_MISMATCH,
        )
    result: dict[str, ConfigValue] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise ContractError("config keys must be strings", code=FailureCode.WRONG_TYPE)
        if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
            raise ContractError(
                "config keys must not contain unpaired surrogates",
                code=FailureCode.INVALID_UNICODE,
            )
        if key in _RESERVED_CONFIG_NAMES:
            raise ContractError("candidate_id is evaluator-derived and cannot be supplied", code=FailureCode.CONFIG_MISMATCH)
        if type(item) not in {str, int, bool}:
            raise ContractError(
                f"config value for {key!r} must be a string, integer, or boolean",
                code=FailureCode.WRONG_TYPE,
            )
        if type(item) is str and any(0xD800 <= ord(character) <= 0xDFFF for character in item):
            raise ContractError(
                "config values must not contain unpaired surrogates",
                code=FailureCode.INVALID_UNICODE,
            )
        result[key] = item
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_relative_posix_path(self.path)
        validate_sha256(self.sha256)
        validate_non_negative_int(self.size_bytes, label="size_bytes")

    @property
    def digest(self) -> str:
        return self.sha256

    @property
    def size(self) -> int:
        return self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def verify(self, root: str) -> None:
        from .paths import verify_artifact

        verify_artifact(root, self)

    @classmethod
    def from_file(cls, root: str, path: str) -> "Artifact":
        from .paths import file_identity

        size_bytes, sha256 = file_identity(root, path)
        return cls(path, sha256, size_bytes)

    @classmethod
    def from_dict(cls, value: Any) -> "Artifact":
        data = _object(value, label="artifact")
        _exact_fields(data, {"path", "sha256", "size_bytes"}, label="artifact")
        return cls(data["path"], data["sha256"], data["size_bytes"])

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "Artifact":
        return cls.from_dict(strict_json_loads(value))


@dataclass(frozen=True, slots=True, init=False)
class Knob:
    """A bounded, typed configuration dimension."""

    name: str
    type: str
    values: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    def __init__(
        self,
        name: str,
        type: str | None = None,
        values: tuple[str, ...] | list[str] = (),
        minimum: int | None = None,
        maximum: int | None = None,
        *,
        kind: str | None = None,
    ) -> None:
        if type is None:
            type = kind
        elif kind is not None and kind != type:
            raise ContractError("knob.type and knob.kind disagree", code=FailureCode.INVALID_KNOB)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        self.__post_init__()

    def __post_init__(self) -> None:
        _string(self.name, label="knob.name")
        if self.name in _RESERVED_CONFIG_NAMES:
            raise ContractError("candidate_id is evaluator-derived and cannot be a knob", code=FailureCode.INVALID_KNOB)
        if type(self.type) is not str or self.type not in {"enum", "integer", "bool"}:
            raise ContractError("knob.type must be enum, integer, or bool", code=FailureCode.INVALID_KNOB)
        if type(self.values) not in {tuple, list}:
            raise ContractError("knob.values must be a tuple or list", code=FailureCode.INVALID_KNOB)
        values = tuple(self.values)
        object.__setattr__(self, "values", values)
        if self.type == "enum":
            if not values or any(type(item) is not str or not item for item in values):
                raise ContractError("enum knobs need non-empty string values", code=FailureCode.INVALID_KNOB)
            if any(any(0xD800 <= ord(character) <= 0xDFFF for character in item) for item in values):
                raise ContractError(
                    "enum knob values must not contain unpaired surrogates",
                    code=FailureCode.INVALID_UNICODE,
                )
            if len(set(values)) != len(values):
                raise ContractError("enum knob values must be unique", code=FailureCode.INVALID_KNOB)
            if self.minimum is not None or self.maximum is not None:
                raise ContractError("enum knobs cannot define integer bounds", code=FailureCode.INVALID_KNOB)
        elif self.type == "integer":
            if values:
                raise ContractError("integer knobs cannot define enum values", code=FailureCode.INVALID_KNOB)
            _integer(self.minimum, label="knob.minimum", bound_code=FailureCode.INVALID_KNOB)
            _integer(self.maximum, label="knob.maximum", bound_code=FailureCode.INVALID_KNOB)
            if self.minimum > self.maximum:
                raise ContractError("knob.minimum cannot exceed knob.maximum", code=FailureCode.INVALID_KNOB)
        else:
            if values or self.minimum is not None or self.maximum is not None:
                raise ContractError("bool knobs cannot define values or bounds", code=FailureCode.INVALID_KNOB)

    @property
    def kind(self) -> str:
        return self.type

    @property
    def allowed_values(self) -> tuple[str, ...]:
        return self.values

    @property
    def min_value(self) -> int | None:
        return self.minimum

    @property
    def max_value(self) -> int | None:
        return self.maximum

    def accepts(self, value: Any) -> bool:
        if self.type == "enum":
            return type(value) is str and value in self.values
        if self.type == "integer":
            return type(value) is int and self.minimum <= value <= self.maximum
        return type(value) is bool

    def validate(self, value: Any) -> ConfigValue:
        if not self.accepts(value):
            raise ContractError(
                f"value for knob {self.name!r} is outside its declared type and bounds",
                code=FailureCode.CONFIG_MISMATCH,
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "values": list(self.values),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "Knob":
        data = _object(value, label="knob")
        _exact_fields(data, {"name", "type", "values", "minimum", "maximum"}, label="knob")
        values = data["values"]
        if type(values) is not list:
            raise ContractError("knob.values must be an array", code=FailureCode.WRONG_TYPE)
        return cls(data["name"], data["type"], tuple(values), data["minimum"], data["maximum"])

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "Knob":
        return cls.from_dict(strict_json_loads(value))


@dataclass(frozen=True, slots=True)
class FrozenWorkload:
    """A complete immutable workload identity, including all declared data."""

    name: str
    artifacts: tuple[Artifact, ...] = ()
    knobs: tuple[Knob, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _string(self.name, label="workload.name")
        artifacts = _sequence(self.artifacts, label="workload.artifacts")
        knobs = _sequence(self.knobs, label="workload.knobs")
        if any(not isinstance(item, Artifact) for item in artifacts):
            raise ContractError("workload.artifacts must contain only Artifact values", code=FailureCode.WRONG_TYPE)
        if any(not isinstance(item, Knob) for item in knobs):
            raise ContractError("workload.knobs must contain only Knob values", code=FailureCode.WRONG_TYPE)
        if len(knobs) > MAX_CONFIG_ENTRIES:
            raise ContractError(
                f"workload.knobs cannot contain more than {MAX_CONFIG_ENTRIES} entries",
                code=FailureCode.CONFIG_MISMATCH,
            )
        if len({item.path for item in artifacts}) != len(artifacts):
            raise ContractError("workload artifact paths must be unique", code=FailureCode.INVALID_VALUE)
        if len({item.name for item in knobs}) != len(knobs):
            raise ContractError("workload knob names must be unique", code=FailureCode.INVALID_KNOB)
        if not isinstance(self.parameters, Mapping):
            raise ContractError("workload.parameters must be an object", code=FailureCode.WRONG_TYPE)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "knobs", knobs)
        object.__setattr__(self, "parameters", _freeze_json(self.parameters, path="workload.parameters"))

    @property
    def workload_id(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "knobs": [item.to_dict() for item in self.knobs],
            "parameters": _thaw_json(self.parameters),
        }

    @property
    def workload_hash(self) -> str:
        return sha256_hex(self.to_dict())

    @property
    def identity(self) -> str:
        return self.workload_hash

    @property
    def hash(self) -> str:
        return self.workload_hash

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "FrozenWorkload":
        data = _object(value, label="workload")
        _exact_fields(data, {"name", "artifacts", "knobs", "parameters"}, label="workload")
        if type(data["artifacts"]) is not list or type(data["knobs"]) is not list:
            raise ContractError("workload artifacts and knobs must be arrays", code=FailureCode.WRONG_TYPE)
        return cls(
            data["name"],
            tuple(Artifact.from_dict(item) for item in data["artifacts"]),
            tuple(Knob.from_dict(item) for item in data["knobs"]),
            data["parameters"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "FrozenWorkload":
        return cls.from_dict(strict_json_loads(value))


def validate_config(workload_or_knobs: FrozenWorkload | tuple[Knob, ...], config: Mapping[str, Any]) -> MappingProxyType:
    """Validate completeness and types before a candidate can exist."""

    if isinstance(workload_or_knobs, FrozenWorkload):
        knobs = workload_or_knobs.knobs
    else:
        knobs = _sequence(workload_or_knobs, label="knobs")
        if any(not isinstance(knob, Knob) for knob in knobs):
            raise ContractError("knobs must contain only Knob values", code=FailureCode.WRONG_TYPE)
    frozen = _freeze_config(config)
    if _RESERVED_CONFIG_NAMES.intersection(frozen):
        raise ContractError("candidate_id is evaluator-derived and cannot be supplied", code=FailureCode.CONFIG_MISMATCH)
    expected = {knob.name for knob in knobs}
    actual = set(frozen)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"undeclared={extra}")
        raise ContractError(f"candidate config does not exactly match declared knobs ({', '.join(detail)})", code=FailureCode.CONFIG_MISMATCH)
    for knob in knobs:
        knob.validate(frozen[knob.name])
    return frozen


@dataclass(frozen=True, slots=True, init=False)
class CandidateProposal:
    """A provider/workload/config tuple whose candidate ID is never caller-selected."""

    provider_id: str
    workload: FrozenWorkload
    config: Mapping[str, ConfigValue]
    workload_hash: str
    candidate_id: str

    def __init__(
        self,
        provider_id: str | None = None,
        workload: FrozenWorkload | None = None,
        config: Mapping[str, Any] | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        if provider_id is None:
            provider_id = provider
        elif provider is not None and provider != provider_id:
            raise ContractError("provider and provider_id disagree", code=FailureCode.INVALID_VALUE)
        _string(provider_id, label="provider_id")
        if not isinstance(workload, FrozenWorkload):
            raise ContractError("workload must be a FrozenWorkload", code=FailureCode.WRONG_TYPE)
        if config is None:
            raise ContractError("config is required", code=FailureCode.WRONG_TYPE)
        frozen_config = validate_config(workload, config)
        workload_hash = workload.workload_hash
        candidate_id = sha256_hex(
            {"provider_id": provider_id, "workload_hash": workload_hash, "config": dict(frozen_config)}
        )
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "workload", workload)
        object.__setattr__(self, "config", frozen_config)
        object.__setattr__(self, "workload_hash", workload_hash)
        object.__setattr__(self, "candidate_id", candidate_id)

    @property
    def provider(self) -> str:
        return self.provider_id

    @property
    def candidate_hash(self) -> str:
        return self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "workload_hash": self.workload_hash,
            "config": dict(self.config),
            "candidate_id": self.candidate_id,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def create(cls, provider_id: str, workload: FrozenWorkload, config: Mapping[str, Any]) -> "CandidateProposal":
        return cls(provider_id, workload, config)

    @classmethod
    def from_dict(cls, value: Any, workload: FrozenWorkload) -> "CandidateProposal":
        data = _object(value, label="candidate proposal")
        _exact_fields(data, {"provider_id", "workload_hash", "config", "candidate_id"}, label="candidate proposal")
        proposal = cls(data["provider_id"], workload, data["config"])
        if data["workload_hash"] != proposal.workload_hash or data["candidate_id"] != proposal.candidate_id:
            raise ContractError("candidate identity does not match provider, workload, and config", code=FailureCode.INVALID_VALUE)
        return proposal

    @classmethod
    def from_json(cls, value: str | bytes | bytearray, workload: FrozenWorkload) -> "CandidateProposal":
        return cls.from_dict(strict_json_loads(value), workload)


THERMAL_GATE_POLICIES: Final = frozenset({"tag", "refuse"})


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    warmup_runs: int = 1
    measurement_runs: int = 3
    timeout_seconds: int = 300
    max_output_bytes: int = 1_048_576
    # Decides what happens when the pre-block thermal preflight (see
    # auto_mlx.thermal) still reports throttled after its one retry:
    # "tag" proceeds with the block, annotating it thermally-suspect in the
    # receipt; "refuse" skips the block's samples outright (they surface as
    # ordinary missing-sample rejections downstream). Default is "tag" --
    # never silently pooling a thermally-suspect block, never refusing by
    # default either.
    thermal_gate_policy: str = "tag"
    # Wave B statistics fields (see docs/measurement.md and
    # auto_mlx.statistics). ``k_repetitions`` is the count of in-runner,
    # eval-fenced timed iterations the runner performs per launch after its
    # one uncounted warmup (see auto_mlx.runners.reference_matmul);
    # ``measurement_runs`` above is the STARTING/minimum paired-block count
    # for sequential sampling, and ``max_measurement_runs`` is the cap it
    # may extend to when the bootstrap verdict stays inconclusive.
    # ``min_effect_bps`` is the minimum-effect promotion threshold in basis
    # points (parts per 10,000) of the baseline point-estimate reference
    # (default 200 = 2.00%; JSON forbids floats in this codebase, hence
    # basis points rather than a fraction). ``bootstrap_resamples`` is the
    # BCa bootstrap resample count. ``calibration`` marks an A/A
    # (candidate == baseline) evaluation; calibration receipts are
    # structurally valid evidence but are never promotable (see
    # auto_mlx.promotion).
    k_repetitions: int = 50
    max_measurement_runs: int = 20
    min_effect_bps: int = 200
    bootstrap_resamples: int = 10_000
    calibration: bool = False

    def __post_init__(self) -> None:
        _integer(self.warmup_runs, label="warmup_runs", minimum=0, maximum=MAX_WARMUP_RUNS, bound_code=FailureCode.INVALID_POLICY)
        _integer(
            self.measurement_runs,
            label="measurement_runs",
            minimum=1,
            maximum=MAX_MEASUREMENT_RUNS,
            bound_code=FailureCode.INVALID_POLICY,
        )
        _integer(self.timeout_seconds, label="timeout_seconds", minimum=1, maximum=3600, bound_code=FailureCode.INVALID_POLICY)
        _integer(
            self.max_output_bytes,
            label="max_output_bytes",
            minimum=1,
            maximum=MAX_POLICY_OUTPUT_BYTES,
            bound_code=FailureCode.INVALID_POLICY,
        )
        if self.thermal_gate_policy not in THERMAL_GATE_POLICIES:
            raise ContractError(
                f"thermal_gate_policy must be one of {sorted(THERMAL_GATE_POLICIES)}",
                code=FailureCode.INVALID_POLICY,
            )
        _integer(self.k_repetitions, label="k_repetitions", minimum=1, maximum=MAX_K_REPETITIONS, bound_code=FailureCode.INVALID_POLICY)
        _integer(
            self.max_measurement_runs,
            label="max_measurement_runs",
            minimum=1,
            maximum=MAX_MEASUREMENT_RUNS,
            bound_code=FailureCode.INVALID_POLICY,
        )
        if self.max_measurement_runs < self.measurement_runs:
            raise ContractError(
                "max_measurement_runs must be >= measurement_runs",
                code=FailureCode.INVALID_POLICY,
            )
        _integer(self.min_effect_bps, label="min_effect_bps", minimum=0, maximum=MAX_MIN_EFFECT_BPS, bound_code=FailureCode.INVALID_POLICY)
        _integer(
            self.bootstrap_resamples,
            label="bootstrap_resamples",
            minimum=MIN_BOOTSTRAP_RESAMPLES,
            maximum=MAX_BOOTSTRAP_RESAMPLES,
            bound_code=FailureCode.INVALID_POLICY,
        )
        if type(self.calibration) is not bool:
            raise ContractError("calibration must be a boolean", code=FailureCode.INVALID_POLICY)

    @property
    def runs(self) -> int:
        return self.measurement_runs

    @property
    def warmups(self) -> int:
        return self.warmup_runs

    @property
    def measurements(self) -> int:
        return self.measurement_runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_runs": self.warmup_runs,
            "measurement_runs": self.measurement_runs,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "thermal_gate_policy": self.thermal_gate_policy,
            "k_repetitions": self.k_repetitions,
            "max_measurement_runs": self.max_measurement_runs,
            "min_effect_bps": self.min_effect_bps,
            "bootstrap_resamples": self.bootstrap_resamples,
            "calibration": self.calibration,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationPolicy":
        data = _object(value, label="evaluation policy")
        _exact_fields(
            data,
            {
                "warmup_runs", "measurement_runs", "timeout_seconds", "max_output_bytes", "thermal_gate_policy",
                "k_repetitions", "max_measurement_runs", "min_effect_bps", "bootstrap_resamples", "calibration",
            },
            label="evaluation policy",
        )
        return cls(
            data["warmup_runs"],
            data["measurement_runs"],
            data["timeout_seconds"],
            data["max_output_bytes"],
            data["thermal_gate_policy"],
            data["k_repetitions"],
            data["max_measurement_runs"],
            data["min_effect_bps"],
            data["bootstrap_resamples"],
            data["calibration"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "EvaluationPolicy":
        return cls.from_dict(strict_json_loads(value))


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    runtime: str
    version: str
    platform: str
    machine: str

    def __post_init__(self) -> None:
        _string(self.runtime, label="runtime")
        _string(self.version, label="version")
        _string(self.platform, label="platform")
        _string(self.machine, label="machine")

    def to_dict(self) -> dict[str, str]:
        return {
            "runtime": self.runtime,
            "version": self.version,
            "platform": self.platform,
            "machine": self.machine,
        }

    @property
    def identity(self) -> str:
        return sha256_hex(self.to_dict())

    @property
    def hash(self) -> str:
        return self.identity

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def current(cls) -> "RuntimeIdentity":
        return cls(
            runtime="python",
            version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.system(),
            machine=platform.machine(),
        )

    from_current = current

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeIdentity":
        data = _object(value, label="runtime identity")
        _exact_fields(data, {"runtime", "version", "platform", "machine"}, label="runtime identity")
        return cls(data["runtime"], data["version"], data["platform"], data["machine"])

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "RuntimeIdentity":
        return cls.from_dict(strict_json_loads(value))


def canonical_contract(value: Any) -> str:
    """Canonicalize an already serialized contract value for receipts."""

    return canonical_json(value)


__all__: Final = [
    "Artifact",
    "CandidateProposal",
    "ConfigValue",
    "EvaluationPolicy",
    "FrozenWorkload",
    "JSONValue",
    "Knob",
    "MAX_CONFIG_ENTRIES",
    "MAX_JSON_DEPTH",
    "MAX_MEASUREMENT_RUNS",
    "MAX_POLICY_OUTPUT_BYTES",
    "MAX_WARMUP_RUNS",
    "RuntimeIdentity",
    "THERMAL_GATE_POLICIES",
    "canonical_contract",
    "validate_config",
]
