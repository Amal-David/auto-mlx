"""Declarative candidate providers with no code or command execution surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Final, Protocol, runtime_checkable

from .canonical import canonical_json, strict_json_loads
from .contracts import CandidateProposal, FrozenWorkload, validate_config, _freeze_config
from .errors import ContractError, FailureCode, UnknownFieldError


MAX_PROVIDER_CONFIGS: Final = 4096


@runtime_checkable
class CandidateProvider(Protocol):
    """The only provider capability: turn a frozen workload into typed proposals."""

    @property
    def provider_id(self) -> str: ...

    def propose(self, workload: FrozenWorkload) -> tuple[CandidateProposal, ...]: ...


@dataclass(frozen=True, slots=True)
class DeclarativeProvider:
    """A provider backed solely by immutable scalar configuration maps."""

    provider_id: str
    configs: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if type(self.provider_id) is not str or not self.provider_id:
            raise ContractError("provider_id must be a non-empty string", code=FailureCode.PROVIDER_ERROR)
        if any(0xD800 <= ord(character) <= 0xDFFF for character in self.provider_id):
            raise ContractError("provider_id must not contain unpaired surrogates", code=FailureCode.INVALID_UNICODE)
        if type(self.configs) not in {list, tuple}:
            raise ContractError("provider.configs must be an array", code=FailureCode.PROVIDER_ERROR)
        if len(self.configs) > MAX_PROVIDER_CONFIGS:
            raise ContractError(
                f"provider.configs cannot contain more than {MAX_PROVIDER_CONFIGS} entries",
                code=FailureCode.PROVIDER_ERROR,
            )
        frozen_configs = []
        seen: set[str] = set()
        for config in self.configs:
            frozen = _freeze_config(config)
            identity = canonical_json(frozen)
            if identity in seen:
                raise ContractError(
                    "provider.configs contains duplicate normalized configurations",
                    code=FailureCode.CONFIG_MISMATCH,
                )
            seen.add(identity)
            frozen_configs.append(frozen)
        object.__setattr__(self, "configs", tuple(frozen_configs))

    @property
    def candidate_configs(self) -> tuple[Mapping[str, object], ...]:
        return self.configs

    def propose(self, workload: FrozenWorkload) -> tuple[CandidateProposal, ...]:
        if not isinstance(workload, FrozenWorkload):
            raise ContractError("provider requires a FrozenWorkload", code=FailureCode.PROVIDER_ERROR)
        proposals = []
        for config in self.configs:
            validated = validate_config(workload, config)
            proposals.append(CandidateProposal(self.provider_id, workload, validated))
        return tuple(proposals)

    def proposals(self, workload: FrozenWorkload) -> tuple[CandidateProposal, ...]:
        return self.propose(workload)

    def to_dict(self) -> dict[str, object]:
        return {"provider_id": self.provider_id, "configs": [dict(config) for config in self.configs]}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "DeclarativeProvider":
        if type(value) is not dict:
            raise ContractError("provider must be a JSON object", code=FailureCode.WRONG_TYPE)
        expected = {"provider_id", "configs"}
        if any(type(key) is not str for key in value):
            raise ContractError("provider field names must be strings", code=FailureCode.WRONG_TYPE)
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown:
            raise UnknownFieldError(f"provider has unknown field(s): {', '.join(sorted(unknown))}")
        if missing:
            raise ContractError(f"provider is missing field(s): {', '.join(sorted(missing))}", code=FailureCode.INVALID_VALUE)
        configs = value["configs"]
        if type(configs) is not list:
            raise ContractError("provider.configs must be an array", code=FailureCode.WRONG_TYPE)
        return cls(value["provider_id"], tuple(configs))

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "DeclarativeProvider":
        return cls.from_dict(strict_json_loads(value))


__all__: Final = ["CandidateProvider", "DeclarativeProvider", "MAX_PROVIDER_CONFIGS"]
