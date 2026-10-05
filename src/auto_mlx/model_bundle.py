"""Offline, content-addressed native MLX-LM model bundles (no ML imports).

An artifact-verified bundle is not a claim of source-framework parity, task
quality, or a speedup. Those require separate evaluator receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import math
from importlib.metadata import PackageNotFoundError, version

from .canonical import canonical_bytes, sha256_hex, strict_json_loads
from .contracts import Artifact, FrozenWorkload, Knob
from .errors import ContractError, FailureCode
from .paths import _open_verified_file

MODEL_TYPES = frozenset({"qwen2", "qwen3", "llama"})
PACKAGES = ("mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "safetensors", "numpy")
MAX_FILES = 128
MAX_MODEL_BYTES = 64 * 1024**3
MAX_METADATA_BYTES = 2 * 1024**2
MAX_PROMPT_BYTES = 65536
BASELINE_PREFILL = 2048
WORKLOAD_NAME = "mlx-lm-text-v1"
_METADATA = frozenset({"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "generation_config.json", "vocab.json", "merges.txt", "tokenizer.model", "model.safetensors.index.json", "README.md", "LICENSE", "LICENSE.txt", ".gitattributes"})
_TOKENIZERS = frozenset({"Qwen2Tokenizer", "Qwen2TokenizerFast", "LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast", "TokenizersBackend"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message, code=FailureCode.INVALID_VALUE)


def exact(value: object, fields: set[str], label: str) -> dict:
    require(type(value) is dict and set(value) == fields, f"{label} must contain exactly {sorted(fields)}")
    return value


def integer(value: object, low: int, high: int, label: str) -> int:
    require(type(value) is int and low <= value <= high, f"{label} must be an integer in [{low}, {high}]")
    return value


def runtime_fingerprint() -> dict[str, str]:
    result = {"python": platform.python_version(), "system": platform.system(), "release": platform.release(), "machine": platform.machine()}
    for key, name in (("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
        result[key] = subprocess.check_output(("/usr/sbin/sysctl", "-n", name), text=True, timeout=2).strip() if platform.system() == "Darwin" else "unavailable"
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate model metadata key")
        result[key] = value
    return result


def read_metadata(root: Path, name: str) -> dict:
    fd = _open_verified_file(str(root), name)
    try:
        require(os.fstat(fd).st_size <= MAX_METADATA_BYTES, f"{name} exceeds metadata byte limit")
        raw = b""
        while len(raw) <= MAX_METADATA_BYTES:
            chunk = os.read(fd, min(65536, MAX_METADATA_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        require(len(raw) <= MAX_METADATA_BYTES, f"{name} exceeds metadata byte limit")
    finally:
        os.close(fd)
    def reject_constant(value: str) -> None:
        require(False, f"non-finite model metadata: {value}")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError(f"invalid {name}: {exc}", code=FailureCode.INVALID_JSON) from exc
    require(type(value) is dict, f"{name} must be a JSON object")
    return value


def _safe_metadata(value: object, depth: int = 0) -> None:
    require(depth < 64, "model metadata nesting exceeds limit")
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite model metadata is forbidden")
    if isinstance(value, dict):
        require(not value.get("auto_map") and not value.get("trust_remote_code"), "remote/custom model code is not supported")
        for key, item in value.items():
            if key.endswith(("_file", "_path")) and isinstance(item, str) and item:
                require(not Path(item).is_absolute() and ".." not in Path(item).parts, "external model metadata paths are forbidden")
            _safe_metadata(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _safe_metadata(item, depth + 1)


def inspect_model_files(root: Path) -> tuple[dict, tuple[Artifact, ...]]:
    require(root.is_dir() and not root.is_symlink(), "model must be a local, non-symlink directory")
    names = []
    total_bytes = 0
    with os.scandir(root) as entries:
        for entry in entries:
            require(len(names) < MAX_FILES, "model has too many files")
            require(entry.is_file(follow_symlinks=False), "model entries must be ordinary files, not links or directories")
            require(entry.name in _METADATA or re.fullmatch(r"model(?:-[0-9]+-of-[0-9]+)?\.safetensors", entry.name) is not None, f"unsupported model file: {entry.name}")
            total_bytes += entry.stat(follow_symlinks=False).st_size
            require(total_bytes <= MAX_MODEL_BYTES, "model exceeds total artifact byte limit")
            names.append(entry.name)
    require({"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(names), "model requires config.json, tokenizer.json, and tokenizer_config.json")
    weights = {name for name in names if name.endswith(".safetensors")}
    require(bool(weights), "model has no safetensors weights")
    config = read_metadata(root, "config.json")
    tokenizer = read_metadata(root, "tokenizer_config.json")
    for value in (config, tokenizer):
        _safe_metadata(value)
    require(type(config.get("model_type")) is str and config.get("model_type") in MODEL_TYPES, f"native adapter supports only {sorted(MODEL_TYPES)}")
    require(not config.get("vision_config") and not config.get("text_config") and not config.get("num_experts"), "multimodal and MoE configurations are not supported")
    require(type(tokenizer.get("tokenizer_class")) is str and tokenizer.get("tokenizer_class") in _TOKENIZERS, "unsupported or custom tokenizer class")
    integer(config.get("max_position_embeddings"), 1, 1048576, "max_position_embeddings")
    if "model.safetensors.index.json" in names:
        index = read_metadata(root, "model.safetensors.index.json")
        mapping = index.get("weight_map")
        require(type(mapping) is dict and bool(mapping), "weight index is missing its map")
        require(all(type(v) is str for v in mapping.values()) and set(mapping.values()) == weights, "weight index must cover exactly the local safetensors shards")
    artifacts = tuple(Artifact.from_file(str(root), name) for name in sorted(names))
    require(sum(a.size_bytes for a in artifacts) <= MAX_MODEL_BYTES, "model exceeds total artifact byte limit")
    return config, artifacts


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Immutable canonical bytes, with all identities derived rather than supplied."""
    payload: bytes

    def __post_init__(self) -> None:
        require(type(self.payload) is bytes and len(self.payload) <= 65536, "bundle must be at most 65536 canonical bytes")
        value = strict_json_loads(self.payload)
        exact(value, {"schema_version", "backend", "device", "source", "model_type", "context_limit", "artifacts", "fixtures", "runtime"}, "bundle")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1, "unsupported bundle schema")
        require(value["backend"] == "mlx-lm" and type(value["model_type"]) is str and value["model_type"] in MODEL_TYPES, "unsupported native backend/model")
        require(type(value["device"]) is str and value["device"] in {"gpu", "cpu"}, "device must be explicitly cpu or gpu")
        source = exact(value["source"], {"id", "revision", "license"}, "source")
        for field in ("id", "license"):
            require(type(source[field]) is str and 0 < len(source[field]) <= 512, f"source.{field} must be explicit")
        require(type(source["revision"]) is str and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source["revision"]) is not None, "source revision must be an immutable commit/content hash")
        integer(value["context_limit"], 1, 1048576, "context_limit")
        require(type(value["artifacts"]) is list and 1 <= len(value["artifacts"]) <= MAX_FILES, "invalid artifact inventory")
        artifacts = [Artifact.from_dict(a) for a in value["artifacts"]]
        names = [a.path for a in artifacts]
        require(names == sorted(set(names)), "artifact inventory must be unique and sorted")
        require({"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(names), "required model artifacts are missing")
        require(any(n.endswith(".safetensors") for n in names), "weight artifact is missing")
        require(sum(a.size_bytes for a in artifacts) <= MAX_MODEL_BYTES, "model exceeds byte limit")
        require(type(value["fixtures"]) is list and 1 <= len(value["fixtures"]) <= 16, "one to sixteen verification fixtures are required")
        for fixture in value["fixtures"]:
            exact(fixture, {"prompt", "max_tokens"}, "fixture")
            require(type(fixture["prompt"]) is str and 0 < len(fixture["prompt"].encode()) <= MAX_PROMPT_BYTES, "invalid fixture prompt")
            integer(fixture["max_tokens"], 1, 256, "fixture.max_tokens")
        runtime = exact(value["runtime"], set(PACKAGES) | {"python", "system", "release", "machine", "chip", "memory_bytes"}, "runtime")
        require(all(type(v) is str and 0 < len(v) <= 256 for v in runtime.values()), "runtime fields must be version strings")
        object.__setattr__(self, "payload", canonical_bytes(value))

    @classmethod
    def from_dict(cls, value: dict) -> "ModelBundle":
        return cls(canonical_bytes(value))

    def to_dict(self) -> dict:
        return strict_json_loads(self.payload)

    @property
    def bundle_id(self) -> str:
        return sha256_hex(self.to_dict())

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return tuple(Artifact.from_dict(a) for a in self.to_dict()["artifacts"])

    def verify(self, root: Path, *, check_runtime: bool = False) -> None:
        config, actual = inspect_model_files(root)
        require(actual == self.artifacts, "model bytes or inventory differ from the bundle")
        require(config["model_type"] == self.to_dict()["model_type"], "model type differs from bundle")
        require(self.to_dict()["context_limit"] <= config["max_position_embeddings"], "bundle exceeds model context limit")
        if check_runtime:
            require(self.to_dict()["runtime"] == runtime_fingerprint(), "runtime changed; prepare and evaluate a new bundle")

    def workload(self) -> FrozenWorkload:
        return FrozenWorkload(WORKLOAD_NAME, self.artifacts, (Knob("prefill_step_size", "integer", minimum=128, maximum=4096),), {"bundle": self.to_dict(), "baseline_prefill_step_size": BASELINE_PREFILL})


def prepare_bundle(root: Path, *, source: dict, fixtures: list[dict], context_limit: int | None = None, device: str = "gpu") -> ModelBundle:
    config, artifacts = inspect_model_files(root)
    limit = config["max_position_embeddings"] if context_limit is None else context_limit
    integer(limit, 1, config["max_position_embeddings"], "context_limit")
    return ModelBundle.from_dict({"schema_version": 1, "backend": "mlx-lm", "device": device, "source": source, "model_type": config["model_type"], "context_limit": limit, "artifacts": [a.to_dict() for a in artifacts], "fixtures": fixtures, "runtime": runtime_fingerprint()})


def bundle_from_workload(workload: FrozenWorkload) -> ModelBundle:
    params = workload.to_dict()["parameters"]
    exact(params, {"bundle", "baseline_prefill_step_size"}, "MLX-LM workload parameters")
    bundle = ModelBundle.from_dict(params["bundle"])
    require(workload.to_dict() == bundle.workload().to_dict(), "MLX-LM workload differs from its closed runner contract")
    require(bundle.to_dict()["runtime"] == runtime_fingerprint(), "MLX-LM workload runtime differs from this host")
    return bundle
