"""Trait-indexed evidence-graph contracts (evidence-graph schema version 2).

The evidence graph is the knowledge layer shared between this tool and the
external knowledge repository: optimization mechanisms, model architecture
traits, hardware identities, and reified measurement results, connected by
typed relations. It is a *decision aid with citations*, never measurement
evidence itself:

- The graph never promotes. Activation still requires a fresh, attested,
  independently recomputed receipt through the existing promotion lane.
- Hyperedges are reified: an ``applied_result`` node binds one mechanism to
  one model, hardware, and workload through typed binary edges
  (``instantiates``/``applied_on``/``measured_on``/``under_workload``) and
  carries the structured measurement effect, so hypergraph semantics come
  from plain, strictly-validated binary edges.
- Effects are integers in basis points (``delta_bp_ci``); the strict-JSON
  discipline of this package (no floats, no duplicate keys, no non-finite
  numbers) applies to graph documents exactly as it does to every other
  contract.
- Everything fails closed: unknown kinds, relations, or fields, dangling
  edges, duplicate node ids, malformed identifiers, and a missing effect on
  an ``applied_result`` are all typed contract errors, never warnings.

The packaged JSON Schema twin is ``schemas/evidence-graph.schema.json``; this
module is the executable authority and enforces the structural rules the JSON
Schema cannot express (global id uniqueness, dangling-edge rejection, and
endpoint-kind constraints for the typed relations).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_json, sha256_hex, strict_json_loads, validate_json_value
from .errors import ContractError, FailureCode, UnknownFieldError
from .paths import validate_sha256


EVIDENCE_GRAPH_SCHEMA_VERSION: Final = 2

NODE_KINDS: Final = frozenset(
    {
        "model",
        "trait",
        "hardware",
        "workload",
        "mechanism",
        "applied_result",
        "hypothesis",
        "constraint",
        "finding",
        "external_reference",
        "frontier",
    }
)

EDGE_RELATIONS: Final = frozenset(
    {
        # Inherited from the v1 campaign vocabulary.
        "contains",
        "supports",
        "constrains",
        "depends_on",
        "overlaps",
        "contradicts",
        "invalidates",
        "suggests",
        "requires_gate",
        "supersedes",
        "cannot_validate",
        "competes_with",
        "duplicates",
        # v2 transfer vocabulary.
        "exhibits",
        "applies_to",
        "conditioned_on",
        "instantiates",
        "applied_on",
        "measured_on",
        "under_workload",
        "transfer_predicted",
        "confirms",
        "refutes",
    }
)

CONFIDENCES: Final = frozenset({"high", "medium", "low"})

PROVENANCES: Final = frozenset(
    {
        "official_verified",
        "code_verified",
        "local_measured",
        "author_claim",
        "contributor_claim",
        "replicated",
        "research_inference",
        "transfer_inference",
    }
)

EFFECT_VERDICTS: Final = frozenset({"improved", "regressed", "inconclusive"})

METRIC_DIRECTIONS: Final = frozenset({"higher_is_better", "lower_is_better"})

# The id prefix (everything before the first colon) must match the node's
# kind, so an id can never silently disagree with what it names.
KIND_ID_PREFIXES: Final[Mapping[str, str]] = {
    "model": "model",
    "trait": "trait",
    "hardware": "hardware",
    "workload": "workload",
    "mechanism": "mechanism",
    "applied_result": "result",
    "hypothesis": "hypothesis",
    "constraint": "constraint",
    "finding": "finding",
    "external_reference": "external",
    "frontier": "frontier",
}

# A reified hyperedge is only a hyperedge if its anchors are complete and
# unambiguous: every applied_result must bind exactly one model, hardware,
# and workload, and instantiate at least one mechanism.
REIFICATION_EXACTLY_ONE: Final = ("applied_on", "measured_on", "under_workload")
REIFICATION_AT_LEAST_ONE: Final = ("instantiates",)

EXACTNESS_CLASSES: Final = frozenset(
    {"exact_by_construction", "needs_parity_proof", "approximate_legal"}
)

# Endpoint-kind constraints for the typed v2 relations. A relation absent
# from this table is intentionally unconstrained (the inherited v1 vocabulary
# plus ``conditioned_on``). ``confirms``/``refutes`` close the transfer loop:
# only a reified measurement may confirm or refute a mechanism or hypothesis.
TYPED_RELATION_ENDPOINTS: Final[Mapping[str, tuple[frozenset[str], frozenset[str]]]] = {
    "exhibits": (frozenset({"model"}), frozenset({"trait"})),
    "applies_to": (frozenset({"mechanism"}), frozenset({"trait"})),
    "instantiates": (frozenset({"applied_result"}), frozenset({"mechanism"})),
    "applied_on": (frozenset({"applied_result"}), frozenset({"model"})),
    "measured_on": (frozenset({"applied_result"}), frozenset({"hardware"})),
    "under_workload": (frozenset({"applied_result"}), frozenset({"workload"})),
    "transfer_predicted": (frozenset({"mechanism"}), frozenset({"model"})),
    "confirms": (frozenset({"applied_result"}), frozenset({"mechanism", "hypothesis"})),
    "refutes": (frozenset({"applied_result"}), frozenset({"mechanism", "hypothesis"})),
}

MAX_GRAPH_NODES: Final = 100_000
MAX_GRAPH_EDGES: Final = 500_000
MAX_NODE_EVIDENCE_ENTRIES: Final = 64
MAX_NODE_TAGS: Final = 64
MAX_IDENTITY_ENTRIES: Final = 32
MAX_EFFECT_DELTA_BP: Final = 1_000_000

_MAX_ID_LENGTH: Final = 300
_MAX_TITLE_LENGTH: Final = 300
_MAX_STATUS_LENGTH: Final = 100
_MAX_SUMMARY_LENGTH: Final = 4000
_MAX_EVIDENCE_LENGTH: Final = 1000
_MAX_TAG_LENGTH: Final = 100
_MAX_OBSERVED_AT_LENGTH: Final = 64
_MAX_GRAPH_ID_LENGTH: Final = 200
_MAX_METRIC_LENGTH: Final = 100
_MAX_SAMPLE_NOTE_LENGTH: Final = 1000
_MAX_BASELINE_ID_LENGTH: Final = 300
_MAX_RATIONALE_LENGTH: Final = 2000

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._/@-]*:[A-Za-z0-9][A-Za-z0-9._/@+-]*$")


def _object(value: Any, *, label: str) -> dict[str, Any]:
    validate_json_value(value)
    if type(value) is not dict:
        raise ContractError(f"{label} must be a JSON object", code=FailureCode.WRONG_TYPE)
    return value


def _fields(value: dict[str, Any], required: frozenset[str], optional: frozenset[str], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ContractError(f"{label} field names must be strings", code=FailureCode.WRONG_TYPE)
    unknown = set(value) - required - optional
    missing = required - set(value)
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ContractError(
            f"{label} is missing field(s): {', '.join(sorted(missing))}",
            code=FailureCode.INVALID_VALUE,
        )


def _string(value: Any, *, label: str, max_length: int, non_empty: bool = True) -> str:
    if type(value) is not str or (non_empty and not value):
        raise ContractError(f"{label} must be a {'non-empty ' if non_empty else ''}string", code=FailureCode.WRONG_TYPE)
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractError(f"{label} must not contain unpaired surrogates", code=FailureCode.INVALID_UNICODE)
    if len(value) > max_length:
        raise ContractError(f"{label} cannot exceed {max_length} characters", code=FailureCode.INVALID_VALUE)
    return value


def _identifier(value: Any, *, label: str) -> str:
    text = _string(value, label=label, max_length=_MAX_ID_LENGTH)
    if len(text) < 3 or _IDENTIFIER_PATTERN.fullmatch(text) is None:
        raise ContractError(
            f"{label} must match 'prefix:slug' (lowercase prefix, e.g. 'mechanism:replay-prefetch')",
            code=FailureCode.INVALID_VALUE,
        )
    return text


def _integer(value: Any, *, label: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ContractError(f"{label} must be an integer", code=FailureCode.WRONG_TYPE)
    if minimum is not None and value < minimum:
        raise ContractError(f"{label} must be >= {minimum}", code=FailureCode.INVALID_VALUE)
    if maximum is not None and value > maximum:
        raise ContractError(f"{label} must be <= {maximum}", code=FailureCode.INVALID_VALUE)
    return value


_EFFECT_REQUIRED: Final = frozenset({"metric", "verdict", "delta_bp_ci"})
_EFFECT_OPTIONAL: Final = frozenset({"metric_direction", "receipt_id", "baseline_id", "sample_note"})


@dataclass(frozen=True, slots=True)
class GraphEffect:
    """The structured, integer-basis-point measurement effect of one applied result."""

    metric: str
    verdict: str
    delta_bp_ci: tuple[int, int]
    metric_direction: str | None = None
    receipt_id: str | None = None
    baseline_id: str | None = None
    sample_note: str | None = None

    def __post_init__(self) -> None:
        _string(self.metric, label="effect.metric", max_length=_MAX_METRIC_LENGTH)
        if self.verdict not in EFFECT_VERDICTS:
            raise ContractError("effect.verdict is not a closed verdict", code=FailureCode.INVALID_VALUE)
        if type(self.delta_bp_ci) is not tuple or len(self.delta_bp_ci) != 2:
            raise ContractError("effect.delta_bp_ci must be a [lower, upper] pair", code=FailureCode.WRONG_TYPE)
        lower, upper = self.delta_bp_ci
        _integer(lower, label="effect.delta_bp_ci[0]", minimum=-MAX_EFFECT_DELTA_BP, maximum=MAX_EFFECT_DELTA_BP)
        _integer(upper, label="effect.delta_bp_ci[1]", minimum=-MAX_EFFECT_DELTA_BP, maximum=MAX_EFFECT_DELTA_BP)
        if lower > upper:
            raise ContractError("effect.delta_bp_ci lower bound cannot exceed upper bound", code=FailureCode.INVALID_VALUE)
        if self.metric_direction is not None:
            if self.metric_direction not in METRIC_DIRECTIONS:
                raise ContractError("effect.metric_direction is not closed", code=FailureCode.INVALID_VALUE)
            # With a declared direction, a decisive verdict must agree with
            # the interval's sign; without one, no sign claim is checkable.
            better = lower > 0 if self.metric_direction == "higher_is_better" else upper < 0
            worse = upper < 0 if self.metric_direction == "higher_is_better" else lower > 0
            if self.verdict == "improved" and not better:
                raise ContractError(
                    "effect.verdict 'improved' contradicts delta_bp_ci under the declared metric_direction",
                    code=FailureCode.INVALID_VALUE,
                )
            if self.verdict == "regressed" and not worse:
                raise ContractError(
                    "effect.verdict 'regressed' contradicts delta_bp_ci under the declared metric_direction",
                    code=FailureCode.INVALID_VALUE,
                )
        if self.receipt_id is not None:
            validate_sha256(self.receipt_id)
        if self.baseline_id is not None:
            _string(self.baseline_id, label="effect.baseline_id", max_length=_MAX_BASELINE_ID_LENGTH)
        if self.sample_note is not None:
            _string(self.sample_note, label="effect.sample_note", max_length=_MAX_SAMPLE_NOTE_LENGTH)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "metric": self.metric,
            "verdict": self.verdict,
            "delta_bp_ci": list(self.delta_bp_ci),
        }
        if self.metric_direction is not None:
            result["metric_direction"] = self.metric_direction
        if self.receipt_id is not None:
            result["receipt_id"] = self.receipt_id
        if self.baseline_id is not None:
            result["baseline_id"] = self.baseline_id
        if self.sample_note is not None:
            result["sample_note"] = self.sample_note
        return result

    @classmethod
    def from_dict(cls, value: Any, *, label: str = "effect") -> "GraphEffect":
        data = _object(value, label=label)
        _fields(data, _EFFECT_REQUIRED, _EFFECT_OPTIONAL, label=label)
        ci = data["delta_bp_ci"]
        if type(ci) is not list or len(ci) != 2:
            raise ContractError(f"{label}.delta_bp_ci must be a [lower, upper] pair", code=FailureCode.WRONG_TYPE)
        return cls(
            metric=data["metric"],
            verdict=data["verdict"],
            delta_bp_ci=(ci[0], ci[1]),
            metric_direction=data.get("metric_direction"),
            receipt_id=data.get("receipt_id"),
            baseline_id=data.get("baseline_id"),
            sample_note=data.get("sample_note"),
        )


_NODE_REQUIRED: Final = frozenset(
    {"id", "kind", "title", "status", "confidence", "provenance", "summary", "evidence", "tags", "observed_at"}
)
_NODE_OPTIONAL: Final = frozenset({"identity", "exactness_class", "effect"})


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One strictly-validated evidence-graph node."""

    node_id: str
    kind: str
    title: str
    status: str
    confidence: str
    provenance: str
    summary: str
    evidence: tuple[str, ...]
    tags: tuple[str, ...]
    observed_at: str
    identity: Mapping[str, Any] | None = None
    exactness_class: str | None = None
    effect: GraphEffect | None = None

    def __post_init__(self) -> None:
        _identifier(self.node_id, label="node.id")
        if self.kind not in NODE_KINDS:
            raise ContractError(f"node {self.node_id} has unknown kind {self.kind!r}", code=FailureCode.INVALID_VALUE)
        expected_prefix = KIND_ID_PREFIXES[self.kind]
        if self.node_id.split(":", 1)[0] != expected_prefix:
            raise ContractError(
                f"node {self.node_id} of kind {self.kind!r} must use the id prefix {expected_prefix!r}:",
                code=FailureCode.IDENTITY_MISMATCH,
            )
        _string(self.title, label=f"node {self.node_id} title", max_length=_MAX_TITLE_LENGTH)
        _string(self.status, label=f"node {self.node_id} status", max_length=_MAX_STATUS_LENGTH)
        if self.confidence not in CONFIDENCES:
            raise ContractError(f"node {self.node_id} confidence is not closed", code=FailureCode.INVALID_VALUE)
        if self.provenance not in PROVENANCES:
            raise ContractError(f"node {self.node_id} provenance is not closed", code=FailureCode.INVALID_VALUE)
        _string(self.summary, label=f"node {self.node_id} summary", max_length=_MAX_SUMMARY_LENGTH)
        if type(self.evidence) is not tuple:
            raise ContractError(f"node {self.node_id} evidence must be an array", code=FailureCode.WRONG_TYPE)
        if not self.evidence:
            raise ContractError(
                f"node {self.node_id} must carry at least one evidence locator",
                code=FailureCode.INVALID_VALUE,
            )
        if len(self.evidence) > MAX_NODE_EVIDENCE_ENTRIES:
            raise ContractError(
                f"node {self.node_id} evidence cannot exceed {MAX_NODE_EVIDENCE_ENTRIES} entries",
                code=FailureCode.INVALID_VALUE,
            )
        for index, entry in enumerate(self.evidence):
            _string(entry, label=f"node {self.node_id} evidence[{index}]", max_length=_MAX_EVIDENCE_LENGTH)
        if type(self.tags) is not tuple:
            raise ContractError(f"node {self.node_id} tags must be an array", code=FailureCode.WRONG_TYPE)
        if len(self.tags) > MAX_NODE_TAGS:
            raise ContractError(
                f"node {self.node_id} tags cannot exceed {MAX_NODE_TAGS} entries", code=FailureCode.INVALID_VALUE
            )
        for index, tag in enumerate(self.tags):
            _string(tag, label=f"node {self.node_id} tags[{index}]", max_length=_MAX_TAG_LENGTH)
        if len(set(self.tags)) != len(self.tags):
            raise ContractError(f"node {self.node_id} tags must be unique", code=FailureCode.INVALID_VALUE)
        _string(self.observed_at, label=f"node {self.node_id} observed_at", max_length=_MAX_OBSERVED_AT_LENGTH)
        if self.identity is not None:
            identity = _object(dict(self.identity), label=f"node {self.node_id} identity")
            if len(identity) > MAX_IDENTITY_ENTRIES:
                raise ContractError(
                    f"node {self.node_id} identity cannot exceed {MAX_IDENTITY_ENTRIES} entries",
                    code=FailureCode.INVALID_VALUE,
                )
            for key, entry in identity.items():
                if type(entry) not in {str, int, bool}:
                    raise ContractError(
                        f"node {self.node_id} identity[{key!r}] must be a string, integer, or boolean",
                        code=FailureCode.WRONG_TYPE,
                    )
            object.__setattr__(self, "identity", identity)
        if self.exactness_class is not None:
            if self.kind != "mechanism":
                raise ContractError(
                    f"node {self.node_id} exactness_class is only valid on mechanism nodes",
                    code=FailureCode.INVALID_VALUE,
                )
            if self.exactness_class not in EXACTNESS_CLASSES:
                raise ContractError(
                    f"node {self.node_id} exactness_class is not closed", code=FailureCode.INVALID_VALUE
                )
        if self.kind == "applied_result":
            if self.effect is None:
                raise ContractError(
                    f"applied_result node {self.node_id} must carry a structured effect",
                    code=FailureCode.INVALID_VALUE,
                )
        elif self.effect is not None:
            raise ContractError(
                f"node {self.node_id} effect is only valid on applied_result nodes",
                code=FailureCode.INVALID_VALUE,
            )
        if self.effect is not None and not isinstance(self.effect, GraphEffect):
            raise ContractError(f"node {self.node_id} effect must be a GraphEffect", code=FailureCode.WRONG_TYPE)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.node_id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "tags": list(self.tags),
            "observed_at": self.observed_at,
        }
        if self.identity is not None:
            result["identity"] = dict(self.identity)
        if self.exactness_class is not None:
            result["exactness_class"] = self.exactness_class
        if self.effect is not None:
            result["effect"] = self.effect.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "GraphNode":
        data = _object(value, label="graph node")
        _fields(data, _NODE_REQUIRED, _NODE_OPTIONAL, label=f"graph node {data.get('id', '<missing id>')!r}")
        evidence = data["evidence"]
        if type(evidence) is not list:
            raise ContractError("graph node evidence must be an array", code=FailureCode.WRONG_TYPE)
        tags = data["tags"]
        if type(tags) is not list:
            raise ContractError("graph node tags must be an array", code=FailureCode.WRONG_TYPE)
        effect_value = data.get("effect")
        return cls(
            node_id=data["id"],
            kind=data["kind"],
            title=data["title"],
            status=data["status"],
            confidence=data["confidence"],
            provenance=data["provenance"],
            summary=data["summary"],
            evidence=tuple(evidence),
            tags=tuple(tags),
            observed_at=data["observed_at"],
            identity=data.get("identity"),
            exactness_class=data.get("exactness_class"),
            effect=None if effect_value is None else GraphEffect.from_dict(
                effect_value, label=f"node {data.get('id', '<missing id>')!r} effect"
            ),
        )


_EDGE_REQUIRED: Final = frozenset({"from", "to", "relation", "confidence", "rationale"})
_EDGE_OPTIONAL: Final = frozenset({"observed_at"})


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One strictly-validated, typed evidence-graph edge."""

    from_id: str
    to_id: str
    relation: str
    confidence: str
    rationale: str
    observed_at: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.from_id, label="edge.from")
        _identifier(self.to_id, label="edge.to")
        if self.from_id == self.to_id:
            raise ContractError(f"edge {self.from_id} cannot reference itself", code=FailureCode.INVALID_VALUE)
        if self.relation not in EDGE_RELATIONS:
            raise ContractError(
                f"edge {self.from_id} -> {self.to_id} has unknown relation {self.relation!r}",
                code=FailureCode.INVALID_VALUE,
            )
        if self.confidence not in CONFIDENCES:
            raise ContractError(
                f"edge {self.from_id} -> {self.to_id} confidence is not closed", code=FailureCode.INVALID_VALUE
            )
        _string(
            self.rationale,
            label=f"edge {self.from_id} -> {self.to_id} rationale",
            max_length=_MAX_RATIONALE_LENGTH,
        )
        if self.observed_at is not None:
            _string(
                self.observed_at,
                label=f"edge {self.from_id} -> {self.to_id} observed_at",
                max_length=_MAX_OBSERVED_AT_LENGTH,
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "from": self.from_id,
            "to": self.to_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }
        if self.observed_at is not None:
            result["observed_at"] = self.observed_at
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "GraphEdge":
        data = _object(value, label="graph edge")
        _fields(data, _EDGE_REQUIRED, _EDGE_OPTIONAL, label="graph edge")
        return cls(
            from_id=data["from"],
            to_id=data["to"],
            relation=data["relation"],
            confidence=data["confidence"],
            rationale=data["rationale"],
            observed_at=data.get("observed_at"),
        )


_GRAPH_REQUIRED: Final = frozenset({"schema_version", "graph_id", "generated_at", "nodes", "edges"})
_GRAPH_OPTIONAL: Final = frozenset()


@dataclass(frozen=True, slots=True)
class EvidenceGraph:
    """A closed, immutable, identity-bearing evidence-graph document."""

    graph_id: str
    generated_at: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        _string(self.graph_id, label="graph_id", max_length=_MAX_GRAPH_ID_LENGTH)
        _string(self.generated_at, label="generated_at", max_length=_MAX_OBSERVED_AT_LENGTH)
        if type(self.nodes) is not tuple:
            raise ContractError("graph nodes must be an array", code=FailureCode.WRONG_TYPE)
        if type(self.edges) is not tuple:
            raise ContractError("graph edges must be an array", code=FailureCode.WRONG_TYPE)
        if len(self.nodes) > MAX_GRAPH_NODES:
            raise ContractError(
                f"graph cannot contain more than {MAX_GRAPH_NODES} nodes", code=FailureCode.INPUT_TOO_LARGE
            )
        if len(self.edges) > MAX_GRAPH_EDGES:
            raise ContractError(
                f"graph cannot contain more than {MAX_GRAPH_EDGES} edges", code=FailureCode.INPUT_TOO_LARGE
            )
        kinds: dict[str, str] = {}
        for node in self.nodes:
            if not isinstance(node, GraphNode):
                raise ContractError("graph nodes must be GraphNode instances", code=FailureCode.WRONG_TYPE)
            if node.node_id in kinds:
                raise ContractError(
                    f"graph contains duplicate node id {node.node_id}", code=FailureCode.IDENTITY_MISMATCH
                )
            kinds[node.node_id] = node.kind
        for edge in self.edges:
            if not isinstance(edge, GraphEdge):
                raise ContractError("graph edges must be GraphEdge instances", code=FailureCode.WRONG_TYPE)
            for endpoint, endpoint_id in (("from", edge.from_id), ("to", edge.to_id)):
                if endpoint_id not in kinds:
                    raise ContractError(
                        f"edge {edge.from_id} -[{edge.relation}]-> {edge.to_id} has a dangling {endpoint} reference",
                        code=FailureCode.INVALID_VALUE,
                    )
            constraint = TYPED_RELATION_ENDPOINTS.get(edge.relation)
            if constraint is not None:
                from_kinds, to_kinds = constraint
                if kinds[edge.from_id] not in from_kinds:
                    raise ContractError(
                        f"relation {edge.relation!r} requires a from-node of kind "
                        f"{'/'.join(sorted(from_kinds))}, got {kinds[edge.from_id]!r} ({edge.from_id})",
                        code=FailureCode.INVALID_VALUE,
                    )
                if kinds[edge.to_id] not in to_kinds:
                    raise ContractError(
                        f"relation {edge.relation!r} requires a to-node of kind "
                        f"{'/'.join(sorted(to_kinds))}, got {kinds[edge.to_id]!r} ({edge.to_id})",
                        code=FailureCode.INVALID_VALUE,
                    )
        # A reified hyperedge must be complete and unambiguous: exactly one
        # model, hardware, and workload anchor, at least one mechanism.
        anchor_counts: dict[str, dict[str, int]] = {
            node.node_id: dict.fromkeys(REIFICATION_EXACTLY_ONE + REIFICATION_AT_LEAST_ONE, 0)
            for node in self.nodes
            if node.kind == "applied_result"
        }
        for edge in self.edges:
            counts = anchor_counts.get(edge.from_id)
            if counts is not None and edge.relation in counts:
                counts[edge.relation] += 1
        for node_id, counts in anchor_counts.items():
            for relation in REIFICATION_EXACTLY_ONE:
                if counts[relation] != 1:
                    raise ContractError(
                        f"applied_result {node_id} must have exactly one {relation!r} edge, found {counts[relation]}",
                        code=FailureCode.INVALID_VALUE,
                    )
            for relation in REIFICATION_AT_LEAST_ONE:
                if counts[relation] < 1:
                    raise ContractError(
                        f"applied_result {node_id} must have at least one {relation!r} edge",
                        code=FailureCode.INVALID_VALUE,
                    )

    @property
    def graph_sha256(self) -> str:
        return sha256_hex(self.to_dict())

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ContractError(f"graph has no node {node_id!r}", code=FailureCode.INVALID_VALUE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_GRAPH_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "generated_at": self.generated_at,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceGraph":
        data = _object(value, label="evidence graph")
        _fields(data, _GRAPH_REQUIRED, _GRAPH_OPTIONAL, label="evidence graph")
        if data["schema_version"] != EVIDENCE_GRAPH_SCHEMA_VERSION:
            raise ContractError(
                f"evidence graph schema_version must be {EVIDENCE_GRAPH_SCHEMA_VERSION}",
                code=FailureCode.INVALID_VALUE,
            )
        nodes_value = data["nodes"]
        if type(nodes_value) is not list:
            raise ContractError("graph nodes must be an array", code=FailureCode.WRONG_TYPE)
        edges_value = data["edges"]
        if type(edges_value) is not list:
            raise ContractError("graph edges must be an array", code=FailureCode.WRONG_TYPE)
        return cls(
            graph_id=data["graph_id"],
            generated_at=data["generated_at"],
            nodes=tuple(GraphNode.from_dict(entry) for entry in nodes_value),
            edges=tuple(GraphEdge.from_dict(entry) for entry in edges_value),
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "EvidenceGraph":
        return cls.from_dict(strict_json_loads(value))


__all__: Final = [
    "CONFIDENCES",
    "EDGE_RELATIONS",
    "EFFECT_VERDICTS",
    "EVIDENCE_GRAPH_SCHEMA_VERSION",
    "EXACTNESS_CLASSES",
    "EvidenceGraph",
    "GraphEdge",
    "GraphEffect",
    "GraphNode",
    "KIND_ID_PREFIXES",
    "MAX_EFFECT_DELTA_BP",
    "MAX_GRAPH_EDGES",
    "MAX_GRAPH_NODES",
    "METRIC_DIRECTIONS",
    "NODE_KINDS",
    "PROVENANCES",
    "REIFICATION_AT_LEAST_ONE",
    "REIFICATION_EXACTLY_ONE",
    "TYPED_RELATION_ENDPOINTS",
]
