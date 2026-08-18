from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx.errors import ContractError, FailureCode, UnknownFieldError
from auto_mlx.graph import (
    EVIDENCE_GRAPH_SCHEMA_VERSION,
    EvidenceGraph,
    GraphEdge,
    GraphEffect,
    GraphNode,
    TYPED_RELATION_ENDPOINTS,
)


def _node(node_id: str = "mechanism:example", kind: str = "mechanism", **overrides) -> dict:
    data = {
        "id": node_id,
        "kind": kind,
        "title": "Example node",
        "status": "active",
        "confidence": "high",
        "provenance": "code_verified",
        "summary": "An example node.",
        "evidence": ["docs/example.md"],
        "tags": ["example"],
        "observed_at": "2026-08-17",
    }
    data.update(overrides)
    return data


def _effect(**overrides) -> dict:
    data = {
        "metric": "iteration_time",
        "verdict": "regressed",
        "delta_bp_ci": [-310, -257],
    }
    data.update(overrides)
    return data


def _result_node(node_id: str = "result:example", **overrides) -> dict:
    return _node(node_id, kind="applied_result", effect=_effect(), **overrides)


def _edge(from_id: str, to_id: str, relation: str, **overrides) -> dict:
    data = {
        "from": from_id,
        "to": to_id,
        "relation": relation,
        "confidence": "high",
        "rationale": "Example rationale.",
    }
    data.update(overrides)
    return data


def _graph(nodes: list[dict], edges: list[dict]) -> dict:
    return {
        "schema_version": EVIDENCE_GRAPH_SCHEMA_VERSION,
        "graph_id": "test-graph",
        "generated_at": "2026-08-17",
        "nodes": nodes,
        "edges": edges,
    }


def _reified_base() -> tuple[list[dict], list[dict]]:
    """A minimal valid graph holding one fully-anchored applied_result."""

    nodes = [
        _node("mechanism:m", kind="mechanism"),
        _node("model:m", kind="model"),
        _node("hardware:h", kind="hardware"),
        _node("workload:w", kind="workload"),
        _result_node("result:r"),
    ]
    edges = [
        _edge("result:r", "mechanism:m", "instantiates"),
        _edge("result:r", "model:m", "applied_on"),
        _edge("result:r", "hardware:h", "measured_on"),
        _edge("result:r", "workload:w", "under_workload"),
    ]
    return nodes, edges


class GraphEffectTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        effect = GraphEffect.from_dict(_effect(receipt_id="a" * 64, baseline_id="baseline:x", sample_note="one run"))
        self.assertEqual(effect.delta_bp_ci, (-310, -257))
        self.assertEqual(GraphEffect.from_dict(effect.to_dict()), effect)

    def test_optional_fields_are_omitted_from_dict(self) -> None:
        effect = GraphEffect.from_dict(_effect())
        self.assertNotIn("receipt_id", effect.to_dict())

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(UnknownFieldError):
            GraphEffect.from_dict(_effect(surprise=1))

    def test_rejects_open_verdict(self) -> None:
        with self.assertRaises(ContractError) as caught:
            GraphEffect.from_dict(_effect(verdict="won"))
        self.assertEqual(caught.exception.code, FailureCode.INVALID_VALUE)

    def test_rejects_inverted_interval(self) -> None:
        with self.assertRaises(ContractError):
            GraphEffect.from_dict(_effect(delta_bp_ci=[10, -10]))

    def test_rejects_non_pair_interval(self) -> None:
        with self.assertRaises(ContractError):
            GraphEffect.from_dict(_effect(delta_bp_ci=[1, 2, 3]))

    def test_rejects_malformed_receipt_id(self) -> None:
        with self.assertRaises(ContractError):
            GraphEffect.from_dict(_effect(receipt_id="not-a-digest"))


class GraphNodeTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        node = GraphNode.from_dict(_node(identity={"chip": "m4-pro", "memory_gb": 48}, exactness_class="needs_parity_proof"))
        self.assertEqual(GraphNode.from_dict(node.to_dict()), node)

    def test_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node(kind="idea"))

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(UnknownFieldError):
            GraphNode.from_dict(_node(surprise=True))

    def test_rejects_open_provenance(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node(provenance="verified"))

    def test_rejects_malformed_identifier(self) -> None:
        for bad in ("noprefix", "Upper:slug", ":slug", "prefix:"):
            with self.subTest(identifier=bad), self.assertRaises(ContractError):
                GraphNode.from_dict(_node(node_id=bad))

    def test_rejects_duplicate_tags(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node(tags=["x", "x"]))

    def test_applied_result_requires_effect(self) -> None:
        with self.assertRaises(ContractError) as caught:
            GraphNode.from_dict(_node("result:x", kind="applied_result"))
        self.assertIn("structured effect", str(caught.exception))

    def test_effect_is_rejected_outside_applied_result(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node(effect=_effect()))

    def test_exactness_class_is_rejected_outside_mechanism(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node("trait:x", kind="trait", exactness_class="approximate_legal"))

    def test_identity_values_must_be_scalars(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node(identity={"nested": {"a": 1}}))


class GraphEdgeTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        edge = GraphEdge.from_dict(_edge("mechanism:a", "trait:b", "applies_to", observed_at="2026-08-17"))
        self.assertEqual(GraphEdge.from_dict(edge.to_dict()), edge)

    def test_rejects_unknown_relation(self) -> None:
        with self.assertRaises(ContractError):
            GraphEdge.from_dict(_edge("mechanism:a", "trait:b", "inspires"))

    def test_rejects_self_reference(self) -> None:
        with self.assertRaises(ContractError):
            GraphEdge.from_dict(_edge("mechanism:a", "mechanism:a", "supports"))

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaises(UnknownFieldError):
            GraphEdge.from_dict(_edge("mechanism:a", "trait:b", "applies_to", weight=3))


class EvidenceGraphTests(unittest.TestCase):
    def test_round_trip_and_stable_identity(self) -> None:
        document = _graph(
            [_node("mechanism:a"), _node("trait:b", kind="trait")],
            [_edge("mechanism:a", "trait:b", "applies_to")],
        )
        graph = EvidenceGraph.from_dict(document)
        rebuilt = EvidenceGraph.from_json(graph.to_json())
        self.assertEqual(rebuilt, graph)
        self.assertEqual(rebuilt.graph_sha256, graph.graph_sha256)

    def test_rejects_duplicate_node_ids(self) -> None:
        with self.assertRaises(ContractError) as caught:
            EvidenceGraph.from_dict(_graph([_node("mechanism:a"), _node("mechanism:a")], []))
        self.assertEqual(caught.exception.code, FailureCode.IDENTITY_MISMATCH)

    def test_rejects_dangling_edges(self) -> None:
        with self.assertRaises(ContractError) as caught:
            EvidenceGraph.from_dict(_graph([_node("mechanism:a")], [_edge("mechanism:a", "trait:missing", "applies_to")]))
        self.assertIn("dangling", str(caught.exception))

    def test_rejects_wrong_schema_version(self) -> None:
        document = _graph([], [])
        document["schema_version"] = 1
        with self.assertRaises(ContractError):
            EvidenceGraph.from_dict(document)

    def test_rejects_float_effects_via_strict_json(self) -> None:
        document = _graph([_result_node()], [])
        document["nodes"][0]["effect"]["delta_bp_ci"] = [-3.1, -2.57]
        payload = json.dumps(document)
        with self.assertRaises(Exception) as caught:
            EvidenceGraph.from_json(payload)
        self.assertIn("float", str(caught.exception).lower())

    def test_every_typed_relation_enforces_endpoint_kinds(self) -> None:
        kind_examples = {
            "model": _node("model:m", kind="model"),
            "trait": _node("trait:t", kind="trait"),
            "hardware": _node("hardware:h", kind="hardware"),
            "workload": _node("workload:w", kind="workload"),
            "mechanism": _node("mechanism:m", kind="mechanism"),
            "applied_result": _result_node("result:r"),
            "hypothesis": _node("hypothesis:h", kind="hypothesis"),
        }
        base_nodes, base_edges = _reified_base()
        for relation, (from_kinds, to_kinds) in TYPED_RELATION_ENDPOINTS.items():
            good_from = sorted(from_kinds)[0]
            good_to = sorted(to_kinds)[0]
            with self.subTest(relation=relation, case="accepts"):
                if good_from == "applied_result":
                    # The reified base already satisfies anchor cardinality;
                    # confirms/refutes get their extra target added on top.
                    nodes = list(base_nodes)
                    edges = list(base_edges)
                    if relation in {"confirms", "refutes"}:
                        target = kind_examples[good_to]
                        if all(node["id"] != target["id"] for node in nodes):
                            nodes.append(target)
                        edges.append(_edge("result:r", target["id"], relation))
                    EvidenceGraph.from_dict(_graph(nodes, edges))
                else:
                    EvidenceGraph.from_dict(
                        _graph(
                            [kind_examples[good_from], kind_examples[good_to]],
                            [_edge(kind_examples[good_from]["id"], kind_examples[good_to]["id"], relation)],
                        )
                    )
            with self.subTest(relation=relation, case="rejects_from"):
                with self.assertRaises(ContractError):
                    EvidenceGraph.from_dict(
                        _graph(
                            [_node("constraint:c", kind="constraint"), kind_examples[good_to]],
                            [_edge("constraint:c", kind_examples[good_to]["id"], relation)],
                        )
                    )
            with self.subTest(relation=relation, case="rejects_to"):
                with self.assertRaises(ContractError):
                    EvidenceGraph.from_dict(
                        _graph(
                            [kind_examples[good_from], _node("constraint:c", kind="constraint")],
                            [_edge(kind_examples[good_from]["id"], "constraint:c", relation)],
                        )
                    )

    def test_untyped_relations_are_unconstrained(self) -> None:
        EvidenceGraph.from_dict(
            _graph(
                [_node("constraint:c", kind="constraint"), _node("finding:f", kind="finding")],
                [_edge("constraint:c", "finding:f", "invalidates")],
            )
        )

    def test_checked_in_example_validates(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "evidence-graph.json"
        graph = EvidenceGraph.from_json(example.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(graph.nodes), 6)
        self.assertGreaterEqual(len(graph.edges), 6)

    def test_node_lookup(self) -> None:
        graph = EvidenceGraph.from_dict(_graph([_node("mechanism:a")], []))
        self.assertEqual(graph.node("mechanism:a").kind, "mechanism")
        with self.assertRaises(ContractError):
            graph.node("mechanism:missing")


class ReificationAndStrictnessTests(unittest.TestCase):
    """Rules adopted from the knowledge-repo integration review."""

    def test_id_prefix_must_match_kind(self) -> None:
        with self.assertRaises(ContractError) as caught:
            GraphNode.from_dict(_node("mechanism:x", kind="trait"))
        self.assertEqual(caught.exception.code, FailureCode.IDENTITY_MISMATCH)
        # applied_result and external_reference use their short prefixes.
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node("applied_result:x", kind="applied_result", effect=_effect()))
        GraphNode.from_dict(_node("external:x", kind="external_reference"))

    def test_evidence_must_be_non_empty(self) -> None:
        with self.assertRaises(ContractError):
            GraphNode.from_dict(_node(evidence=[]))

    def test_applied_result_requires_complete_anchors(self) -> None:
        nodes, edges = _reified_base()
        EvidenceGraph.from_dict(_graph(nodes, edges))  # complete: valid
        for missing in ("instantiates", "applied_on", "measured_on", "under_workload"):
            with self.subTest(missing=missing):
                with self.assertRaises(ContractError) as caught:
                    EvidenceGraph.from_dict(
                        _graph(nodes, [edge for edge in edges if edge["relation"] != missing])
                    )
                self.assertIn(missing, str(caught.exception))

    def test_applied_result_rejects_ambiguous_anchor(self) -> None:
        nodes, edges = _reified_base()
        nodes = nodes + [_node("model:other", kind="model")]
        edges = edges + [_edge("result:r", "model:other", "applied_on")]
        with self.assertRaises(ContractError) as caught:
            EvidenceGraph.from_dict(_graph(nodes, edges))
        self.assertIn("exactly one", str(caught.exception))

    def test_second_instantiates_edge_is_allowed(self) -> None:
        nodes, edges = _reified_base()
        nodes = nodes + [_node("mechanism:extra", kind="mechanism")]
        edges = edges + [_edge("result:r", "mechanism:extra", "instantiates")]
        EvidenceGraph.from_dict(_graph(nodes, edges))

    def test_metric_direction_checks_verdict_sign(self) -> None:
        GraphEffect.from_dict(_effect(metric_direction="higher_is_better"))  # regressed, CI < 0: consistent
        with self.assertRaises(ContractError):
            GraphEffect.from_dict(_effect(verdict="improved", metric_direction="higher_is_better"))
        GraphEffect.from_dict(_effect(verdict="improved", metric_direction="lower_is_better"))
        # Inconclusive never carries a sign claim, so any interval is fine.
        GraphEffect.from_dict(_effect(verdict="inconclusive", delta_bp_ci=[-50, 50], metric_direction="higher_is_better"))
        with self.assertRaises(ContractError):
            GraphEffect.from_dict(_effect(metric_direction="sideways"))


if __name__ == "__main__":
    unittest.main()
