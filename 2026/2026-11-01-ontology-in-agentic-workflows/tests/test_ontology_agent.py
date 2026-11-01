"""Offline tests for ontology_agent.py: no network, no cloud, no model. All data is fictional."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Literal
from rdflib.namespace import OWL, RDF, RDFS, SKOS

import ontology_agent as oa

SHOP = oa.SHOP


@pytest.fixture()
def graph():
    return oa.load_ontology()


@pytest.fixture()
def shapes(graph):
    return oa.build_shapes(graph)


def link(subject, relation, obj):
    return {"subject": subject, "relation": relation, "object": obj}


# ---------- 1. Entities ----------

def test_three_synonyms_resolve_to_one_entity(graph):
    assert {oa.resolve_entity(graph, w) for w in ("client", "buyer", "account")} == {"Customer"}


@pytest.mark.parametrize("word, entity", [
    ("Customer", "Customer"), ("  BUYER ", "Customer"), ("parcel", "Shipment"), ("purchase", "Order"),
])
def test_resolve_matches_name_and_labels_ignoring_case_and_spaces(graph, word, entity):
    assert oa.resolve_entity(graph, word) == entity


@pytest.mark.parametrize("word", ["", "   ", "invoice", "custom"])
def test_resolve_returns_none_for_empty_or_unknown_words(graph, word):
    assert oa.resolve_entity(graph, word) is None


def test_entity_types_are_the_ontology_classes(graph):
    assert oa.entity_types(graph) == ["Customer", "Order", "Shipment"]


# ---------- 2. Relationships ----------

def test_relations_carry_domain_and_range(graph):
    assert oa.relations(graph) == [oa.Relation("placedBy", "Order", "Customer"),
                                   oa.Relation("shipsTo", "Shipment", "Customer")]


def test_relation_without_range_is_an_ontology_error(graph):
    graph.add((SHOP.billedTo, RDF.type, OWL.ObjectProperty))
    graph.add((SHOP.billedTo, RDFS.domain, SHOP.Order))
    with pytest.raises(oa.OntologyError, match="billedTo"):
        oa.relations(graph)


# ---------- 3. Bindings ----------

def test_bindings_map_each_entity_to_its_table_and_key(graph):
    b = oa.bindings(graph)
    assert {(x.entity, x.table, x.key, x.id_prefix) for x in b.values()} == {
        ("Customer", "crm.customers", "customer_id", "cust"),
        ("Order", "sales.orders", "order_id", "order"),
        ("Shipment", "wms.shipments", "ship_id", "ship"),
    }


def test_class_without_binding_is_an_ontology_error(graph):
    graph.add((SHOP.Invoice, RDF.type, OWL.Class))
    with pytest.raises(oa.OntologyError, match="Invoice has no binding"):
        oa.bindings(graph)


@pytest.mark.parametrize("instance_id, entity", [
    ("cust:7", "Customer"), ("order:42", "Order"), ("ship:9", "Shipment"),
    ("cust:99", None), ("vendor:7", None), ("cust7", None), ("cust:", None), ("", None),
])
def test_type_of_reads_the_row_through_the_binding(graph, instance_id, entity):
    assert oa.type_of(instance_id, oa.bindings(graph), oa.FAKE_TABLES) == entity


def test_get_entity_reads_through_the_binding(graph):
    tool = oa.build_tool_schemas(graph)["get_entity"]
    assert oa.get_entity(graph, tool, {"entity": "Order", "key": "42"})["isError"] is False
    missing = oa.get_entity(graph, tool, {"entity": "Order", "key": "99"})
    assert missing["isError"] is True and "sales.orders" in missing["content"][0]["text"]


# ---------- 4. Tools ----------

def test_list_entity_types_tool_returns_synonyms_and_tables(graph):
    rows = {r["entity"]: r for r in oa.list_entity_types(graph)}
    assert rows["Customer"]["synonyms"] == ["account", "buyer", "client"]
    assert rows["Shipment"]["table"] == "wms.shipments"


def test_search_ontology_resolves_a_synonym_and_its_relations(graph):
    assert oa.search_ontology(graph, "buyer") == {"term": "buyer", "entity": "Customer",
                                                  "relations": ["placedBy", "shipsTo"]}
    assert oa.search_ontology(graph, "invoice") == {"term": "invoice", "entity": None, "relations": []}


# ---------- 5. Contract ----------

def test_tool_enums_are_generated_from_the_ontology(graph):
    tools = oa.build_tool_schemas(graph)
    assert tools["get_entity"]["inputSchema"]["properties"]["entity"]["enum"] == oa.entity_types(graph)
    assert tools["propose_link"]["inputSchema"]["properties"]["relation"]["enum"] == ["placedBy", "shipsTo"]
    assert all(t["inputSchema"]["additionalProperties"] is False for t in tools.values())


def test_a_new_class_and_relation_reach_the_enums_without_editing_code(graph):
    graph.add((SHOP.Invoice, RDF.type, OWL.Class))
    graph.add((SHOP.billedTo, RDF.type, OWL.ObjectProperty))
    graph.add((SHOP.billedTo, RDFS.domain, SHOP.Invoice))
    graph.add((SHOP.billedTo, RDFS.range, SHOP.Customer))
    tools = oa.build_tool_schemas(graph)
    assert "Invoice" in tools["get_entity"]["inputSchema"]["properties"]["entity"]["enum"]
    assert "billedTo" in tools["propose_link"]["inputSchema"]["properties"]["relation"]["enum"]


@pytest.mark.parametrize("arguments, fragment", [
    (link("order:42", "orderedBy", "cust:7"), "'orderedBy' is not one of"),
    ({**link("order:42", "placedBy", "cust:7"), "note": "x"}, "Additional properties"),
    ({"subject": "order:42", "relation": "placedBy"}, "'object' is a required property"),
    (link("order 42", "placedBy", "cust:7"), "does not match"),
    ([], "is not of type 'object'"),
])
def test_server_rejects_arguments_outside_the_contract(graph, shapes, arguments, fragment):
    result = oa.call_tool(graph, shapes, "propose_link", arguments)
    assert result["isError"] is True
    assert fragment in result["content"][0]["text"]


def test_get_entity_rejects_an_entity_outside_the_enum(graph, shapes):
    result = oa.call_tool(graph, shapes, "get_entity", {"entity": "buyer", "key": "7"})
    assert result["isError"] is True and "is not one of" in result["content"][0]["text"]


# ---------- 6. Gate ----------

def test_shapes_are_generated_two_per_relation(graph, shapes):
    from rdflib.namespace import SH
    assert len(set(shapes.subjects(RDF.type, SH.NodeShape))) == 2 * len(oa.relations(graph))


def test_gate_accepts_a_link_that_matches_domain_and_range(graph, shapes):
    result = oa.call_tool(graph, shapes, "propose_link", link("order:42", "placedBy", "cust:7"))
    assert result == oa.tool_result("Accepted: order:42 placedBy cust:7", False)


def test_gate_rejects_a_wrong_range_and_says_what_it_got(graph, shapes):
    result = oa.call_tool(graph, shapes, "propose_link", link("order:42", "placedBy", "ship:9"))
    assert result["isError"] is True
    assert "placedBy: range is Customer, got Shipment" in result["content"][0]["text"]


def test_gate_rejects_a_reversed_link_on_its_domain(graph, shapes):
    result = oa.call_tool(graph, shapes, "propose_link", link("cust:8", "placedBy", "order:43"))
    assert result["isError"] is True
    assert "placedBy: domain is Order, got Customer" in result["content"][0]["text"]


def test_gate_rejects_an_instance_that_has_no_row(graph, shapes):
    result = oa.call_tool(graph, shapes, "propose_link", link("order:42", "placedBy", "cust:99"))
    assert result == oa.tool_result("Unknown instance: cust:99", True)


def test_unknown_tool_is_an_error_result(graph, shapes):
    assert oa.call_tool(graph, shapes, "delete_everything", {})["isError"] is True


def test_error_results_have_the_mcp_shape():
    result = oa.tool_result("x", True)
    assert result == {"content": [{"type": "text", "text": "x"}], "isError": True}


# ---------- The demo and the command line ----------

def test_demo_blocks_two_of_five_planner_actions(graph):
    lines = []
    counts = oa.run_demo(graph, oa.load_actions(), out=lines.append)
    assert counts == {"synonyms": 3, "entity_types": 1, "actions": 5, "blocked": 2}
    assert "2 of 5 planner actions blocked before they ran." in lines


def test_main_runs_the_demo_and_exits_0(capsys):
    assert oa.main([]) == 0
    assert "2 of 5 planner actions blocked" in capsys.readouterr().out


def test_main_resolve_prints_the_search_result(capsys):
    assert oa.main(["--resolve", "client"]) == 0
    assert json.loads(capsys.readouterr().out)["entity"] == "Customer"


def test_main_exits_2_on_an_empty_resolve_word(capsys):
    assert oa.main(["--resolve", " "]) == 2


def test_main_exits_2_on_an_unknown_option(capsys):
    assert oa.main(["--nope"]) == 2


def test_main_exits_2_when_the_ontology_is_missing(tmp_path, capsys):
    assert oa.main(["--ontology", str(tmp_path / "missing.ttl")]) == 2
    assert "not found" in capsys.readouterr().err


def test_main_exits_2_on_malformed_turtle(tmp_path, capsys):
    bad = tmp_path / "bad.ttl"
    bad.write_text("this is not turtle", encoding="utf-8")
    assert oa.main(["--ontology", str(bad)]) == 2


def test_main_exits_2_on_an_ontology_without_classes(tmp_path, capsys):
    empty = tmp_path / "empty.ttl"
    empty.write_text("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n", encoding="utf-8")
    assert oa.main(["--ontology", str(empty)]) == 2
    assert "no owl:Class" in capsys.readouterr().err


@pytest.mark.parametrize("content, fragment", [
    ("", "holds no actions"),
    ("not json\n", "not JSON"),
    ('{"arguments": {}}\n', "needs a \"tool\""),
    ("[1, 2]\n", "needs a \"tool\""),
])
def test_main_exits_2_on_a_bad_actions_file(tmp_path, capsys, content, fragment):
    actions = tmp_path / "actions.jsonl"
    actions.write_text(content, encoding="utf-8")
    assert oa.main(["--actions", str(actions)]) == 2
    assert fragment in capsys.readouterr().err


def test_main_exits_2_when_the_actions_file_is_missing(tmp_path, capsys):
    assert oa.main(["--actions", str(tmp_path / "none.jsonl")]) == 2


def test_label_literal_language_tags_still_resolve(graph):
    graph.add((SHOP.Customer, SKOS.altLabel, Literal("patron", lang="en")))
    assert oa.resolve_entity(graph, "patron") == "Customer"
