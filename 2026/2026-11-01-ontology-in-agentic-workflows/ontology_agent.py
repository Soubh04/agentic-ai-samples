"""Ground an agent in an ontology, in six layers, in one small offline script.

The six layers from the LinkedIn post "Ontology in agentic workflows: name the business once":
  entities   one OWL class per business concept; synonyms are SKOS altLabels (ontology.ttl)
  relations  rdfs:domain and rdfs:range say which entity may link to which
  bindings   each class names its source table, key column and ID prefix; rows stay in the tables
  tools      the agent reads the ontology through tools (list_entity_types, search_ontology), not a pasted glossary
  contract   the tool's JSON Schema enums are generated from the ontology, with additionalProperties false
  gate       SHACL shapes, generated from domain and range, check each proposed link before it is written;
             a failure goes back to the planner as an MCP tool result with isError true

The planner is scripted (planner_actions.jsonl) and the tables are a small in-memory fake.
No model, no network, no cloud service.

    uv run python ontology_agent.py
    uv run python ontology_agent.py --resolve buyer

Exit codes: 0 success, 2 invalid argument, ontology file or actions file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from pyshacl import validate
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, SKOS

HERE = Path(__file__).parent
ONTOLOGY_FILE = HERE / "ontology.ttl"
ACTIONS_FILE = HERE / "planner_actions.jsonl"
SHOP = Namespace("https://example.org/shop#")
BIND = Namespace("https://example.org/bind#")
DATA = Namespace("https://example.org/data/")
SHAPES = Namespace("https://example.org/shapes#")

# Fictional rows, keyed by table, then by key value. The ontology binds to these tables; it never copies them.
FAKE_TABLES: dict[str, set[str]] = {
    "crm.customers": {"7", "8"},
    "sales.orders": {"42", "43"},
    "wms.shipments": {"9"},
}


class OntologyError(Exception):
    """The ontology file is missing, unreadable or incomplete."""


@dataclass(frozen=True)
class Binding:
    entity: str
    table: str
    key: str
    id_prefix: str


@dataclass(frozen=True)
class Relation:
    name: str
    domain: str
    range: str


# ---------- 1. Entities ----------

def load_ontology(path: Path = ONTOLOGY_FILE) -> Graph:
    """Parse a Turtle ontology; raise OntologyError when it is missing, malformed or has no classes."""
    graph = Graph()
    try:
        graph.parse(path, format="turtle")
    except FileNotFoundError as exc:
        raise OntologyError(f"ontology file not found: {path}") from exc
    except Exception as exc:  # rdflib raises several parser exception types
        raise OntologyError(f"cannot parse {path}: {exc}") from exc
    if not entity_types(graph):
        raise OntologyError(f"{path} declares no owl:Class")
    return graph


def local_name(uri: URIRef) -> str:
    return str(uri).split("#")[-1]


def entity_types(graph: Graph) -> list[str]:
    """Every owl:Class in the shop namespace, sorted by name."""
    return sorted(local_name(c) for c in graph.subjects(RDF.type, OWL.Class)
                  if isinstance(c, URIRef) and str(c).startswith(str(SHOP)))


def resolve_entity(graph: Graph, term: str) -> str | None:
    """Map a user's word to one entity type through its name, prefLabel or altLabels; None when nothing matches."""
    wanted = term.strip().casefold()
    if not wanted:
        return None
    for name in entity_types(graph):
        cls = SHOP[name]
        labels = {name, *(str(v) for p in (SKOS.prefLabel, SKOS.altLabel) for v in graph.objects(cls, p))}
        if wanted in {label.casefold() for label in labels}:
            return name
    return None


# ---------- 2. Relationships ----------

def relations(graph: Graph) -> list[Relation]:
    """Every owl:ObjectProperty with exactly one domain and one range, sorted by name."""
    found = []
    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        domains, ranges = list(graph.objects(prop, RDFS.domain)), list(graph.objects(prop, RDFS.range))
        if len(domains) != 1 or len(ranges) != 1:
            raise OntologyError(f"{local_name(prop)} needs exactly one rdfs:domain and one rdfs:range")
        found.append(Relation(local_name(prop), local_name(domains[0]), local_name(ranges[0])))
    return sorted(found, key=lambda r: r.name)


# ---------- 3. Bindings ----------

def bindings(graph: Graph) -> dict[str, Binding]:
    """Each entity type's source table, key column and ID prefix, read from the ontology's bind: annotations."""
    out = {}
    for name in entity_types(graph):
        cls = SHOP[name]
        values = {p: graph.value(cls, BIND[p]) for p in ("table", "key", "idPrefix")}
        missing = [p for p, v in values.items() if v is None]
        if missing:
            raise OntologyError(f"{name} has no binding for {', '.join(missing)}")
        out[name] = Binding(name, str(values["table"]), str(values["key"]), str(values["idPrefix"]))
    return out


def type_of(instance_id: str, binds: Mapping[str, Binding], tables: Mapping[str, set[str]]) -> str | None:
    """The entity type of an ID such as "order:42": its prefix picks the binding, and the row must exist."""
    prefix, sep, key = instance_id.partition(":")
    if not sep or not key:
        return None
    for b in binds.values():
        if b.id_prefix == prefix and key in tables.get(b.table, set()):
            return b.entity
    return None


# ---------- 4. Tools: the agent reads the ontology, it is never pasted into a prompt ----------

def list_entity_types(graph: Graph) -> list[dict[str, Any]]:
    """Tool: the entity types an agent may use, with their synonyms and bound tables."""
    binds = bindings(graph)
    return [{"entity": name,
             "synonyms": sorted(str(v) for v in graph.objects(SHOP[name], SKOS.altLabel)),
             "table": binds[name].table}
            for name in entity_types(graph)]


def search_ontology(graph: Graph, term: str) -> dict[str, Any]:
    """Tool: resolve a word to an entity type and list the relationships it takes part in."""
    entity = resolve_entity(graph, term)
    if entity is None:
        return {"term": term, "entity": None, "relations": []}
    rels = [r.name for r in relations(graph) if entity in (r.domain, r.range)]
    return {"term": term, "entity": entity, "relations": rels}


# ---------- 5. Contract: the tool schema is generated from the ontology ----------

def build_tool_schemas(graph: Graph) -> dict[str, dict[str, Any]]:
    """MCP tool definitions whose enums come from the ontology. Nothing here is typed by hand."""
    entities = entity_types(graph)
    rel_names = [r.name for r in relations(graph)]
    id_pattern = "^[a-z]+:[0-9A-Za-z_-]+$"
    return {
        "get_entity": {
            "name": "get_entity",
            "description": "Look up one business entity by its type and key.",
            "inputSchema": {
                "type": "object",
                "properties": {"entity": {"type": "string", "enum": entities},
                               "key": {"type": "string", "minLength": 1}},
                "required": ["entity", "key"],
                "additionalProperties": False,
            },
        },
        "propose_link": {
            "name": "propose_link",
            "description": "Propose one relationship between two entities, for example an order placed by a customer.",
            "inputSchema": {
                "type": "object",
                "properties": {"subject": {"type": "string", "pattern": id_pattern},
                               "relation": {"type": "string", "enum": rel_names},
                               "object": {"type": "string", "pattern": id_pattern}},
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
            },
        },
    }


def check_arguments(tool: Mapping[str, Any], arguments: Any) -> list[str]:
    """Server-side input validation against the tool's inputSchema; an empty list means valid."""
    validator = Draft202012Validator(tool["inputSchema"])
    return sorted(e.message for e in validator.iter_errors(arguments))


# ---------- 6. Gate: SHACL shapes from domain and range, checked before anything is written ----------

def build_shapes(graph: Graph) -> Graph:
    """Two SHACL node shapes per relationship: the subject must be its domain, the value must be its range."""
    shapes = Graph()
    shapes.bind("sh", SH)
    for rel in relations(graph):
        prop = SHOP[rel.name]
        domain_shape, range_shape = SHAPES[f"{rel.name}Domain"], SHAPES[f"{rel.name}Range"]
        shapes.add((domain_shape, RDF.type, SH.NodeShape))
        shapes.add((domain_shape, SH.targetSubjectsOf, prop))
        shapes.add((domain_shape, SH["class"], SHOP[rel.domain]))
        shapes.add((domain_shape, SH.message, Literal(f"{rel.name}: domain is {rel.domain}")))
        prop_shape = BNode()
        shapes.add((range_shape, RDF.type, SH.NodeShape))
        shapes.add((range_shape, SH.targetSubjectsOf, prop))
        shapes.add((range_shape, SH.property, prop_shape))
        shapes.add((prop_shape, SH.path, prop))
        shapes.add((prop_shape, SH["class"], SHOP[rel.range]))
        shapes.add((prop_shape, SH.message, Literal(f"{rel.name}: range is {rel.range}")))
    return shapes


def tool_result(text: str, is_error: bool) -> dict[str, Any]:
    """An MCP CallToolResult: an error goes back to the model as content, so it can correct the call."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def instance_uri(instance_id: str) -> URIRef:
    prefix, _, key = instance_id.partition(":")
    return DATA[f"{prefix}/{key}"]


def gate(graph: Graph, shapes: Graph, tool: Mapping[str, Any], arguments: Any,
         tables: Mapping[str, set[str]] = FAKE_TABLES) -> dict[str, Any]:
    """Validate one propose_link call: the schema first, then the bindings, then the SHACL shapes."""
    problems = check_arguments(tool, arguments)
    if problems:
        return tool_result("Invalid arguments: " + "; ".join(problems), True)
    subject, rel, obj = arguments["subject"], arguments["relation"], arguments["object"]
    binds = bindings(graph)
    types = {i: type_of(i, binds, tables) for i in (subject, obj)}
    unknown = [i for i, t in types.items() if t is None]
    if unknown:
        return tool_result("Unknown instance: " + ", ".join(unknown), True)

    data = Graph()
    for instance_id, entity in types.items():
        data.add((instance_uri(instance_id), RDF.type, SHOP[entity]))
    data.add((instance_uri(subject), SHOP[rel], instance_uri(obj)))
    conforms, report, _ = validate(data, shacl_graph=shapes, ont_graph=graph, inference="none")
    if conforms:
        return tool_result(f"Accepted: {subject} {rel} {obj}", False)
    messages = sorted({str(m) for m in report.objects(None, SH.resultMessage)})
    detail = "; ".join(_with_actual(m, types[subject], types[obj]) for m in messages)
    return tool_result(f"Rejected {subject} {rel} {obj}: {detail}", True)


def _with_actual(message: str, subject_type: str, object_type: str) -> str:
    """Add the type that was actually found to a shape message such as "placedBy: range is Customer"."""
    got = subject_type if ": domain is " in message else object_type
    return f"{message}, got {got}"


def get_entity(graph: Graph, tool: Mapping[str, Any], arguments: Any,
               tables: Mapping[str, set[str]] = FAKE_TABLES) -> dict[str, Any]:
    """Validate one get_entity call, then read the row through the entity's binding."""
    problems = check_arguments(tool, arguments)
    if problems:
        return tool_result("Invalid arguments: " + "; ".join(problems), True)
    b = bindings(graph)[arguments["entity"]]
    if arguments["key"] not in tables.get(b.table, set()):
        return tool_result(f"No {b.entity} with {b.key} {arguments['key']} in {b.table}", True)
    return tool_result(f"Found {b.id_prefix}:{arguments['key']} in {b.table}", False)


def call_tool(graph: Graph, shapes: Graph, name: str, arguments: Any,
              tables: Mapping[str, set[str]] = FAKE_TABLES) -> dict[str, Any]:
    """The tool server: every call is checked against its generated schema before it touches data."""
    tools = build_tool_schemas(graph)
    if name == "get_entity":
        return get_entity(graph, tools[name], arguments, tables)
    if name == "propose_link":
        return gate(graph, shapes, tools[name], arguments, tables)
    return tool_result(f"Unknown tool: {name}", True)


# ---------- The demo ----------

def load_actions(path: Path = ACTIONS_FILE) -> list[dict[str, Any]]:
    """Read the scripted planner's tool calls, one JSON object per line."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"actions file not found: {path}") from exc
    actions = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            action = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not JSON ({exc.msg})") from exc
        if not isinstance(action, dict) or not isinstance(action.get("tool"), str) or "arguments" not in action:
            raise ValueError(f"{path}:{number}: needs a \"tool\" name and \"arguments\"")
        actions.append(action)
    if not actions:
        raise ValueError(f"{path} holds no actions")
    return actions


def run_demo(graph: Graph, actions: Sequence[Mapping[str, Any]], out=print) -> dict[str, int]:
    """Print the six layers and run every planner action through the gate; return the counts."""
    out("== 1. Entities ==")
    synonyms = ["client", "buyer", "account"]
    resolved = {term: resolve_entity(graph, term) for term in synonyms}
    for term, entity in resolved.items():
        out(f"{term:>8} -> {entity}")
    targets = set(resolved.values()) - {None}
    out(f"{len(synonyms)} words, {len(targets)} entity type{'s' if len(targets) != 1 else ''}.")

    out("\n== 2. Relationships ==")
    for rel in relations(graph):
        out(f"{rel.domain} --{rel.name}--> {rel.range}")

    out("\n== 3. Bindings ==")
    for b in bindings(graph).values():
        out(f"{b.entity:<9} {b.key:<12} {b.table:<14} IDs like {b.id_prefix}:<key>")
    lookup = {"entity": "Customer", "key": "7"}
    found = get_entity(graph, build_tool_schemas(graph)["get_entity"], lookup)
    out("get_entity " + json.dumps(lookup) + " -> " + found["content"][0]["text"])

    out("\n== 4. Tools ==")
    out("list_entity_types -> " + ", ".join(e["entity"] for e in list_entity_types(graph)))
    out("search_ontology('buyer') -> " + json.dumps(search_ontology(graph, "buyer")))

    out("\n== 5. Contract ==")
    tools = build_tool_schemas(graph)
    for tool in tools.values():
        props = tool["inputSchema"]["properties"]
        enums = {k: v["enum"] for k, v in props.items() if "enum" in v}
        out(f"{tool['name']}: enums {json.dumps(enums)}, additionalProperties false")
    shapes = build_shapes(graph)
    off_contract = {"subject": "order:42", "relation": "orderedBy", "object": "cust:7"}
    out("Off-contract call " + json.dumps(off_contract) + " -> "
        + json.dumps(call_tool(graph, shapes, "propose_link", off_contract)))

    out("\n== 6. Gate ==")
    blocked = 0
    for action in actions:
        result = call_tool(graph, shapes, action["tool"], action["arguments"])
        blocked += result["isError"]
        mark = "x" if result["isError"] else "ok"
        out(f"[{mark:>2}] {result['content'][0]['text']}")
        if result["isError"]:
            out("     -> " + json.dumps({"isError": True}) + " back to the planner")
    out(f"{blocked} of {len(actions)} planner actions blocked before they ran.")
    return {"synonyms": len(synonyms), "entity_types": len(targets), "actions": len(actions), "blocked": blocked}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ontology", type=Path, default=ONTOLOGY_FILE, help="Turtle ontology file")
    parser.add_argument("--actions", type=Path, default=ACTIONS_FILE, help="planner tool calls, one JSON per line")
    parser.add_argument("--resolve", metavar="WORD", help="only resolve one word to an entity type and exit")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on a bad argument and 0 on --help
        return int(exc.code or 0)
    try:
        graph = load_ontology(args.ontology)
        relations(graph), bindings(graph)  # fail early on an incomplete ontology
        if args.resolve is not None:
            if not args.resolve.strip():
                print("error: --resolve needs a word", file=sys.stderr)
                return 2
            print(json.dumps(search_ontology(graph, args.resolve)))
            return 0
        actions = load_actions(args.actions)
    except (OntologyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run_demo(graph, actions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
