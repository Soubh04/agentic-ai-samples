# Ontology in agentic workflows: name the business once

A small Python script that grounds an agent in an ontology, layer by layer, from the LinkedIn post "Ontology in agentic
workflows: name the business once". It loads a Turtle ontology of a fictional shop, resolves synonyms such as "client",
"buyer" and "account" to one `Customer` class, reads each class's data binding, exposes the ontology to the agent
through two tools, generates the tools' JSON Schema enums and ID pattern from the ontology, and runs a scripted
planner's proposed links through SHACL shapes generated from `rdfs:domain` and `rdfs:range`. A rejected link goes back
to the planner as an MCP tool result with `isError: true`, and the scripted planner corrects one of them on a retry.

LinkedIn post: (link added after publishing)

> **This sample calls no cloud service and no model.** The planner is a scripted list of tool calls, and the tables are
> a small in-memory fake. The tools are local Python functions in the shape of MCP tools. They are not the Microsoft
> tools the post names (the Foundry `fabric_iq_preview` tool and the Copilot Studio ontology MCP server), which are in
> preview. This sample was not run against a live Microsoft Fabric, Microsoft Foundry or Copilot Studio resource by the
> author.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.10 or later
- This sample calls no cloud service and needs no credentials. It uses `rdflib`, `pyshacl` and `jsonschema`, which
  `uv` installs from `uv.lock`.

## Setup and run

```bash
uv run python ontology_agent.py                    # the six layers and the five planner actions
uv run python ontology_agent.py --resolve buyer    # resolve one word through the resolve_term tool
```

| Option | Default | Meaning |
|---|---|---|
| `--ontology` | `ontology.ttl` | Turtle ontology: `owl:Class` entities with SKOS labels and `bind:` annotations, `owl:ObjectProperty` relationships with one domain and one range |
| `--actions` | `planner_actions.jsonl` | The planner's tool calls, one JSON object per line with `tool`, `arguments` and an optional `retry` (the corrected arguments the planner sends after an error) |
| `--resolve` | none | Print the `resolve_term` result for one word and exit |

Exit codes: 0 on success, 2 for an invalid argument, ontology file or actions file. A blocked planner action is not an
error: it is the gate doing its job, and the script still exits 0.

What the default run prints:

```text
== 1. Entities ==
  client -> Customer
   buyer -> Customer
 account -> Customer
3 words, 1 entity type.

== 2. Relationships ==
Order --placedBy--> Customer
Shipment --shipsTo--> Customer

== 3. Bindings ==
Customer  customer_id  crm.customers  IDs like cust:<key>
Order     order_id     sales.orders   IDs like order:<key>
Shipment  ship_id      wms.shipments  IDs like ship:<key>
get_entity {"entity": "Customer", "key": "7"} -> Found cust:7 in crm.customers

== 4. Tools ==
list_entity_types -> Customer, Order, Shipment
resolve_term('buyer') -> {"term": "buyer", "entity": "Customer", "relations": ["placedBy", "shipsTo"]}

== 5. Contract ==
get_entity: enums {"entity": ["Customer", "Order", "Shipment"]}, additionalProperties false
propose_link: enums {"relation": ["placedBy", "shipsTo"]}, additionalProperties false
Off-contract call {"subject": "order:42", "relation": "orderedBy", "object": "cust:7"} -> {"resultType": "complete", "content": [{"type": "text", "text": "Invalid arguments: 'orderedBy' is not one of ['placedBy', 'shipsTo']"}], "isError": true}

== 6. Gate ==
[ok] Accepted: order:42 placedBy cust:7
[ x] Rejected order:43 placedBy ship:9: placedBy: range is Customer, got Shipment
     -> {"isError": true} back to the planner
     planner retries: Accepted: order:43 placedBy cust:8
[ok] Accepted: ship:9 shipsTo cust:7
[ x] Rejected cust:8 placedBy order:43: placedBy: domain is Order, got Customer; placedBy: range is Customer, got Order
     -> {"isError": true} back to the planner
[ok] Accepted: ship:10 shipsTo cust:8
2 of 5 planner actions blocked before they ran; 1 corrected on retry.
```

## Post to code

| Post | What the script does | Code |
|---|---|---|
| 1. Entities: one concept, named once; OWL and SKOS | `ontology.ttl` declares one `owl:Class` per concept; synonyms are `skos:altLabel` values on that class. `resolve_entity()` maps "client", "buyer" and "account" to `Customer`; `check_ontology()` refuses a label shared by two classes | `ontology.ttl`, `load_ontology()`, `check_ontology()`, `entity_types()`, `resolve_entity()` |
| 2. Relationships: domain and range; no unique name assumption | `placedBy` and `shipsTo` carry exactly one `rdfs:domain` and one `rdfs:range`, both declared classes; anything else is an error. In RDFS, domain and range only infer types and reject nothing: the gate in row 6 turns them into checks | `relations()` |
| 3. Bindings: a virtual layer over the source tables | Each class names its table, key column and ID prefix. `type_of()` and `get_entity()` read the row through the binding at call time; nothing is copied. Two classes may not share an ID prefix | `bindings()`, `type_of()`, `get_entity()` |
| 4. Tools: agents read the ontology through tools | `list_entity_types()` and `resolve_term()` return the entity types, synonyms, tables and relationships. `call_tool()` answers an unknown tool with a JSON-RPC protocol error (`-32602`), not a tool result | `list_entity_types()`, `resolve_term()`, `call_tool()`, `protocol_error()` |
| 5. Contract: JSON Schema `inputSchema`, servers validate inputs, strict mode | `build_tool_schemas()` generates the `entity` and `relation` enums and the instance ID pattern from the ontology, with `additionalProperties: false`. `check_arguments()` validates every call on the server side, so an off-contract relation such as `orderedBy` is rejected even without strict mode on the model | `build_tool_schemas()`, `check_arguments()`, `call_tool()` |
| 6. Gate: check before acting; the tool error goes back to the model | `build_shapes()` turns each relationship's domain and range into two SHACL node shapes. `gate()` validates the proposed link with `pyshacl` before anything is written and returns an MCP `CallToolResult` with `isError: true` and the reason when it fails. The scripted planner sends its corrected arguments once, after an error | `build_shapes()`, `gate()`, `tool_result()`, `run_demo()` |

The image's summary line maps to the script like this:

| Image | In the script |
|---|---|
| client, buyer, account ▸ 1 Customer | Section 1 of the run; `test_three_synonyms_resolve_to_one_entity` |
| 5 planner actions, 2 bad links blocked | Section 6 of the run: a wrong range (`order:43 placedBy ship:9`) and a reversed link (`cust:8 placedBy order:43`); `test_demo_blocks_two_of_five_planner_actions_and_the_planner_corrects_one` |
| Tool schema generated, never typed by hand | `build_tool_schemas()`; `test_a_new_class_and_relation_reach_the_enums_without_editing_code` |

The MCP result shape follows the
[MCP 2026-07-28 tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools). The
ontology uses [OWL 2](https://www.w3.org/TR/owl2-overview/), [RDFS](https://www.w3.org/TR/rdf-schema/) and
[SKOS](https://www.w3.org/TR/skos-reference/); the gate uses [SHACL](https://www.w3.org/TR/shacl/).

## What the script leaves out

- A model and a real planner. The five tool calls and the one retry are scripted, so the run is the same every time. With a model, pass
  the generated `inputSchema` as the tool definition and turn on strict tool use (Claude) or strict function calling
  (Azure OpenAI) so the model stays inside the enums; keep `check_arguments()` on the server anyway.
- Microsoft Fabric IQ. The sample does not call the Fabric IQ ontology, the Foundry `fabric_iq_preview` tool or the
  Copilot Studio ontology MCP server. Fabric IQ can import an RDF or OWL ontology; check its preview limits first.
- An MCP server. The tools are plain functions that return the MCP `CallToolResult` shape; wrap them in your own MCP
  server to expose them to agents.
- Identity across sources. OWL has no unique name assumption, and `owl:sameAs` or `SameIndividual` can state that two
  IDs are one customer. The sample gives each instance one ID.
- Inference. The gate runs SHACL with `inference="none"`: instance types come from the bindings, not from reasoning
  over the domain and range. The ontology is passed to `pyshacl` as `ont_graph`, so `rdfs:subClassOf` still counts: an
  instance of a subclass of `Customer` passes a range of `Customer`.
- Writing the accepted links anywhere. An accepted call returns a success result only.

## Tests

```bash
uv run pytest -q
```

```text
.....................................................................    [100%]
69 passed in 0.35s
```

The tests run offline. They cover the golden path; empty and unknown words; an ontology with a label shared by two
classes, a shared ID prefix, a property without one domain and one range, or a range that is not a declared class;
subclasses at the gate; instances with no row; arguments outside the contract; an unknown tool as a JSON-RPC error; the
planner's retry; and exit code 2 for a missing or malformed ontology file, a bad or unreadable actions file and invalid
options. Ten deliberate breaks were each caught before the code was restored, among them a gate that skips SHACL (4
tests fail), a resolver that ignores `skos:altLabel` (11 fail), a hand-typed relation enum or ID pattern (1 each), a
gate without the ontology graph (1), and a planner that never retries (1).

All data is fictional: the shop, its tables, IDs and rows do not describe any real organization.
