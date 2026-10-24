# Agent memory: short-term vs long-term

A small Python script that models the two kinds of agent memory from the LinkedIn post "Agent memory: short-term vs
long-term". Short-term memory is the session transcript, and only the newest turns that fit a token budget reach the
model. Long-term memory is a set of distilled facts kept per user across sessions: extracted, consolidated and
retrieved. The demo runs two sessions for a fictional user, then shows scope isolation, expiry and deletion.

LinkedIn post: (link added after publishing)

> **This sample calls no cloud service and no model.** Extraction is three plain rules, retrieval is word overlap,
> and the store lives in process memory, so everything is gone when the script exits. It teaches the shape of the
> two memories. It does not reproduce how Microsoft Foundry Memory, LangGraph or the Claude memory tool work inside.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.10 or later
- Nothing else: no cloud resource, no credentials, no packages beyond the standard library

## Setup and run

```bash
uv run python agent_memory.py
uv run python agent_memory.py --token-budget 25   # a smaller window: more of the transcript falls out of view
```

`--token-budget` must be a whole number of at least 1. Anything else exits with code 2. Tokens are counted as words.

What a run prints:

```text
== Session 1 (user-asha), token budget 40 ==
Transcript holds 9 turns; the model sees the newest 6.
Session ends. Extract and consolidate: {'added': 3, 'unchanged': 0, 'updated': 0}
Long-term memory: allergy=peanuts, city=Pune, name=Asha Verma

== Session 2: a new thread for the same user ==
Transcript at the start: 0 turns.
Prompt the model would see:
SYSTEM: You are a helpful assistant.
KNOWN ABOUT THIS USER (long-term memory):
- allergy: peanuts
- city: Pune
- name: Asha Verma
CONVERSATION (short-term memory):
user: What is my name?
Session ends. Extract and consolidate: {'added': 0, 'unchanged': 0, 'updated': 1}
Long-term memory: city=Hyderabad, allergy=peanuts, name=Asha Verma

== Scope: another user sees nothing ==
user-ravi: nothing

== Expiry: a 60 second TTL on a fake clock ==
At 0 s: city=Pune
At 61 s: nothing

== Delete: forget one user ==
Removed 3 memories for user-asha.
user-asha: nothing
```

## Post to code

| Post | What the script does | Code |
|---|---|---|
| Short-term: the session | Keeps the whole transcript. The model sees only the newest turns that fit the token budget, and the stored transcript is never cut | `ShortTermMemory`, `window()` |
| Why short-term runs out | Replaying the full transcript grows the prompt every turn. The windowed prompt stops growing once the budget is full | `build_prompt()`, `Session.prompt()` |
| Long-term: what survives | Facts outlive the session. They are extracted from user turns, consolidated, and retrieved in a later session | `extract_facts()`, `LongTermMemory.remember()`, `LongTermMemory.retrieve()` |
| How they work together | Memories are loaded once at the start of a session. Anything the latest user message matches is added on each turn. New facts are extracted when the session ends | `Session`, `Session.end()` |
| Consolidation | A repeated fact is stored once. A newer value for the same key replaces the older one | `LongTermMemory.remember()` |
| Which one gets the job: delete | One scope per user, an optional expiry, and deletion of everything one user has. A fact deleted or expired while a session is open drops out of that session's next prompt | `LongTermMemory(ttl_seconds=...)`, `forget_scope()`, `LongTermMemory.get()` |
| One caution: a write path | Only user turns are read for facts. Tool output and assistant text never write to memory | `extract_facts()` |

The names scope, TTL and delete-by-scope follow what the Foundry Memory documentation calls them. See
[What is Memory?](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/what-is-memory) and
[Create and use memory in Foundry Agent Service](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/memory-usage).
Foundry Memory is in preview, so check those pages for the current behaviour.

## What the script leaves out

- A model. A real extractor is an LLM that reads the conversation and decides what is worth keeping, and a real
  consolidation step merges overlapping facts. Here, three regular expressions and "one value per key" stand in.
- Good extraction. The three rules are naive: "My order is late." is stored as `order: late`, and "I live in Pune and
  my name is Asha" becomes one city value. A real system uses an LLM here.
- Embeddings. Retrieval here is word overlap, so "Which city do I live in?" matches a memory keyed `city` and a
  paraphrase with no shared word does not.
- Real token counts. One word counts as one token.
- Procedural memory (how-to routines learned from past interactions), summaries of earlier sessions, and
  summarising or clearing old tool results to stay inside the window.
- Deletion during an open session. `forget_scope()` empties the store and the open session's next prompt, but that
  session's transcript still holds what the user said, so end it without calling `Session.end()`.
- Persistence, concurrent writers, and protection against a user who states something false. The user-turns-only
  rule is a toy guard. The Foundry documentation names prompt injection and memory corruption as risks and points to
  prompt-injection detection and adversarial testing.

## Tests

```bash
uv run pytest -q
```

They run offline with a fake clock, so no network and no credentials. They cover the window keeping the newest turns in
order, the stored transcript staying whole, an empty transcript, a newest turn bigger than the budget, a new session
starting with an empty transcript, the full replay growing while the windowed prompt does not, extraction from user
turns only, key normalisation, duplicate and conflicting facts, ranked retrieval, scope isolation, deletion of one
scope, expiry and its reset, deletion and expiry while a session is open, a fact crossing from session one to session
two, and the exit code 2 for an invalid argument.

```text
........................................                                 [100%]
40 passed in 0.02s
```

All data is fictional.
