# Email Assistant

An AI executive-assistant agent that **triages incoming email** and **drafts responses**, built on [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) with long-term memory via [langmem](https://github.com/langchain-ai/langmem).

## How it works

Each email flows through a two-stage LangGraph workflow:

```
                 ┌─────────────────┐
  email_input ──▶│  triage_router  │  classify: ignore / notify / respond
                 └────────┬────────┘
                          │ respond
                          ▼
                 ┌─────────────────┐
                 │  response_agent │  draft reply using tools + memory
                 └────────┬────────┘
                          ▼
                         END
```

1. **Triage** — an LLM with structured output (`Router`) classifies the email as `ignore`, `notify`, or `respond`. Before deciding, it retrieves similar past triage decisions from **episodic memory** and uses them as few-shot examples; after deciding, it records the new decision as an episode. Only `respond` continues; the others end the run.
2. **Response agent** — a tool-calling agent drafts a reply. It can send email, schedule meetings, check calendar availability, and store/recall long-term memory scoped per user. Its instructions come from **procedural memory** and can be rewritten from feedback.

## Requirements

- Python 3.11+
- An OpenAI API key

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** the code targets the LangChain/LangGraph **1.x** stack, pinned in `requirements.txt`:
> `langchain 1.0.5`, `langchain-openai 1.0.2`, `langchain-anthropic 1.4.8`, `langchain-core 1.4.9`, `langgraph 1.0.7`, `langmem 0.0.30`, `pydantic 2.12.4`. Older LangGraph/langmem versions are incompatible (`ImportError: CONFIG_KEY_STORE`).

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

Run the demo:

```bash
python3 main.py
```

Or use the assistant directly:

```python
from assistant import EmailAssistant
from config import profile, prompt_instructions

assistant = EmailAssistant(profile, prompt_instructions, user_id="ibrahim")

# One-off question to the agent
print(assistant.ask("what is my availability for tuesday?"))

# Triage + draft a response to an email
assistant.process_email({
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": "Hi John, could you clarify the missing auth endpoints? Thanks, Alice",
})
```

## Configuration

- **`config.py`** — `profile` (the assistant's owner) and `prompt_instructions` (triage rules and agent instructions). `prompt_instructions` **seeds** procedural memory on first run; after that the live instructions live in the store and can be rewritten via `update_instructions` (see [Procedural memory](#procedural-memory--how-to-behave)).
- **`EmailAssistant(..., model=..., user_id=...)`** — choose the model (default `gpt-4o`) and the memory namespace user (default `"default"`). Each `user_id` gets an isolated memory store.

## Project structure

| File | Responsibility |
|------|----------------|
| `main.py` | Entry point; wires the assistant to sample data and runs a demo |
| `assistant.py` | `EmailAssistant` class — LLM, tools, graph, triage logic, memory |
| `tools.py` | Agent tools: email/meeting/calendar stubs + langmem memory tools |
| `schemas.py` | `Router` (triage output), `TriageRules` (procedural memory), and `State` (graph state) |
| `prompts.py` | Prompt templates for triage and the agent |
| `config.py` | Owner profile and triage/agent instructions |
| `samples.py` | Sample emails for the demo |
| `utils.py` | Email-parsing and few-shot formatting helpers (episodic memory) |

## Long-term memory

The assistant has three complementary kinds of long-term memory, all backed by the same `InMemoryStore` with an OpenAI embedding index and all namespaced per `user_id` so users don't share context.

| Type | Stores | Namespace |
|------|--------|-----------|
| Semantic | facts about people & context | `("email_assistant", user_id, "collection")` |
| Episodic | past triage decisions as examples | `("email_assistant", user_id, "examples")` |
| Procedural | the agent's own instructions & triage rules | `("email_assistant", user_id, "procedures")` |

### Semantic memory — *facts*

The response agent uses langmem's `manage_memory` and `search_memory` tools to store and recall facts about contacts, conversations, and the owner's preferences (e.g. "Alice is the CFO", "Bob prefers morning meetings"). Namespace: `("email_assistant", user_id, "collection")`.

### Episodic memory — *past decisions*

Triage learns from precedent. Each triage decision is stored as an episode (`email → routing`), and every new email retrieves the most semantically similar past episodes to use as few-shot examples. Namespace: `("email_assistant", user_id, "examples")`.

```
new email
   │
   ├─▶ _retrieve_examples()  ──► store.search(..., "examples")   # 3 most similar past emails + routing
   ├─▶ router LLM sees rules + those examples  ──► classification
   └─▶ remember_triage()     ──► store.put(..., "examples")      # this email becomes a future example
```

By default `triage_router` records its own guess as the "correct" label — convenient for the demo, but it will reinforce its own mistakes. In production, record human-verified labels instead:

```python
# after a human corrects a misroute:
assistant.remember_triage(email_input, correct_routing="respond", original_routing="notify")
```

### Procedural memory — *how to behave*

The agent's operating instructions and triage rules live in the store instead of being hard-coded. On first run they are seeded from `prompt_instructions` in `config.py`; the prompts then read the live copy on every call. Feedback rewrites them via an LLM, and the change takes effect immediately (the agent and graph are rebuilt, since `create_agent`'s system prompt is fixed at creation). Namespace: `("email_assistant", user_id, "procedures")`.

```python
# teach the response agent a new behavior
assistant.update_instructions("Always CC my manager sarah@company.com on external replies.")

# adjust triage behavior instead of agent behavior
assistant.update_instructions("Treat recruiter emails as 'notify', not 'respond'.", kind="triage")

# subsequent emails follow the revised instructions
assistant.process_email({...})
```

`InMemoryStore` lives only for the process lifetime, so all three memories reset each run. For persistence across runs, swap it for a durable `BaseStore` (e.g. a Postgres-backed store) in `assistant.py`.

## Notes

- The `write_email`, `schedule_meeting`, and `check_calendar_availability` tools are **placeholder stubs** that return formatted strings. Wire them to real email/calendar services for production use.
- There are currently no automated tests.
