"""EmailAssistant: triages incoming email and drafts responses via a LangGraph workflow."""

import os
import uuid
import warnings

from dotenv import load_dotenv
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore

from schemas import Router, State
from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt_memory
from utils import format_few_shot_examples
from tools import (
    write_email,
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool,
)

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


class EmailAssistant:
    """Executive-assistant agent that triages incoming email and drafts responses."""

    def __init__(self, profile: dict, prompt_instructions: dict, model: str = "gpt-4o", user_id: str = "default"):
        load_dotenv(override=True)
        self.profile = profile
        self.prompt_instructions = prompt_instructions
        self.model = model
        self.user_id = user_id
        # Resolves the `{langgraph_user_id}` placeholder in the memory tool namespaces.
        self.config = {"configurable": {"langgraph_user_id": user_id}}

        # Store backing the long-term memory tools. The index enables semantic
        # search for `search_memory_tool`.
        self.store = InMemoryStore(
            index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        )
        qry = self.store.search(('email_assistant', 'Ibrahim1', 'collection'), query="Alice Smith's")
        print(f"Query: {qry}")
        self.llm = ChatOpenAI(model=model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.llm_router = self.llm.with_structured_output(Router)

        self.tools = [
            write_email,
            schedule_meeting,
            check_calendar_availability,
            manage_memory_tool,
            search_memory_tool,
        ]
        self.agent = create_agent(
            tools=self.tools,
            model=model,
            system_prompt=self._agent_system_prompt(),
            store=self.store,
        )

        self.workflow = self._build_graph()

    # --- Prompt builders --------------------------------------------------
    def _agent_system_prompt(self) -> str:
        return agent_system_prompt_memory.format(instructions=self.prompt_instructions["agent_instructions"], profile=self.profile, **self.profile)

    def _triage_system_prompt(self, examples: str | None = None) -> str:
        return triage_system_prompt.format(
            full_name=self.profile["full_name"],
            name=self.profile["name"],
            user_profile_background=self.profile["user_profile_background"],
            triage_no=self.prompt_instructions["triage_rules"]["ignore"],
            triage_notify=self.prompt_instructions["triage_rules"]["notify"],
            triage_email=self.prompt_instructions["triage_rules"]["respond"],
            examples=examples or "No examples yet.",
        )

    # --- Graph ------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("triage_router", self.triage_router)
        graph.add_node("response_agent", self.agent)
        graph.add_edge(START, "triage_router")
        graph.add_edge("response_agent", END)
        return graph.compile(store=self.store)

    # --- Episodic memory --------------------------------------------------
    def _episodic_namespace(self) -> tuple:
        """Namespace holding past triage episodes for the current user."""
        return ("email_assistant", self.user_id, "examples")

    def _retrieve_examples(self, email_data: dict, limit: int = 3) -> str | None:
        """Fetch the most similar past triage episodes as few-shot examples."""
        query = (
            f"Subject: {email_data['subject']}\n"
            f"From: {email_data['author']}\n"
            f"{email_data['email_thread']}"
        )
        results = self.store.search(self._episodic_namespace(), query=query, limit=limit)
        if not results:
            return None
        return format_few_shot_examples(results)

    def remember_triage(self, email_data: dict, correct_routing: str, original_routing: str | None = None) -> None:
        """Record a triage decision as an episode for future few-shot retrieval.

        Call this after a triage (or after a human correction) so the assistant
        learns from precedent. `correct_routing` is the label that should have
        been used; `original_routing` is what the router first produced.
        """
        value = (
            f"{email_data} "
            f"Original routing: {original_routing or correct_routing} "
            f"Correct routing: {correct_routing}"
        )
        self.store.put(self._episodic_namespace(), key=str(uuid.uuid4()), value={"value": value})

    def triage_router(self, state: State) -> Command[Literal["response_agent", "__end__"]]:
        email_data = state["email_input"]
        user_prompt = triage_user_prompt.format(
            author=email_data["author"],
            to=email_data["to"],
            subject=email_data["subject"],
            email_thread=email_data["email_thread"],
        )
        examples = self._retrieve_examples(email_data)
        result = self.llm_router.invoke(
            [
                {"role": "system", "content": self._triage_system_prompt(examples)},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Record this triage as an episode so future, similar emails retrieve it
        # as a few-shot example. In production, prefer calling `remember_triage`
        # with a human-verified label instead of the router's raw guess.
        self.remember_triage(email_data, correct_routing=result.classification)

        if result.classification == "respond":
            print("📧 Classification: RESPOND - This email requires a response")
            return Command(
                goto="response_agent",
                update={
                    "messages": [
                        {"role": "user", "content": f"Respond to the email {email_data}"}
                    ]
                },
            )
        elif result.classification == "ignore":
            print("🚫 Classification: IGNORE - This email can be safely ignored")
            return Command(goto=END, update=None)
        elif result.classification == "notify":
            # If real life, this would do something else
            print("🔔 Classification: NOTIFY - This email contains important information")
            return Command(goto=END, update=None)
        else:
            raise ValueError(f"Invalid classification: {result.classification}")

    # --- Public API -------------------------------------------------------
    def ask(self, message: str):
        """Run a one-off request against the response agent."""
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]}, config=self.config
        )
        return response["messages"][-1]

    def process_email(self, email_input: dict) -> dict:
        """Triage an email and, when a reply is needed, draft it through the graph."""
        return self.workflow.invoke({"email_input": email_input}, config=self.config)
