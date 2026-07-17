from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.graph import add_messages



class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

class TriageRules(BaseModel):
    """Editable triage rules stored in procedural memory."""

    ignore: str = Field(description="What kinds of emails to ignore.")
    notify: str = Field(description="What kinds of emails to notify about without replying.")
    respond: str = Field(description="What kinds of emails require a direct response.")

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
