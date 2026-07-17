"""Agent tools. Placeholder implementations - a real app would call out to
email / calendar services here."""

from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool

# Long-term memory tools. `{langgraph_user_id}` is resolved at runtime from the
# invocation config, so each user gets an isolated memory namespace. Both tools
# require a BaseStore to be attached to the graph/agent at compile time.
manage_memory = create_manage_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)
search_memory = create_search_memory_tool(
    namespace=("email_assistant", "{langgraph_user_id}", "collection")
)

@tool
def write_email(to: str, subject: str, content: str):
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"


@tool
def schedule_meeting(attendees: list[str], subject: str, duration_minutes: int, preferred_day: str) -> str:
    """Schedule a calendar meeting."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
