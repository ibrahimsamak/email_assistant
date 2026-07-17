"""Static configuration: the assistant's owner profile and triage rules."""

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": (
        "Use these tools to manage John's tasks efficiently. Follow this exact sequence "
        "for every email, without asking the user for confirmation at any step:\n"
        "1. FIRST call search_memory to look up prior context about the sender.\n"
        "2. THEN send the reply by actually calling the write_email tool (do not just "
        "draft it or ask whether to send it -- send it directly).\n"
        "3. FINALLY call manage_memory to store the key facts from this email (the "
        "sender, what they asked, and any commitments made). This step is mandatory on "
        "every email so future emails can reference it."
    ),
}
