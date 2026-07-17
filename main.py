"""Entry point. Wires the assistant to sample data and runs a demo."""

from assistant import EmailAssistant
from config import profile, prompt_instructions
from samples import email_input, email_input1


def main():
    assistant = EmailAssistant(profile, prompt_instructions, user_id="Ibrahim1")
    print(assistant.ask("what is my availability for tuesday?").pretty_print())
    response = assistant.process_email(email_input)
    for m in response["messages"]:
         m.pretty_print()

    email_input2 = {
            "author": "Alice Smith <alice.smith@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Follow up",
            "email_thread": """Hi John,

            Any update on my previous ask?""",
    }
    
    response2 = assistant.process_email(email_input2)
    for m in response2["messages"]:
        m.pretty_print()

    #response = assistant.process_email(email_input1)

if __name__ == "__main__":
    main()
