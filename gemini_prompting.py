import os

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel

from google import genai
from google.genai import types

from common import setup_logging, interactive_menu

# Load environment variables
load_dotenv(find_dotenv())

# Logger setup
logger = setup_logging(__name__)

# Rich console
console = Console()


class GeminiClient:
    """
    Simple Gemini AI client.
    """

    def __init__(self, model: str, api_key: str):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=api_key)
        self.model = model

        self.system_prompt = (
            "You are a helpful AI assistant. "
            "Provide clear and concise answers."
        )

    def run(self, prompt: str, system_instruction: str = None) -> str:
        """
        Execute the model with a given prompt and an optional system instruction.
        """
        logger.info(f"Calling model: {self.model}")
        
        # Use provided system instruction or fall back to the default
        active_system_prompt = system_instruction if system_instruction else self.system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,  # <-- INCREASED TO 8192
                system_instruction=active_system_prompt,
            ),
        )

        # Token usage logging
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            logger.info(
                f"Token usage - "
                f"Prompt: {getattr(usage, 'prompt_token_count', 0)}, "
                f"Candidates: {getattr(usage, 'candidates_token_count', 0)}, "
                f"Total: {getattr(usage, 'total_token_count', 0)}"
            )

        # Check if the API cut off the response early (e.g., due to safety or token limits)
        if response.candidates and hasattr(response.candidates[0], "finish_reason"):
            reason = response.candidates[0].finish_reason
            # In the GenAI SDK, finish reasons might be enums or strings. "STOP" means it finished normally.
            if str(reason) not in ["STOP", "FinishReason.STOP", "1"]: 
                console.print(f"[bold yellow]Warning: Output stopped early. Reason: {reason}[/bold yellow]")

        return response.text if hasattr(response, "text") else "No output generated."

SUPPORT_TICKETS = [
    {
        "label": "Ticket 1 — Performance complaint",
        "text": (
            "Subject: App is super slow after the update\n\n"
            "Hi, ever since the latest update the app takes forever to load anything. "
            "Pages that used to be instant now hang for 10+ seconds. I'm on Wi-Fi and "
            "everything else works fine. This is really frustrating — I need this for work. "
            "Can you please fix this ASAP?"
        ),
    },
    {
        "label": "Ticket 2 — Feature not working",
        "text": (
            "Subject: Export button doesn't work\n\n"
            "I've been trying to export my report but nothing happens when I click the "
            "export button. I've tried multiple times. I'm using Chrome on Windows. "
            "My colleague says it works for them but I can't figure out what I'm doing wrong. "
            "Is this a known issue?"
        ),
    },
]

TICKET_LABELS = [t["label"] for t in SUPPORT_TICKETS]

PROMPT_CONFIGS = [
    {
        "label": "A: Generic Assistant",
        "system": "You are a helpful assistant. Help analyze this support ticket.",
    },
    {
        "label": "B: Role-Assigned Expert",
        "system": (
            "You are a senior support engineer at a SaaS company. You've triaged thousands "
            "of tickets. When analyzing tickets, you identify the most likely root cause, "
            "estimate severity, and recommend next steps. You don't hedge — you make a call "
            "based on experience."
        ),
    },
    {
        "label": "C: Role + Constraints + Format",
        "system": (
            "You are a senior support engineer at a SaaS company. You've triaged thousands "
            "of tickets.\n\n"
            "Respond in EXACTLY these sections:\n\n"
            "CATEGORY: Bug / User Error / Feature Request / Configuration\n\n"
            "ROOT CAUSE: One sentence.\n\n"
            "SEVERITY: P1-P4\n\n"
            "NEXT ACTION: One concrete step for the support team.\n\n"
            "Be terse. No explanations beyond what's requested."
        ),
    },
]

CONFIG_LABELS = [c["label"] for c in PROMPT_CONFIGS]

def main() -> None:
    """
    Interactive terminal chat interface.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        console.print(
            Panel(
                "[bold red]GEMINI_API_KEY not found in environment variables.[/bold red]",
                title="Error",
                border_style="red",
            )
        )
        return

    client = GeminiClient(
        model="gemini-2.5-flash",
        api_key=api_key,
    )

    # Welcome screen
    header = Panel(
        "[bold cyan]System Prompts & Role Engineering[/bold cyan]\n\n"
        "Comparing 3 system prompt configurations on support ticket triage.\n"
        "Watch how the response style and actionability change with better prompts.",
        title="Prompt Engineering — Gemini",
    )

    while True:
        try:
            # Step 1: Select a support ticket
            selection = interactive_menu(
                console,
                TICKET_LABELS,
                title="Select a Support Ticket",
                header=header,
                allow_custom=True,
                custom_prompt="Enter a custom support ticket",
            )
            if not selection:
                break

            ticket = next((t for t in SUPPORT_TICKETS if t["label"] == selection), None)
            ticket_text = ticket["text"] if ticket else selection
            ticket_label = ticket["label"] if ticket else "Custom Ticket"
            user_prompt = f"Analyze this support ticket:\n\n{ticket_text}"

            ticket_header = Panel(
                f"[bold magenta]{ticket_label}[/bold magenta]\n[dim]{ticket_text}[/dim]",
                title="Selected Ticket",
                border_style="magenta",
            )

            while True:
                # Step 2: Select a configuration
                config_selection = interactive_menu(
                    console,
                    CONFIG_LABELS,
                    title="Select a Prompt Configuration",
                    header=ticket_header,
                )
                if not config_selection:
                    break

                config = next(c for c in PROMPT_CONFIGS if c["label"] == config_selection)

                console.print(f"\n[bold yellow]━━━ {config['label']} ━━━[/bold yellow]")
                console.print(Panel(config["system"], title="System Prompt", border_style="dim"))

                try:
                    # Pass the selected system prompt into the run method
                    response = client.run(user_prompt, system_instruction=config["system"])
                    console.print(Panel(response, title=config["label"], border_style="green"))
                except Exception as e:
                    logger.error("Error with config %s: %s", config["label"], e)

                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()

        except KeyboardInterrupt:
            console.print(
                "\n[bold red]Interrupted by user.[/bold red]"
            )
            break

        except Exception as e:
            logger.exception("Error occurred")

            console.print(
                Panel(
                    f"[bold red]{str(e)}[/bold red]",
                    title="Error",
                    border_style="red",
                )
            )


if __name__ == "__main__":
    main()