"""
Few-Shot & Chain-of-Thought Prompting (Gemini)

Demonstrates three prompting techniques, each on a task where it shines:
  1. Zero-shot  — sentiment analysis (well-understood task, no examples needed)
  2. Few-shot   — classification with custom domain labels (teaches YOUR taxonomy)
  3. Chain-of-thought — root cause analysis (multi-step reasoning needed)

Each demo shows WHY you'd pick that technique over the others.
"""

import csv
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from common import setup_logging, interactive_menu

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Paths — logs/ and token_usage.csv sit next to this file
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_LOG_DIR = _HERE / "logs"
_TOKEN_CSV = _HERE / "logs" / "token_usage.csv"
_LOG_FILE = _LOG_DIR / "gemini_few_shot_cot.log"

_LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — Rich handler (terminal) + FileHandler (logs/)
# ---------------------------------------------------------------------------

logger = setup_logging(__name__)          # Rich terminal handler via common

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
)
logging.getLogger().addHandler(_file_handler)

logger.info("Log file: %s", _LOG_FILE)

# ---------------------------------------------------------------------------
# CSV — write header if file is new
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    "timestamp",
    "demo",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

if not _TOKEN_CSV.exists():
    with _TOKEN_CSV.open("w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(_CSV_HEADERS)


def _append_token_row(
    demo: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Append one row to token_usage.csv."""
    with _TOKEN_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            demo,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ])

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

# Demo A: Zero-Shot — Sentiment Analysis
REVIEWS = [
    "This laptop is incredible — fast, lightweight, and the battery lasts all day.",
    "The charging cable broke after two weeks. Total waste of money.",
    "It's fine for the price. Nothing special but gets the job done.",
]

# Demo B: Few-Shot — Custom Domain Labels
FEW_SHOT_EXAMPLES = [
    ("I was charged twice for the same subscription", "BILLING_DISPUTE"),
    ("Can't log in even after resetting my password three times", "ACCOUNT_ACCESS"),
    ("The export function crashes when the report has more than 1000 rows", "TECHNICAL_BUG"),
    ("It would be great if we could schedule reports to run automatically", "FEATURE_REQUEST"),
]

FEW_SHOT_TEST_INPUTS = [
    "My invoice shows a charge from last month that I already disputed",
    "The dashboard keeps showing a spinning wheel and never loads the charts",
    "Would love to be able to tag tickets with custom labels for our team",
]

# Demo C: Chain-of-Thought — Root Cause Analysis
BUG_REPORT = (
    "Users report that the app works fine in the morning but becomes extremely slow "
    "after lunch. The slowdown affects all users simultaneously, not just individual "
    "sessions. Restarting the app server temporarily fixes the issue but it returns "
    "within a few hours. Memory usage on the server appears normal."
)

# Base inter-call delay (seconds) — minimum backoff floor for tenacity.
CALL_DELAY_SECONDS: float = 1.0

# Daily-quota error signature — these are NOT retryable within a session.
_DAILY_QUOTA_ID = "GenerateRequestsPerDayPerProjectPerModel"


def _is_retryable_429(exc: BaseException) -> bool:
    """Return True only for per-minute 429s; let daily quota errors propagate."""
    msg = str(exc)
    return "429" in msg and _DAILY_QUOTA_ID not in msg

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ZERO_SHOT_INSTRUCTIONS = (
    "Classify the sentiment of the following product review.\n"
    "Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."
)

FEW_SHOT_INSTRUCTIONS_TEMPLATE = (
    "Classify support tickets into one of these categories: "
    "BILLING_DISPUTE, ACCOUNT_ACCESS, TECHNICAL_BUG, FEATURE_REQUEST\n\n"
    "Examples:\n\n{examples}\n\n"
    "Respond with ONLY the category name."
)

COT_INSTRUCTIONS = (
    "You are a senior engineer. Analyze this bug report step by step:\n"
    "1. What patterns do you observe? (timing, scope, triggers)\n"
    "2. What does each clue rule in or rule out?\n"
    "3. What is the most likely root cause?\n"
    "4. What would you check first to confirm?\n\n"
    "Think through each step before concluding."
)

ZERO_SHOT_BASELINE_INSTRUCTIONS = (
    "You are a senior engineer. "
    "Identify the most likely root cause of this bug.\n"
    "Be concise — one or two sentences."
)

# ---------------------------------------------------------------------------
# Demo labels (used by the interactive menu and demo dispatch table)
# ---------------------------------------------------------------------------

DEMO_LABELS = [
    "A: Zero-Shot — Sentiment Analysis",
    "B: Few-Shot — Custom Label Classification",
    "C: Chain-of-Thought — Root Cause Analysis",
]

# ---------------------------------------------------------------------------
# Token tracker
# ---------------------------------------------------------------------------


class GeminiTokenTracker:
    """Track Gemini token usage across requests."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        # Per-call snapshots for CSV rows: (demo, prompt, completion, total)
        self._pending: list[tuple[str, int, int, int]] = []

    def track(self, usage_metadata, demo: str = "") -> None:
        """Accumulate token counts and queue a CSV row."""
        prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total = getattr(usage_metadata, "total_token_count", 0) or 0

        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total

        self._pending.append((demo, prompt, completion, total))
        logger.debug(
            "Token usage — demo=%s prompt=%d completion=%d total=%d",
            demo or "unknown",
            prompt,
            completion,
            total,
        )

    def flush_csv(self) -> None:
        """Write all queued rows to token_usage.csv and clear the queue."""
        for demo, prompt, completion, total in self._pending:
            _append_token_row(demo, self.model, prompt, completion, total)
        self._pending.clear()

    def report(self, console: Console) -> None:
        """Print a token usage summary table."""
        table = Table(title="Token Usage Summary")
        table.add_column("Type", style="cyan")
        table.add_column("Tokens", style="green")
        table.add_row("Prompt Tokens", str(self.total_prompt_tokens))
        table.add_row("Completion Tokens", str(self.total_completion_tokens))
        table.add_row("Total Tokens", str(self.total_tokens))
        console.print(table)

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

# ---------------------------------------------------------------------------
# Prompting client
# ---------------------------------------------------------------------------


class PromptingClient:
    """Demonstrates zero-shot, few-shot, and chain-of-thought prompting."""

    def __init__(self, model: str, api_key: str, token_tracker: GeminiTokenTracker) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.token_tracker = token_tracker

    @retry(
        retry=retry_if_exception(_is_retryable_429),
        wait=wait_exponential(multiplier=1, min=CALL_DELAY_SECONDS, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call(
        self,
        instructions: str,
        user_input: str,
        max_tokens: int = 256,
        demo: str = "",
    ) -> str:
        """Make a single API call with automatic retry on transient 429 errors."""
        logger.info("API call — model=%s demo=%s", self.model, demo or "unknown")

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_input,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
                system_instruction=instructions,
            ),
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.token_tracker.track(response.usage_metadata, demo=demo)

        return response.text.strip() if hasattr(response, "text") else ""

    # --- Zero-Shot ---
    def classify_sentiment(self, review: str) -> str:
        """Classify product-review sentiment with no examples."""
        return self._call(ZERO_SHOT_INSTRUCTIONS, review, demo=DEMO_LABELS[0])

    # --- Few-Shot ---
    def classify_ticket_few_shot(self, ticket: str) -> str:
        """Classify a support ticket using domain-specific few-shot examples."""
        examples = "\n".join(
            f'Ticket: "{text}"\nCategory: {label}' for text, label in FEW_SHOT_EXAMPLES
        )
        instructions = FEW_SHOT_INSTRUCTIONS_TEMPLATE.format(examples=examples)
        return self._call(
            instructions,
            f'Ticket: "{ticket}"\nCategory:',
            demo=DEMO_LABELS[1],
        )

    # --- Chain-of-Thought ---
    def analyze_cot(self, bug_report: str) -> str:
        """Analyze a bug report with step-by-step chain-of-thought reasoning."""
        return self._call(COT_INSTRUCTIONS, bug_report, max_tokens=512, demo=DEMO_LABELS[2])


# ---------------------------------------------------------------------------
# Demo runners
# ---------------------------------------------------------------------------


def _run_zero_shot(console: Console, client: PromptingClient) -> None:
    """Run the zero-shot sentiment analysis demo."""
    console.print("[dim]No examples needed — the model already understands sentiment.[/dim]\n")
    console.print(Panel(ZERO_SHOT_INSTRUCTIONS, title="Instructions", border_style="dim"))

    table = Table(show_lines=True)
    table.add_column("Review", style="cyan", max_width=55)
    table.add_column("Sentiment", style="green", max_width=12)

    for review in REVIEWS:
        try:
            result = client.classify_sentiment(review)
            table.add_row(review, result)
        except Exception as e:
            logger.error("Sentiment classification error: %s", e)
            table.add_row(review, "ERROR")

    console.print(table)


def _run_few_shot(console: Console, client: PromptingClient) -> None:
    """Run the few-shot custom label classification demo."""
    console.print(
        "[dim]The model doesn't know labels like BILLING_DISPUTE — "
        "examples teach your taxonomy.[/dim]\n"
    )
    examples = "\n".join(
        f'Ticket: "{text}"\nCategory: {label}' for text, label in FEW_SHOT_EXAMPLES
    )
    instructions = FEW_SHOT_INSTRUCTIONS_TEMPLATE.format(examples=examples)
    console.print(Panel(instructions, title="Instructions", border_style="dim"))

    table = Table(show_lines=True)
    table.add_column("Support Ticket", style="cyan", max_width=55)
    table.add_column("Category", style="green", max_width=18)

    for ticket in FEW_SHOT_TEST_INPUTS:
        try:
            result = client.classify_ticket_few_shot(ticket)
            table.add_row(ticket, result)
        except Exception as e:
            logger.error("Few-shot classification error: %s", e)
            table.add_row(ticket, "ERROR")

    console.print(table)


def _run_cot(console: Console, client: PromptingClient) -> None:
    """Run the chain-of-thought root cause analysis demo."""
    console.print("[dim]CoT on a bug report that requires multi-step reasoning.[/dim]\n")
    console.print(Panel(COT_INSTRUCTIONS, title="Instructions", border_style="dim"))
    console.print(Panel(BUG_REPORT, title="Bug Report", border_style="dim"))

    try:
        analysis = client.analyze_cot(BUG_REPORT)
        console.print(Panel(analysis, title="Chain-of-Thought Analysis", border_style="green"))
    except Exception as e:
        logger.error("CoT analysis error: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEMOS = {
    DEMO_LABELS[0]: _run_zero_shot,
    DEMO_LABELS[1]: _run_few_shot,
    DEMO_LABELS[2]: _run_cot,
}

_HEADER = Panel(
    "[bold cyan]Few-Shot & Chain-of-Thought Prompting[/bold cyan]\n\n"
    "Three demos, each using the technique where it shines:\n"
    "  A. Zero-shot  — sentiment analysis (task the model already knows)\n"
    "  B. Few-shot   — custom label classification (teaching YOUR taxonomy)\n"
    "  C. Chain-of-thought — root cause analysis (multi-step reasoning)",
    title="Prompt Engineering — Gemini",
    border_style="cyan",
)

_MODEL = "gemini-2.5-flash-lite"


def main() -> None:
    """Run three demos showing when to use each prompting technique."""
    console = Console()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[bold red]GEMINI_API_KEY not found in environment.[/bold red]",
                title="Error",
                border_style="red",
            )
        )
        return

    token_tracker = GeminiTokenTracker(model=_MODEL)
    client = PromptingClient(
        model=_MODEL,
        api_key=api_key,
        token_tracker=token_tracker,
    )

    try:
        while True:
            selection = interactive_menu(
                console,
                DEMO_LABELS,
                title="Select a Demo",
                header=_HEADER,
            )

            if not selection:
                break

            console.print(f"\n[bold yellow]━━━ {selection} ━━━[/bold yellow]")

            try:
                _DEMOS[selection](console, client)
            except Exception as e:
                logger.exception("Demo error")
                console.print(
                    Panel(
                        f"[bold red]{e}[/bold red]",
                        title="Error",
                        border_style="red",
                    )
                )

            token_tracker.flush_csv()
            token_tracker.report(console)
            token_tracker.reset()

            console.print(f"\n[dim]Token usage saved → {_TOKEN_CSV}[/dim]")
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
