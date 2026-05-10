"""
Structured Output & Prompt Scaffolding (Gemini)

Demonstrates techniques for getting parseable structured output from Gemini:
1. JSON via prompt instructions — asking for JSON in the system prompt
2. Markdown scaffolding — using structured sections to guide output
3. response_schema enforcement — Gemini's native structured output feature

All three methods extract the same product information from one description,
making it easy to compare reliability across techniques.
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from common import interactive_menu, setup_logging

load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Paths — logs/ sits next to this file
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_LOG_DIR = _HERE / "logs"
_LOG_FILE = _LOG_DIR / "structured_output_gemini.log"
_TOKEN_CSV = _LOG_DIR / "token_usage.csv"

_LOG_DIR.mkdir(exist_ok=True)

# File handler — persists all log lines to disk
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
)
logging.getLogger().addHandler(_file_handler)

logger.info("Log file: %s", _LOG_FILE)

# Write CSV header once if the file is new
_CSV_HEADERS = ["timestamp", "method", "model", "prompt_tokens", "completion_tokens", "total_tokens"]
if not _TOKEN_CSV.exists():
    with _TOKEN_CSV.open("w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(_CSV_HEADERS)


def _append_token_row(
    method: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Append one token-usage row to token_usage.csv."""
    with _TOKEN_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            method,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ])


MODEL = "gemini-2.5-flash-lite"

# JSON schema used for prompt-based extraction (methods 1 & 2)
PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Product name",
        },
        "category": {
            "type": "string",
            "description": "Product category",
        },
        "price": {
            "type": "number",
            "description": "Price in USD",
        },
        "features": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "Key product features",
        },
        "in_stock": {
            "type": "boolean",
            "description": "Whether the product is currently available",
        },
    },
    "required": [
        "name",
        "category",
        "price",
        "features",
        "in_stock",
    ],
}



# Single product description — all three methods extract from this same input
PRODUCT_DESCRIPTION = (
    "The UltraSound Pro X1 wireless noise-cancelling headphones deliver studio-quality "
    "audio with 40mm custom drivers and adaptive ANC. Features include 30-hour battery "
    "life, multipoint Bluetooth 5.3 for connecting two devices simultaneously, and a "
    "foldable design with a premium carrying case. Available now at $249.99. "
    "Currently in stock and shipping within 24 hours."
)


# ---------------------------------------------------------------------------
# Pydantic model for method 3 (response_schema enforcement)
# ---------------------------------------------------------------------------


class ProductExtraction(BaseModel):
    """Structured product data extracted from a free-form description."""

    name: str = Field(description="Product name")
    category: str = Field(description="Product category (e.g., Electronics, Clothing)")
    price: float = Field(description="Price in USD")
    features: list[str] = Field(description="Key product features")
    in_stock: bool = Field(description="Whether the product is currently available")


# ---------------------------------------------------------------------------
# Token tracker
# ---------------------------------------------------------------------------


class GeminiTokenTracker:
    """Track token usage across Gemini requests and persist to CSV."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._pending: list[tuple[str, int, int, int]] = []

    def track(self, usage_metadata, method: str = "") -> None:
        """Accumulate counts and queue a CSV row."""
        prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total = getattr(usage_metadata, "total_token_count", 0) or 0
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self._pending.append((method, prompt, completion, total))
        logger.debug(
            "Token usage — method=%s prompt=%d completion=%d total=%d",
            method or "unknown", prompt, completion, total,
        )

    def flush_csv(self) -> None:
        """Write all queued rows to token_usage.csv and clear the queue."""
        for method, prompt, completion, total in self._pending:
            _append_token_row(method, self.model, prompt, completion, total)
        self._pending.clear()

    def report(self, console: Console) -> None:
        """Print a token usage summary table."""
        table = Table(title="Token Usage")
        table.add_column("Type", style="cyan")
        table.add_column("Tokens", style="green", justify="right")
        table.add_row("Prompt", str(self.prompt_tokens))
        table.add_row("Completion", str(self.completion_tokens))
        table.add_row("Total", str(self.total_tokens))
        console.print(table)

    def reset(self) -> None:
        """Reset all counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


# ---------------------------------------------------------------------------
# Structured output client
# ---------------------------------------------------------------------------


class StructuredOutputClient:
    """Demonstrates structured output techniques with Gemini's API."""

    def __init__(self, model: str, api_key: str, token_tracker: GeminiTokenTracker) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.token_tracker = token_tracker

    def _call(
        self,
        instructions: str,
        user_input: str,
        response_mime_type: str = "text/plain",
        response_schema: type[BaseModel] | None = None,
        method: str = "",
    ) -> str:
        """Make an API call, track tokens (with method label), and return text."""
        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
            system_instruction=instructions,
            response_mime_type=response_mime_type,
        )
        if response_schema is not None:
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                system_instruction=instructions,
                response_mime_type="application/json",
                response_schema=response_schema,
            )

        logger.info("API call — model=%s method=%s", self.model, method or "unknown")

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_input,
            config=config,
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.token_tracker.track(response.usage_metadata, method=method)

        return response.text.strip() if hasattr(response, "text") else ""

    def extract_json_prompted(self, description: str) -> str:
        """Extract structured data by asking for JSON in the system prompt."""
        schema_str = json.dumps(PRODUCT_SCHEMA, indent=2)
        instructions = (
            "You are a product data extraction assistant. Extract structured information "
            "from product descriptions.\n\n"
            f"Output ONLY valid JSON matching this schema:\n{schema_str}\n\n"
            "No markdown, no explanation — just the JSON object."
        )
        return self._call(
            instructions,
            description,
            response_mime_type="application/json",
            method=METHOD_LABELS[0],
        )

    def extract_with_scaffolding(self, description: str) -> str:
        """Use markdown sections to scaffold the input and guide the output."""
        schema_str = json.dumps(PRODUCT_SCHEMA, indent=2)
        instructions = (
            "You are a product data extraction assistant. You receive structured inputs "
            "and extract product data as JSON.\n\n"
            "Output ONLY valid JSON matching the provided schema. "
            "No markdown fences, no explanation."
        )
        user_input = (
            f"## Schema\n```json\n{schema_str}\n```\n\n"
            f"## Product Description\n{description}\n\n"
            "## Output\nExtract the product information as JSON:"
        )
        return self._call(
            instructions,
            user_input,
            response_mime_type="application/json",
            method=METHOD_LABELS[1],
        )

    def extract_with_schema(self, description: str) -> str:
        """Use Gemini's native response_schema enforcement — guaranteed valid JSON.

        Gemini accepts the Pydantic class directly — no manual schema conversion needed.
        """
        instructions = (
            "You are a product data extraction assistant. Extract structured information "
            "from product descriptions. Populate all fields based on the description."
        )
        return self._call(
            instructions,
            description,
            response_schema=ProductExtraction,
            method=METHOD_LABELS[2],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_json(raw: str) -> dict | None:
    """Attempt to parse JSON, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed: dict = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s", e)
        return None


def _display_result(console: Console, method_name: str, raw: str) -> None:
    """Parse and display the JSON result from a structured output method."""
    parsed = _try_parse_json(raw)
    if parsed:
        formatted = json.dumps(parsed, indent=2)
        syntax = Syntax(formatted, "json", theme="monokai")
        console.print(Panel(syntax, title=f"{method_name} [green]VALID JSON[/green]"))
    else:
        console.print(Panel(raw[:300], title=f"{method_name} [red]PARSE FAILED[/red]"))


# ---------------------------------------------------------------------------
# Demo runners
# ---------------------------------------------------------------------------

METHOD_LABELS = [
    "1: Prompted JSON",
    "2: Markdown Scaffolding",
    "3: Schema Enforcement (response_schema)",
]


def _run_method_1(console: Console, client: StructuredOutputClient) -> None:
    """Run the prompted JSON extraction method."""
    schema_str = json.dumps(PRODUCT_SCHEMA, indent=2)
    console.print("[dim]Embed the schema in the instructions and ask for JSON output.[/dim]\n")
    prompt_preview = (
        "**Instructions:**\n"
        "```\n"
        "You are a product data extraction assistant...\n"
        f"Output ONLY valid JSON matching this schema:\n{schema_str}\n"
        "No markdown, no explanation — just the JSON object.\n"
        "```\n\n"
        "**Input:** _(raw product description)_\n"
    )
    console.print(Markdown(prompt_preview))

    try:
        raw = client.extract_json_prompted(PRODUCT_DESCRIPTION)
        _display_result(console, "Prompted JSON", raw)
    except Exception as e:
        logger.error("Error in method 1: %s", e)


def _run_method_2(console: Console, client: StructuredOutputClient) -> None:
    """Run the markdown scaffolding extraction method."""
    schema_str = json.dumps(PRODUCT_SCHEMA, indent=2)
    console.print("[dim]Structure the input with markdown sections to guide the output.[/dim]\n")
    prompt_preview = (
        "**Instructions:**\n"
        "```\n"
        "You are a product data extraction assistant.\n"
        "Output ONLY valid JSON matching the provided schema.\n"
        "No markdown fences, no explanation.\n"
        "```\n\n"
        "**Input (markdown-structured):**\n"
        "```markdown\n"
        f"## Schema\n```json\n{schema_str}\n```\n\n"
        "## Product Description\n(product description here)\n\n"
        "## Output\nExtract the product information as JSON:\n"
        "```\n"
    )
    console.print(Markdown(prompt_preview))

    try:
        raw = client.extract_with_scaffolding(PRODUCT_DESCRIPTION)
        _display_result(console, "Markdown Scaffolding", raw)
    except Exception as e:
        logger.error("Error in method 2: %s", e)


def _run_method_3(console: Console, client: StructuredOutputClient) -> None:
    """Run the response_schema enforcement method."""
    console.print(
        "[dim]API-level enforcement via response_schema — guaranteed valid JSON. "
        "Gemini accepts the Pydantic class directly, no conversion needed.[/dim]\n"
    )
    schema_preview = json.dumps(ProductExtraction.model_json_schema(), indent=2)
    prompt_preview = (
        "**Instructions:**\n"
        "```\n"
        "You are a product data extraction assistant...\n"
        "Populate all fields based on the description.\n"
        "```\n\n"
        "**Input:** _(raw product description)_\n\n"
        "**response_schema (Pydantic → Gemini):**\n"
        f"```json\n{schema_preview}\n```\n\n"
        "_The API guarantees the response conforms to this schema — no parsing needed._\n"
    )
    console.print(Markdown(prompt_preview))

    try:
        raw = client.extract_with_schema(PRODUCT_DESCRIPTION)
        _display_result(console, "Schema Enforcement", raw)
    except Exception as e:
        logger.error("Error in method 3: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run one product description through three structured output methods."""
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

    token_tracker = GeminiTokenTracker(model=MODEL)
    client = StructuredOutputClient(MODEL, api_key, token_tracker)

    header = Panel(
        "[bold cyan]Structured Output & Prompt Scaffolding[/bold cyan]\n\n"
        "Comparing 3 techniques for extracting structured JSON from free-form text:\n"
        "  1. JSON via prompt instructions\n"
        "  2. Markdown scaffolding\n"
        "  3. response_schema enforcement (Gemini-native — pass Pydantic directly)\n\n"
        f"[bold]Product Description:[/bold]\n{PRODUCT_DESCRIPTION}",
        title="Prompt Engineering — Gemini",
    )

    methods = {
        METHOD_LABELS[0]: _run_method_1,
        METHOD_LABELS[1]: _run_method_2,
        METHOD_LABELS[2]: _run_method_3,
    }

    try:
        while True:
            selection = interactive_menu(
                console,
                METHOD_LABELS,
                title="Select a Method",
                header=header,
            )
            if not selection:
                break

            console.print(f"\n[bold yellow]━━━ {selection} ━━━[/bold yellow]")

            try:
                methods[selection](console, client)
            except Exception as e:
                logger.error("Method error: %s", e)

            token_tracker.flush_csv()
            token_tracker.report(console)
            token_tracker.reset()

            console.print(f"[dim]Token usage saved → {_TOKEN_CSV}[/dim]")

            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
