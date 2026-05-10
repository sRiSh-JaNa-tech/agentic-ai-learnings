"""
Tool Use (Gemini)

Demonstrates how to enable the model to call functions and use tools.
Uses practical tools: calculator, file reader, and bash command execution.

Gemini-specific notes:
- Tools are declared via types.Tool + types.FunctionDeclaration
- The function parameter schemas are identical to the OpenAI version
- Multi-turn tool loops use a contents list of Content objects
- Tool results are returned as function_response parts
"""

import csv
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from common import setup_logging

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Paths — logs/ sits next to this file
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_LOG_DIR = _HERE / "logs"
_LOG_FILE = _LOG_DIR / "tool_use_gemini.log"
_TOKEN_CSV = _LOG_DIR / "token_usage.csv"

_LOG_DIR.mkdir(exist_ok=True)

logger = setup_logging(__name__)

# File handler — persists all log lines to disk
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
)
logging.getLogger().addHandler(_file_handler)

logger.info("Log file: %s", _LOG_FILE)

# Write CSV header once if the file is new
_CSV_HEADERS = ["timestamp", "turn", "model", "prompt_tokens", "completion_tokens", "total_tokens"]
if not _TOKEN_CSV.exists():
    with _TOKEN_CSV.open("w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(_CSV_HEADERS)


def _append_token_row(
    turn: int,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Append one token-usage row to token_usage.csv."""
    with _TOKEN_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            turn,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ])


MODEL = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Tool schemas — identical parameter definitions to the OpenAI version
# ---------------------------------------------------------------------------

TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description="Performs basic arithmetic operations. Supports addition, subtraction, multiplication, and division.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform",
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["operation", "a", "b"],
            },
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Reads the contents of a file at the specified path. Returns the file content as text.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 100)",
                    },
                },
                "required": ["path"],
            },
        ),
        types.FunctionDeclaration(
            name="run_bash",
            description="Executes a bash command and returns the output. Use for system commands like ls, pwd, echo, date, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                    },
                },
                "required": ["command"],
            },
        ),
    ])
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """Execute calculator tool."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }

    result = operations[operation](a, b)
    logger.info("Calculator: %s %s %s = %s", a, operation, b, result)

    return {"result": result, "operation": operation, "operands": [a, b]}


def read_file(path: str, max_lines: int = 100) -> dict[str, Any]:
    """Read the contents of a file."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        content = "".join(lines[:max_lines])
        truncated = total_lines > max_lines

        logger.info("Read file: %s (%d lines)", path, total_lines)

        return {
            "path": path,
            "content": content,
            "total_lines": total_lines,
            "truncated": truncated,
        }
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except Exception as e:
        return {"error": str(e)}


BLOCKED_COMMANDS = ["rm", "sudo", "chmod", "chown", "mkfs", "dd", "shutdown", "reboot", ">", ">>"]


def run_bash(command: str, timeout: int = 30) -> dict[str, Any]:
    """Execute a bash command and return the output."""
    cmd_lower = command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            logger.warning("Blocked dangerous command: %s", command)
            return {"error": f"Command blocked for safety: contains '{blocked}'"}

    logger.info("Running bash command: %s", command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}


# Tool execution mapping
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "read_file": read_file,
    "run_bash": run_bash,
}


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Execute a tool and return its result."""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        func = TOOL_FUNCTIONS[tool_name]
        return func(**tool_input)  # type: ignore[operator]
    except Exception as e:
        logger.error("Tool execution error (%s): %s", tool_name, e)
        return {"error": str(e)}


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
        self._turn = 0

    def track(self, usage_metadata) -> None:
        """Accumulate counts, log debug info, and write a CSV row."""
        self._turn += 1
        prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total = getattr(usage_metadata, "total_token_count", 0) or 0

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

        logger.debug(
            "Token usage — turn=%d prompt=%d completion=%d total=%d",
            self._turn, prompt, completion, total,
        )
        _append_token_row(self._turn, self.model, prompt, completion, total)

    def report(self, console: Console) -> None:
        """Print a cumulative token usage summary table."""
        table = Table(title="Token Usage Summary")
        table.add_column("Type", style="cyan")
        table.add_column("Tokens", style="green", justify="right")
        table.add_row("Prompt", str(self.prompt_tokens))
        table.add_row("Completion", str(self.completion_tokens))
        table.add_row("Total", str(self.total_tokens))
        console.print(table)
        console.print(f"[dim]Token usage saved → {_TOKEN_CSV}[/dim]")


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------


class ToolUseChat:
    """Chat session with tool-use capabilities powered by Gemini."""

    def __init__(
        self,
        model: str,
        api_key: str,
        token_tracker: GeminiTokenTracker,
        console: Console,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.token_tracker = token_tracker
        self.console = console
        # Gemini multi-turn history: list of types.Content objects
        self.history: list[types.Content] = []

    def send_message(self, user_message: str) -> str:
        """Send a message and handle the tool-use loop until a text reply arrives."""
        # Append the user turn to history
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        while True:
            logger.info("API call — model=%s history_turns=%d", self.model, len(self.history))

            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    tools=TOOLS,
                    temperature=0.0,
                    max_output_tokens=4096,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,  # we handle the tool loop manually
                    ),
                ),
            )

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.token_tracker.track(response.usage_metadata)

            candidate = response.candidates[0]
            content = candidate.content
            parts = (content.parts if content and content.parts else [])

            # Check for function calls in the response parts
            function_calls = [p for p in parts if p.function_call is not None]

            if function_calls:
                # Append the model's reply (which contains the function calls) to history
                self.history.append(
                    types.Content(role="model", parts=parts)
                )

                self.console.print("\n[yellow]-> Executing tools...[/yellow]")

                # Execute each tool and collect function_response parts
                response_parts = []
                for part in function_calls:
                    fc = part.function_call
                    tool_args = dict(fc.args)

                    self.console.print(
                        f"  [dim]* {fc.name}({json.dumps(tool_args, indent=2)})[/dim]"
                    )

                    result = execute_tool(fc.name, tool_args)
                    logger.info("Tool result (%s): %s", fc.name, result)

                    response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                            )
                        )
                    )

                # Append the tool results as a user turn
                self.history.append(
                    types.Content(role="user", parts=response_parts)
                )

                # Loop back to get the model's next reply
                continue

            else:
                # No function calls — extract and return the final text reply.
                # Prefer parts text; fall back to response.text if parts is empty/None.
                if parts:
                    text_parts = [p.text for p in parts if p.text]
                    final_text = "\n".join(text_parts).strip()
                else:
                    final_text = (response.text or "").strip()

                # Append the final model reply to history
                self.history.append(
                    types.Content(role="model", parts=parts or [types.Part(text=final_text)])
                )

                return final_text

    def get_turn_count(self) -> int:
        """Return the number of user/model turn pairs exchanged."""
        return len(self.history) // 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main orchestration function — interactive tool-use chat with Gemini."""
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
    chat = ToolUseChat(MODEL, api_key, token_tracker, console)

    console.print(
        Panel(
            "[bold cyan]Agent with Tools — Gemini[/bold cyan]\n\n"
            "Available tools:\n"
            "* Calculator (add, subtract, multiply, divide)\n"
            "* Read file (read contents of any file)\n"
            "* Run bash (execute shell commands)\n\n"
            "Try: 'What's 123 * 456?' or 'List files in the current directory'\n"
            "Or: 'Read the pyproject.toml file'\n\n"
            "Type 'quit' to exit.",
            title="Tool Use Demo — Gemini",
        )
    )

    try:
        while True:
            console.print("\n[bold green]You:[/bold green] ", end="")
            user_input = input().strip()

            if user_input.lower() in ["quit", "exit", ""]:
                console.print("\n[yellow]Ending chat session...[/yellow]")
                break

            try:
                reply = chat.send_message(user_input)
                if reply:
                    console.print("\n[bold blue]Agent:[/bold blue]")
                    console.print(Markdown(reply))

            except Exception as e:
                logger.error("Error during chat: %s", e)
                console.print(f"\n[red]Error: {e}[/red]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Ending chat session...[/yellow]")

    console.print()
    token_tracker.report(console)
    console.print(f"\n[dim]Total conversation turns: {chat.get_turn_count()}[/dim]")


if __name__ == "__main__":
    main()
