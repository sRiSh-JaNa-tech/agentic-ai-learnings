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
from rich.table import Table
from rich.panel import Panel

from common import setup_logging

MODEL = "gemini-2.0-flash"

load_dotenv(find_dotenv())

logger = setup_logging(__name__)
WORKSPACE = Path(__file__).parent.resolve()

_here = Path(__file__).parent
_LOG_DIR = _here / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "gemini_coding_agent.log"
_TOKEN_CSV = _LOG_DIR / "token_usage.csv"

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
)
logging.getLogger().addHandler(_file_handler)

logger.info("Log file: %s", _LOG_FILE)

# Write CSV header once if the file is new
_CSV_HEADERS = ["timestamp", "demo", "model", "prompt_tokens", "completion_tokens", "total_tokens"]
if not _TOKEN_CSV.exists():
    with _TOKEN_CSV.open("w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(_CSV_HEADERS)


def _append_token_row(
    demo : str,
    model : str,
    prompt_tokens : int,
    completion_tokens : int,
    total_tokens : int,
) -> None:
    with _TOKEN_CSV.open("a", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow([
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            demo,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ])

SYSTEM_PROMPT = """You are a coding agent. Use the provided tools to complete tasks.

Guidelines:
- Read files before modifying them
- Make changes incrementally and verify each step
- If a command fails, analyze the error and try a different approach
- When done, provide a brief summary of what you accomplished"""


TOOLS = [
    types.Tool(
        function_declarations = [
            types.FunctionDeclaration(
                name = "read_file",
                description = "Read the contents of a file at the given path.",
                parameters = {
                    "type" : "object",
                    "properties" : {
                        "path" : {
                            "type" : "string",
                            "description" : "The file path to read",
                        },
                        "max_lines" : {
                            "type" : "integer",
                            "description" : "The maximum number of lines to read from the file (optional)",
                        }
                    },
                    "required" : ["path"]
                }
            ),
            types.FunctionDeclaration(
                name = "write_file",
                description = "Write content to a file at the given path.",
                parameters = {
                    "type" : "object",
                    "properties" : {
                        "path" : {
                            "type" : "string",
                            "description" : "The file path to write to",
                        },
                        "content" : {
                            "type" : "string",
                            "description" : "The content to write",
                        },
                        "append" : {
                            "type" : "boolean",
                            "description" : "Whether to append to the file instead of overwriting",
                        }
                    },
                    "required" : ["path", "content","append"]
                }
            ),
            types.FunctionDeclaration(
                name = "run_bash",
                description = "Execute a bash command and return its output.",
                parameters = {
                    "type" : "object",
                    "properties" : {
                        "command" : {
                            "type" : "string",
                            "description" : "The bash command to execute",
                        }
                    },
                    "required" : ["command"]
                }
            )
        ]
    )
]

def read_file(path : str,max_lines : int = None) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.readlines()
        
        total_lines = len(content)
        if max_lines is not None:
            content = content[:max_lines]
        return {"path": path, "content": content, "total_lines": total_lines}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except Exception as e:
        return {"error": str(e)}
    
def write_file(path : str, content : str, append : bool = False) -> dict[str, Any]:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
        return {"path": str(p), "success": True}
    except Exception as e:
        return {"error": str(e)}
    
def run_bash(command: str, timeout: int = 30) -> dict[str, Any]:
    """Execute a bash command and return the output."""
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
    

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
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

class TokenTracker:
    def __init__(self, model: str):
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._turn = 0

    def track(self, usage_metadata) -> None:
        if not usage_metadata:
            return

        # Handle both object and dictionary formats
        if isinstance(usage_metadata, dict):
            prompt = usage_metadata.get("prompt_token_count", 0) or 0
            completion = usage_metadata.get("candidates_token_count", 0) or 0
            total = usage_metadata.get("total_token_count", 0) or 0
        else:
            prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
            completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
            total = getattr(usage_metadata, "total_token_count", 0) or 0

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        
        _append_token_row(
            demo=f"Coding Agent - Turn {self._turn}",
            model=self.model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )
        self._turn += 1

    def report(self, console : Console) -> None:
        console.print("Model - ", self.model)
        table = Table(title="Token Usage", show_header=True, header_style="green")
        table.add_column("Type", style="cyan")
        table.add_column("Prompt Tokens", justify="right", style="magenta")
        table.add_row("Prompt", str(self.prompt_tokens))
        table.add_row("Completion", str(self.completion_tokens))
        table.add_row("Total", str(self.total_tokens))
        console.print(table)
        console.print(f"[dim]Token usage saved -> {_TOKEN_CSV}[/dim]")

class CodingAgent:
    def __init__(
        self,
        api_key : str,
        token_tracker : TokenTracker,
        console : Console,
        model : str = MODEL,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.token_tracker = token_tracker
        self.console = console
        self.max_iterations = 10
        self.history : list[types.Content] = []

    def run(self, user_message : str)->str:
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        for iterations in range(self.max_iterations):
            logger.info(f"--- Iteration {iterations + 1} ---")
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


def main() -> None:
    """Main orchestration function with agent lifecycle + workspace access."""

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

    console.print(
        Panel(
            "[bold cyan]Interactive Coding Agent[/bold cyan]\n\n"
            "[green]Commands:[/green]\n"
            "  • start   -> Start the agent\n"
            "  • stop    -> Stop the agent\n"
            "  • status  -> Show agent status\n"
            "  • files   -> List workspace files\n"
            "  • clear   -> Clear conversation history\n"
            "  • quit    -> Exit program\n\n"
            "[yellow]Workspace:[/yellow]\n"
            f"  {WORKSPACE}\n\n"
            "[bold]Examples:[/bold]\n"
            "  - Create a calculator app\n"
            "  - Explain the codebase\n"
            "  - Create a React component\n"
            "  - Refactor main.py\n",
            title="Coding Agent",
            border_style="blue",
        )
    )

    token_tracker = TokenTracker(model=MODEL)

    agent = CodingAgent(
        api_key=api_key,
        token_tracker=token_tracker,
        console=console,
        model=MODEL,
    )

    # Agent state
    agent_running = False

    try:
        while True:

            status_text = (
                "[green](RUNNING)[/green]"
                if agent_running
                else "[red](STOPPED)[/red]"
            )

            console.print(
                f"\n{status_text} [bold green]You:[/bold green] ",
                end=""
            )

            user_input = input().strip()

            if not user_input:
                continue

            command = user_input.lower()

            # =========================
            # EXIT
            # =========================
            if command in ("quit", "exit", "q"):

                console.print(
                    "\n[yellow]Ending session...[/yellow]"
                )

                break

            # =========================
            # START AGENT
            # =========================
            elif command == "start":

                if agent_running:
                    console.print(
                        "[yellow]Agent already running.[/yellow]"
                    )
                else:
                    agent_running = True
                    console.print(
                        "[green]Agent started.[/green]"
                    )

                continue

            # =========================
            # STOP AGENT
            # =========================
            elif command == "stop":

                if not agent_running:
                    console.print(
                        "[yellow]Agent already stopped.[/yellow]"
                    )
                else:
                    agent_running = False
                    console.print(
                        "[red]Agent stopped.[/red]"
                    )

                continue

            # =========================
            # STATUS
            # =========================
            elif command == "status":

                state = (
                    "[green]RUNNING[/green]"
                    if agent_running
                    else "[red]STOPPED[/red]"
                )

                console.print(
                    Panel(
                        f"Agent Status: {state}\n"
                        f"Workspace: {WORKSPACE}",
                        title="Status",
                    )
                )

                continue

            # =========================
            # LIST FILES
            # =========================
            elif command == "files":

                files = []

                for path in WORKSPACE.rglob("*"):

                    if path.is_file():

                        try:
                            relative = path.relative_to(WORKSPACE)
                            files.append(str(relative))
                        except Exception:
                            pass

                if not files:
                    console.print(
                        "[yellow]No files found.[/yellow]"
                    )
                else:
                    console.print(
                        Panel(
                            "\n".join(files[:200]),
                            title="Workspace Files",
                            border_style="cyan",
                        )
                    )

                continue

            # =========================
            # CLEAR MEMORY / HISTORY
            # =========================
            elif command == "clear":

                if hasattr(agent, "history"):
                    agent.history.clear()

                console.print(
                    "[green]Conversation history cleared.[/green]"
                )

                continue

            # =========================
            # AGENT NOT RUNNING
            # =========================
            if not agent_running:

                console.print(
                    "[red]Agent is stopped.[/red] "
                    "Type [bold]start[/bold] to activate it."
                )

                continue

            # =========================
            # EXECUTE AGENT TASK
            # =========================
            try:

                console.print(
                    "\n[bold blue]Agent Thinking...[/bold blue]"
                )

                response = agent.run(user_input)

                console.print(
                    "\n[bold blue]Agent:[/bold blue]"
                )

                console.print(Markdown(response))

            except Exception as e:

                console.print(
                    Panel(
                        f"[bold red]Error:[/bold red]\n{str(e)}",
                        title="Execution Error",
                        border_style="red",
                    )
                )

    except KeyboardInterrupt:

        console.print(
            "\n[yellow]Interrupted by user.[/yellow]"
        )

    finally:

        console.print()

        if hasattr(agent, "token_tracker"):
            agent.token_tracker.report(console)


if __name__ == "__main__":
    main()