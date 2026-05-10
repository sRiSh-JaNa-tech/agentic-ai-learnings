"""
Augmented LLM — Codebase Navigator (Anthropic)

Demonstrates the "Augmented LLM" pattern: an LLM enhanced with retrieval (RAG),
tools, and memory. This is the foundational building block of all agentic systems,
as described in Anthropic's "Building Effective Agents" guide.

The Codebase Navigator helps engineers explore and understand unfamiliar codebases.
Point it at any GitHub repo, and it will clone, index, and answer questions using
semantic search — while maintaining memory across sessions.
"""

import json
import time
from pathlib import Path
from typing import Any
from google import genai
from google.genai import types
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common import GeminiTokenTracker, setup_logging
from common.menu import interactive_menu

# Load environment variables
load_dotenv(find_dotenv())

logger = setup_logging(__name__)

# Path to the cumulative token usage log
_TOKEN_CSV = Path(__file__).parent.parent / "logs" / "token_usage.csv"

SUGGESTED_REPOS = [
    "openai/swarm",
    "strands-agents/sdk-python",
    "anthropics/anthropic-sdk-python",
    "pallets/flask",
]


SYSTEM_PROMPT = """You are a Codebase Navigator — an AI assistant that helps software engineers \
explore and understand codebases.

## Your Capabilities

You have access to tools for:
- Cloning and indexing GitHub repositories (clone_and_index)
- Listing indexed repositories (list_repos)
- Searching code semantically (search_code)
- Reading full file contents (read_file)
- Exploring directory structures (list_directory)
- Finding exact patterns with regex (grep)
- Saving memories for future sessions (save_memory)
- Recalling saved memories (recall_memory)

## How to Help Users

When a user mentions a GitHub repo (like "pallets/flask" or "look at the httpie repo"):
1. Use clone_and_index to clone and index it first
2. Then answer their questions using search_code, read_file, etc.

Use search_code for semantic/conceptual questions:
- "how does authentication work?"
- "where is the database connection handled?"

Use grep for exact matches:
- "find all TODO comments"
- "where is UserModel defined?"

Use read_file when you need full context after finding relevant chunks.

## Memory

Save important insights to memory, especially:
- Architectural patterns you discover
- Key files and their purposes
- Connections between different repos
- User preferences for how they like information presented

Check recall_memory at the start of conversations to remember context.

## Response Style

- Be concise but thorough
- Show relevant code snippets with file paths and line numbers
- Explain architectural decisions when you discover them
- Suggest related areas to explore"""


# -- Tool registry ----------------------------------------------------------


def _build_tool_definitions() -> list[dict[str, Any]]:
    """Collect all tool definitions from tool modules."""
    from tools.files import FILE_TOOLS
    from tools.memory import MEMORY_TOOLS
    from tools.repo import REPO_TOOLS
    from tools.search import SEARCH_TOOLS

    all_tools = MEMORY_TOOLS + REPO_TOOLS + FILE_TOOLS + SEARCH_TOOLS
    
    # Wrap in Gemini's expected tool format if needed, 
    # but google-genai can take a list of function declarations.
    return [{"function_declarations": all_tools}]


class CodeNavigatorAgent:
    """
    An LLM augmented with retrieval, tools, and memory.

    Implements the agentic loop: send message → execute tools → send results → repeat
    until the LLM responds with just text.
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self.client = genai.Client()
        self.model = model
        self.token_tracker = GeminiTokenTracker(csv_path=_TOKEN_CSV, model=model)
        self.tools = _build_tool_definitions()
        self.messages: list[types.Content] = []
        self.max_iterations = 15

        # Initialize shared components (the three augmentations)
        from indexer.embedder import Embedder
        from store.memory import MemoryStore
        from store.vector import VectorStore

        self.memory = MemoryStore()
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    def _build_system_prompt(self) -> str:
        """Build system prompt with memory context."""
        memory_summary = self.memory.summary()
        if memory_summary and memory_summary != "No memories stored yet.":
            return SYSTEM_PROMPT + f"\n\n## Recalled Memories\n{memory_summary}"
        return SYSTEM_PROMPT

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate handler."""
        from tools.files import execute_list_directory, execute_read_file
        from tools.memory import execute_recall_memory, execute_save_memory
        from tools.repo import execute_clone_and_index, execute_list_repos
        from tools.search import execute_grep, execute_search_code

        dispatch: dict[str, Any] = {
            "save_memory": lambda inp: execute_save_memory(self.memory, inp),
            "recall_memory": lambda inp: execute_recall_memory(self.memory, inp),
            "clone_and_index": lambda inp: execute_clone_and_index(
                self.vector_store, self.embedder, inp
            ),
            "list_repos": lambda inp: execute_list_repos(self.vector_store, inp),
            "read_file": lambda inp: execute_read_file(self.vector_store, inp),
            "list_directory": lambda inp: execute_list_directory(self.vector_store, inp),
            "search_code": lambda inp: execute_search_code(self.vector_store, self.embedder, inp),
            "grep": lambda inp: execute_grep(self.vector_store, self.embedder, inp),
        }

        handler = dispatch.get(name)
        if not handler:
            return f"Unknown tool: {name}"

        try:
            return str(handler(tool_input))
        except Exception as e:
            logger.error("Tool '%s' failed: %s", name, e)
            return f"Error executing {name}: {e}"

    def chat(self, user_message: str, console: Console) -> str:
        """Send a message and handle the agentic tool-use loop."""
        self.messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        for _iteration in range(self.max_iterations):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.messages,
                    config=types.GenerateContentConfig(
                        system_instruction=self._build_system_prompt(),
                        tools=self.tools,
                    ),
                )
            except Exception as e:
                logger.error("Gemini API error: %s", e)
                return f"API error: {e}"

            self.token_tracker.track(response.usage_metadata, demo_name="Codebase Navigator")

            # Add assistant's response to history
            assistant_content = response.candidates[0].content
            self.messages.append(assistant_content)

            # Check for function calls
            function_calls = [p.function_call for p in assistant_content.parts if p.function_call]
            
            if not function_calls:
                # If no function calls, return the text response
                text_parts = [p.text for p in assistant_content.parts if p.text]
                return "\n".join(text_parts) if text_parts else "Done."

            # Execute each tool and print progress
            tool_responses = []
            for fc in function_calls:
                # Print tool invocation for educational transparency
                # fc.args is a dict
                input_summary = json.dumps(fc.args, separators=(",", ":"))
                if len(input_summary) > 80:
                    input_summary = input_summary[:77] + "..."
                console.print(f"  [dim][tool: {fc.name}] {input_summary}[/dim]")

                result = self._execute_tool(fc.name, fc.args)

                # Print brief result
                result_preview = result.split("\n")[0][:80]
                console.print(f"  [dim]  → {result_preview}[/dim]")

                tool_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result}
                        )
                    )
                )

            # Add tool results to history
            self.messages.append(types.Content(role="user", parts=tool_responses))

        return "Reached maximum iterations. Please try a more specific question."


# -- Main -------------------------------------------------------------------


def main() -> None:
    """Main orchestration function."""

    agent = CodeNavigatorAgent()

    console = Console()

    header = Panel(
        "[bold cyan]Codebase Navigator[/bold cyan]\n\n"
        "An LLM enhanced with [green]Retrieval (RAG)[/green], "
        "[yellow]Tools[/yellow], and [magenta]Memory[/magenta].\n\n"
        "Select a repo to index, then ask questions about the codebase.",
        title="Codebase Navigator",
    )

    repo = interactive_menu(
        console,
        SUGGESTED_REPOS,
        title="Select a Repo to Explore",
        header=header,
        allow_custom=True,
        custom_prompt="Enter owner/repo (e.g., pallets/flask)",
    )
    if not repo:
        return

    console.print(f"\n[bold green]Indexing:[/bold green] {repo}")
    response = agent.chat(f"index the repo {repo}", console)
    if response:
        console.print("\n[bold blue]Navigator:[/bold blue]")
        console.print(Markdown(response))

    console.print("\n[dim]Ask questions about the codebase. Type 'quit' to exit.[/dim]")

    try:
        while True:
            console.print("\n[bold green]You:[/bold green] ", end="")
            try:
                user_input = input().strip()
            except EOFError:
                break

            if user_input.lower() in ("exit", "quit", "q", ""):
                console.print("\n[yellow]Ending session...[/yellow]")
                break

            response = agent.chat(user_input, console)

            if response:
                console.print("\n[bold blue]Navigator:[/bold blue]")
                console.print(Markdown(response))

            agent.token_tracker.report()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")

    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
