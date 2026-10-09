# Agents/etl_analyst.py

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langchain.tools import tool


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_FORMATS = {
    "csv",
    "json",
    "parquet",
}

DEFAULT_FORMAT = "csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_format(file_format: str) -> str:

    if not file_format:
        return DEFAULT_FORMAT

    file_format = str(
        file_format
    ).lower().strip()

    aliases = {
        "xlsx": "csv",
        "xls": "csv",
        "csv file": "csv",
        "json file": "json",
        "parquet file": "parquet",
    }

    file_format = aliases.get(
        file_format,
        file_format,
    )

    if file_format not in SUPPORTED_FORMATS:

        raise ValueError(
            f"Unsupported format '{file_format}'. "
            f"Supported formats: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return file_format


def get_user_question(
    state: ETLAgentSchema,
) -> str:

    messages = list(
        getattr(
            state,
            "messages",
            [],
        )
    )

    # Find the original/latest real user request.
    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            content = str(
                getattr(
                    message,
                    "content",
                    "",
                )
            )

            if content and not content.startswith(
                "ETL TOOL RESULT"
            ):

                return content

    return ""


def find_latest_output_file(
    output_folder: str,
    file_format: str,
) -> str:

    output_path = Path(
        output_folder
    )

    if not output_path.exists():

        return ""

    extension = f".{file_format.lower()}"

    files = [
        file
        for file in output_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() == extension
    ]

    if not files:
        return ""

    latest_file = max(
        files,
        key=lambda file: file.stat().st_mtime,
    )

    return str(
        latest_file
    )


# ============================================================
# TOOL 1: EXTRACT + LOAD
# ============================================================

@tool
def extract_load_tool(
    url: str,
    output_folder: str,
    format: str = DEFAULT_FORMAT,
) -> str:

    """
    Extract data from an API endpoint and save it
    into the requested output folder.
    """

    try:

        if not url or not url.strip():

            return (
                "ETL EXTRACTION FAILED\n\n"
                "Error: API URL cannot be empty."
            )

        if not output_folder or not output_folder.strip():

            return (
                "ETL EXTRACTION FAILED\n\n"
                "Error: Output folder cannot be empty."
            )

        format = normalize_format(
            format
        )

        Path(
            output_folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        etl_tools = ETLTools()

        result = etl_tools.extract_load(
            url.strip(),
            output_folder.strip(),
            format,
        )

        generated_file = find_latest_output_file(
            output_folder,
            format,
        )

        return (
            "ETL EXTRACTION SUCCESSFUL\n\n"
            f"API URL: {url}\n"
            f"Output Folder: {output_folder}\n"
            f"Format: {format}\n"
            f"Generated File: {generated_file}\n\n"
            f"Result:\n{result}"
        )

    except Exception as exc:

        return (
            "ETL EXTRACTION FAILED\n\n"
            f"API URL: {url}\n"
            f"Output Folder: {output_folder}\n"
            f"Format: {format}\n\n"
            f"Error: {type(exc).__name__}: {exc}"
        )


# ============================================================
# TOOL 2: TRANSFORM + LOAD
# ============================================================

@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str,
    output_format: str,
    user_question: str,
) -> str:

    """
    Transform an existing dataset using Pandas
    and save the transformed result.
    """

    try:

        if not input_file_path.strip():

            return (
                "ETL TRANSFORMATION FAILED\n\n"
                "Error: Input file path cannot be empty."
            )

        if not output_folder.strip():

            return (
                "ETL TRANSFORMATION FAILED\n\n"
                "Error: Output folder cannot be empty."
            )

        if not user_question.strip():

            return (
                "ETL TRANSFORMATION FAILED\n\n"
                "Error: User question cannot be empty."
            )

        output_format = normalize_format(
            output_format
        )

        input_path = Path(
            input_file_path
        )

        if not input_path.exists():

            return (
                "ETL TRANSFORMATION FAILED\n\n"
                "Error: Input file does not exist.\n"
                f"Input File: {input_file_path}"
            )

        Path(
            output_folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        etl_tools = ETLTools()

        # ----------------------------------------------------
        # Get dataset context
        # ----------------------------------------------------

        top_3_rows = (
            etl_tools.transform_load_context(
                input_file_path
            )
        )

        # ----------------------------------------------------
        # HIGH MODEL
        # ----------------------------------------------------

        transformation_llm = pick_llm(
            "medium"
        )

        prompt = f"""
You are an expert Python Data Engineer.

Generate executable Pandas code for an ETL transformation.

USER REQUIREMENT:
{user_question}

INPUT FILE:
{input_file_path}

OUTPUT FOLDER:
{output_folder}

OUTPUT FORMAT:
{output_format}

DATA SAMPLE:
{top_3_rows}

REQUIREMENTS:

1. Use pandas.
2. Read the input file into a DataFrame.
3. Apply the user's requested transformation.
4. Do not invent business requirements.
5. Preserve columns unless the user asks to remove them.
6. Handle null values safely.
7. Create the output directory if necessary.
8. Save the transformed DataFrame.
9. Use the requested output format.
10. Do not call external APIs.
11. Do not install packages.
12. Do not use shell commands.
13. Do not delete unrelated files.
14. Keep the code executable.
15. Do not return markdown.
16. Do not return explanations.
17. Return ONLY Python code.

IMPORTANT:

The following variables already exist:

input_file_path
output_folder
output_format

Do not hardcode their values.

The final DataFrame must be saved inside output_folder.
"""

        response = transformation_llm.invoke(
            prompt
        )

        pandas_code = str(
            response.content
        ).strip()

        # ----------------------------------------------------
        # Remove markdown code fences
        # ----------------------------------------------------

        if pandas_code.startswith(
            "```python"
        ):

            pandas_code = pandas_code[
                len("```python"):
            ].strip()

        elif pandas_code.startswith(
            "```"
        ):

            pandas_code = pandas_code[
                len("```"):
            ].strip()

        if pandas_code.endswith(
            "```"
        ):

            pandas_code = pandas_code[
                :-3
            ].strip()

        # ----------------------------------------------------
        # Execute Pandas transformation
        # ----------------------------------------------------

        results = (
            etl_tools.execute_code(
                pandas_code
            )
        )

        generated_file = find_latest_output_file(
            output_folder,
            output_format,
        )

        return (
            "ETL TRANSFORMATION SUCCESSFUL\n\n"
            f"Input File: {input_file_path}\n"
            f"Output Folder: {output_folder}\n"
            f"Output Format: {output_format}\n"
            f"Generated File: {generated_file}\n\n"
            f"Pandas Code Executed:\n"
            f"{pandas_code}\n\n"
            f"Execution Result:\n"
            f"{results}"
        )

    except Exception as exc:

        return (
            "ETL TRANSFORMATION FAILED\n\n"
            f"Input File: {input_file_path}\n"
            f"Output Folder: {output_folder}\n"
            f"Output Format: {output_format}\n\n"
            f"Error: {type(exc).__name__}: {exc}"
        )


# ============================================================
# TOOL REGISTRY
# ============================================================

tools = [
    extract_load_tool,
    transform_load_tool,
]

tools_by_name = {
    current_tool.name: current_tool
    for current_tool in tools
}


# ============================================================
# MEDIUM MODEL
# ============================================================

llm = pick_llm(
    "medium"
)


# ============================================================
# PLANNER RESPONSE PARSER
# ============================================================

def parse_planner_response(
    response_text: str,
) -> Dict[str, Any]:

    response_text = str(
        response_text
    ).strip()

    # Remove accidental markdown fences.

    if response_text.startswith(
        "```json"
    ):

        response_text = response_text[
            len("```json"):
        ].strip()

    elif response_text.startswith(
        "```"
    ):

        response_text = response_text[
            len("```"):
        ].strip()

    if response_text.endswith(
        "```"
    ):

        response_text = response_text[
            :-3
        ].strip()

    # --------------------------------------------------------
    # Find JSON object if model added extra text
    # --------------------------------------------------------

    if not response_text.startswith("{"):

        start = response_text.find("{")
        end = response_text.rfind("}")

        if start != -1 and end != -1:

            response_text = response_text[
                start:end + 1
            ]

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            response_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Planner returned invalid JSON.\n\n"
            f"Planner response:\n"
            f"{response_text}\n\n"
            f"JSON error: {exc}"
        )

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Planner response must be a JSON object."
        )

    action = result.get(
        "action"
    )

    arguments = result.get(
        "arguments",
        {},
    )

    valid_actions = {
        "extract_load_tool",
        "transform_load_tool",
        "final",
    }

    if action not in valid_actions:

        raise ValueError(
            f"Unsupported planner action: {action}"
        )

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            "Planner arguments must be a JSON object."
        )

    return result


# ============================================================
# LLM NODE
# ============================================================

def llm_node(
    state: ETLAgentSchema,
) -> Dict[str, Any]:

    messages = list(
        getattr(
            state,
            "messages",
            [],
        )
    )

    user_question = get_user_question(
        state
    )

    # --------------------------------------------------------
    # Build history
    # --------------------------------------------------------

    history_parts = []

    for message in messages:

        content = getattr(
            message,
            "content",
            "",
        )

        if not content:
            continue

        message_type = type(
            message
        ).__name__

        history_parts.append(
            f"[{message_type}]\n{content}"
        )

    history_text = "\n\n".join(
        history_parts
    )

    # --------------------------------------------------------
    # Planner prompt
    # --------------------------------------------------------

    prompt = f"""
You are an advanced ETL Data Engineering Agent.

You are the ETL PLANNER.

You must decide what ETL operation should happen next.

IMPORTANT:
Do NOT use function calling.
Do NOT call tools directly.
Return ONLY a valid JSON object.

============================================================
USER REQUEST
============================================================

{user_question}


============================================================
AVAILABLE OPERATIONS
============================================================

1. extract_load_tool

Purpose:
- extract API data
- download API data
- ingest API data
- load API data
- save API response data
- create raw dataset

Arguments:

{{
    "url": "...",
    "output_folder": "...",
    "format": "csv|json|parquet"
}}


2. transform_load_tool

Purpose:
- transform existing file
- filter records
- clean records
- rename columns
- create columns
- remove columns
- aggregate data
- perform Pandas transformation
- convert dataset

Arguments:

{{
    "input_file_path": "...",
    "output_folder": "...",
    "output_format": "csv|json|parquet",
    "user_question": "..."
}}


3. final

Use when the requested ETL operation is complete.

Arguments:

{{
    "message": "..."
}}


============================================================
WORKFLOW
============================================================

API extraction only:

extract_load_tool
→ final


Transformation only:

transform_load_tool
→ final


Extraction + transformation:

extract_load_tool
→ identify generated file
→ transform_load_tool
→ final


============================================================
RULES
============================================================

1. Never invent an API URL.

2. Never invent an input file path.

3. Use the exact URL supplied by the user.

4. Use the exact output folder supplied by the user.

5. CSV means "csv".

6. JSON means "json".

7. Parquet means "parquet".

8. If format is not specified, use "csv".

9. Do not perform unnecessary transformations.

10. If the user requests only extraction, use
    extract_load_tool.

11. If the user requests only transformation, use
    transform_load_tool.

12. If extraction and transformation are requested,
    extraction must happen first.

13. After a tool result, inspect the result.

14. If the tool result contains a generated file path,
    use that exact path for the next transformation.

15. If a tool fails, do not pretend it succeeded.

16. Do not expose internal reasoning.

17. Do not mention model names.

18. Return ONLY valid JSON.

19. Do not use Markdown.

20. Do not wrap JSON in code fences.


============================================================
PREVIOUS CONVERSATION / TOOL RESULTS
============================================================

{history_text}


============================================================
OUTPUT FORMAT
============================================================

For extraction:

{{
    "action": "extract_load_tool",
    "arguments": {{
        "url": "...",
        "output_folder": "...",
        "format": "csv"
    }}
}}

For transformation:

{{
    "action": "transform_load_tool",
    "arguments": {{
        "input_file_path": "...",
        "output_folder": "...",
        "output_format": "csv",
        "user_question": "..."
    }}
}}

For completion:

{{
    "action": "final",
    "arguments": {{
        "message": "..."
    }}
}}
"""

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No bind_tools()
    # No Gemini function calling
    # No ToolMessage
    #
    # This avoids thought_signature errors.
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    response_text = str(
        response.content
    ).strip()

    # Store planner response as a normal AI message.
    ai_message = AIMessage(
        content=response_text
    )

    return {
        "messages": [
            ai_message
        ]
    }


# ============================================================
# TOOL NODE
# ============================================================

def tool_node(
    state: ETLAgentSchema,
) -> Dict[str, Any]:

    messages = list(
        getattr(
            state,
            "messages",
            [],
        )
    )

    if not messages:

        return {}

    last_message = messages[-1]

    planner_response = str(
        getattr(
            last_message,
            "content",
            "",
        )
    )

    # --------------------------------------------------------
    # Parse planner response
    # --------------------------------------------------------

    try:

        plan = parse_planner_response(
            planner_response
        )

    except Exception as exc:

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "ETL TOOL RESULT\n\n"
                        "Planner failed.\n\n"
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            ]
        }

    action = plan.get(
        "action"
    )

    arguments = plan.get(
        "arguments",
        {},
    )

    # --------------------------------------------------------
    # Final action
    # --------------------------------------------------------

    if action == "final":

        message = arguments.get(
            "message",
            "ETL operation completed.",
        )

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "ETL FINAL RESULT\n\n"
                        f"{message}"
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # Validate tool
    # --------------------------------------------------------

    if action not in tools_by_name:

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "ETL TOOL RESULT\n\n"
                        "ERROR: Unknown ETL operation.\n"
                        f"Action: {action}"
                    )
                )
            ]
        }

    selected_tool = tools_by_name[
        action
    ]

    # --------------------------------------------------------
    # Execute tool
    # --------------------------------------------------------

    try:

        observation = selected_tool.invoke(
            arguments
        )

    except Exception as exc:

        observation = (
            "ETL TOOL EXECUTION FAILED\n\n"
            f"Tool: {action}\n"
            f"Error: {type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # Add tool result as normal message.
    #
    # Do NOT use ToolMessage because we are not using
    # model-generated function calls.
    # --------------------------------------------------------

    return {
        "messages": [
            HumanMessage(
                content=(
                    "ETL TOOL RESULT\n\n"
                    f"Tool: {action}\n\n"
                    f"{observation}"
                )
            )
        ]
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def is_tool_call(
    state: ETLAgentSchema,
) -> str:

    messages = list(
        getattr(
            state,
            "messages",
            [],
        )
    )

    if not messages:

        return "end"

    last_message = messages[-1]

    # --------------------------------------------------------
    # Planner messages are AIMessage.
    # --------------------------------------------------------

    if isinstance(
        last_message,
        AIMessage,
    ):

        response_text = str(
            getattr(
                last_message,
                "content",
                "",
            )
        )

        try:

            plan = parse_planner_response(
                response_text
            )

            action = plan.get(
                "action"
            )

            if action in tools_by_name:

                return "tool_node"

            return "end"

        except Exception:

            return "end"

    # --------------------------------------------------------
    # Tool result is a HumanMessage.
    # Send it back to planner.
    # --------------------------------------------------------

    if isinstance(
        last_message,
        HumanMessage,
    ):

        content = str(
            getattr(
                last_message,
                "content",
                "",
            )
        )

        if content.startswith(
            "ETL TOOL RESULT"
        ):

            return "llm_node"

        if content.startswith(
            "ETL FINAL RESULT"
        ):

            return "end"

    return "end"


# ============================================================
# LANGGRAPH
# ============================================================

etl_analyst_graph = StateGraph(
    ETLAgentSchema
)

etl_analyst_graph.add_node(
    "llm_node",
    llm_node,
)

etl_analyst_graph.add_node(
    "tool_node",
    tool_node,
)

etl_analyst_graph.add_edge(
    START,
    "llm_node",
)

etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END,
    },
)

etl_analyst_graph.add_edge(
    "tool_node",
    "llm_node",
)

etl_analyst = (
    etl_analyst_graph.compile()
)


# ============================================================
# GRAPH IMAGE
# ============================================================

def generate_graph_image(
    output_path: str = "etl_analyst_graph.png",
):

    try:

        from IPython.display import Image

        image_data = (
            etl_analyst
            .get_graph()
            .draw_mermaid_png()
        )

        output_file = Path(
            output_path
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "wb",
        ) as file:

            file.write(
                image_data
            )

        print(
            f"Graph image generated: "
            f"{output_file}"
        )

    except Exception as exc:

        print(
            "Graph image generation skipped: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "🤖 ETL DATA AGENT"
    )

    print(
        "=" * 70
    )

    print(
        "\nAvailable operations:"
    )

    print(
        "  1. API extraction"
    )

    print(
        "  2. File transformation"
    )

    print(
        "  3. API extraction + transformation"
    )

    print(
        "\nSupported formats:"
    )

    print(
        "  CSV | JSON | Parquet"
    )

    print(
        "\n"
        + "-" * 70
    )

    # --------------------------------------------------------
    # Generate graph
    # --------------------------------------------------------

    generate_graph_image(
        "etl_analyst_graph.png"
    )

    # --------------------------------------------------------
    # User request
    # --------------------------------------------------------

    user_request = """
I want to extract the data from the API endpoint
https://pokeapi.co/api/v2/pokemon

Save it to data/extract folder as CSV.
"""

    print(
        "USER REQUEST"
    )

    print(
        "-" * 70
    )

    print(
        user_request
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(
                    content=user_request
                )
            ]
        }
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "🤖 ETL AGENT RESULT"
    )

    print(
        "=" * 70
    )

    final_messages = response.get(
        "messages",
        [],
    )

    if final_messages:

        final_message = (
            final_messages[-1]
        )

        final_content = str(
            getattr(
                final_message,
                "content",
                final_message,
            )
        )

        # If planner ended directly with final JSON.
        try:

            parsed = parse_planner_response(
                final_content
            )

            if (
                parsed.get("action")
                == "final"
            ):

                print(
                    parsed
                    .get(
                        "arguments",
                        {},
                    )
                    .get(
                        "message",
                        final_content,
                    )
                )

            else:

                print(
                    final_content
                )

        except Exception:

            print(
                final_content
            )

    print(
        "=" * 70
    )

