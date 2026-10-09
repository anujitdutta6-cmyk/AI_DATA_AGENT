# Agents/data_agent.py

import os
import sys
from typing import Literal

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.llm_pick import pick_llm

from Models.schema import (
    RouterSchema,
    DataAgentSchema,
)

from Agents.etl_analyst import etl_analyst
from Agents.sql_analyst import sql_analyst

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)


# =============================================================
# DATA AGENT CONFIGURATION
# =============================================================

DATA_AGENT_NAME = "Data Engineering Agent"

SUPPORTED_ROUTES = {
    "sql",
    "etl",
}


# =============================================================
# ROUTER LLM
# =============================================================

llm = pick_llm("medium")


# Explicit structured output.
#
# This is preferable to relying on implicit behaviour.
# If your Gemini OpenAI-compatible endpoint does not support
# JSON schema mode, replace this with a plain JSON planner.
# =============================================================

try:

    llm_router = llm.with_structured_output(
        RouterSchema,
        method="json_schema",
    )

except TypeError:

    # Compatibility fallback for older LangChain versions.

    llm_router = llm.with_structured_output(
        RouterSchema
    )


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def get_user_message(state: DataAgentSchema) -> str:
    """
    Return the latest user message from DataAgent state.
    """

    messages = getattr(
        state,
        "messages",
        [],
    )

    for message in reversed(messages):

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
            ).strip()

            if content:
                return content

    raise ValueError(
        "No user message was found in DataAgent state."
    )


def normalize_route(route: str) -> str:
    """
    Normalize and validate router output.
    """

    route = str(
        route
    ).strip().lower()

    if route not in SUPPORTED_ROUTES:

        raise ValueError(
            f"Invalid route returned by router: {route}. "
            f"Expected one of: {sorted(SUPPORTED_ROUTES)}"
        )

    return route


def extract_sql_result(response: dict) -> str:
    """
    Extract the final answer from SQL Analyst state.
    """

    if not isinstance(
        response,
        dict,
    ):

        return str(response)

    final_answer = response.get(
        "final_answer"
    )

    if final_answer:

        return str(
            final_answer
        )

    messages = response.get(
        "messages",
        [],
    )

    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            None,
        )

        if content:

            return str(
                content
            )

    return (
        "The SQL operation completed, "
        "but no final response was returned."
    )


def extract_etl_result(response: dict) -> str:
    """
    Extract the final answer/result from ETL Analyst state.
    """

    if not isinstance(
        response,
        dict,
    ):

        return str(response)

    messages = response.get(
        "messages",
        [],
    )

    # Search backwards for the final ETL result.
    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            continue

        content = str(
            content
        ).strip()

        if content.startswith(
            "ETL FINAL RESULT"
        ):

            return content.replace(
                "ETL FINAL RESULT",
                "",
                1,
            ).strip()

    # Fallback to latest message.
    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            None,
        )

        if content:

            return str(
                content
            )

    return (
        "The ETL operation completed, "
        "but no final response was returned."
    )


# =============================================================
# 1. ROUTER NODE
# =============================================================

def router_node(
    state: DataAgentSchema,
) -> dict:
    """
    Classify the user request as SQL or ETL.
    """

    user_message = get_user_message(
        state
    )

    router_prompt = f"""
You are the routing controller for a Data Engineering Agent.

Classify the user's request into exactly one category:

- sql
- etl

==================================================
SQL
==================================================

Choose "sql" when the user wants to:

- query a database
- retrieve database records
- filter database data
- aggregate database data
- inspect database information
- generate SQL
- execute a read-only SQL query
- analyze relational database tables

Examples:

"Show me the different payment methods."

"How many customers are there?"

"Find the top 10 products by revenue."

==================================================
ETL
==================================================

Choose "etl" when the user wants to:

- extract data from an API
- download data
- read files
- transform files
- clean datasets
- convert CSV/JSON/Parquet
- load data
- perform ETL/ELT
- create an ETL pipeline
- manipulate datasets

Examples:

"Extract data from this API and save it as CSV."

"Convert this JSON file to Parquet."

"Remove null records from this CSV."

"Load this API data into a file."

==================================================
IMPORTANT RULES
==================================================

1. Return exactly one route.
2. Do not answer the user's request.
3. Do not execute anything.
4. Do not invent information.
5. If the request is clearly about an API/file/data pipeline,
   choose "etl".
6. If the request is clearly about database querying,
   choose "sql".
7. If the request contains both SQL and ETL concepts, choose
   the operation that represents the primary requested action.

USER REQUEST:

<USER_REQUEST>
{user_message}
</USER_REQUEST>
"""

    try:

        result = llm_router.invoke(
            router_prompt
        )

        route_response = normalize_route(
            result.answer
        )

        comments = str(
            result.comments
        ).strip()

        return {
            "route_response": route_response,
            "messages": [
                AIMessage(
                    content=(
                        f"Request routed to "
                        f"{route_response.upper()} analyst."
                    )
                )
            ],
        }

    except Exception as exc:

        raise RuntimeError(
            "Data Agent router failed.\n"
            f"Error: {type(exc).__name__}: {exc}"
        ) from exc


# =============================================================
# 2. ETL NODE
# =============================================================

def etl_node(
    state: DataAgentSchema,
) -> dict:
    """
    Execute the ETL Analyst subgraph.
    """

    user_message = get_user_message(
        state
    )

    try:

        etl_input = {
            "messages": [
                HumanMessage(
                    content=user_message
                )
            ]
        }

        response = etl_analyst.invoke(
            etl_input
        )

        final_answer = extract_etl_result(
            response
        )

        return {
            "messages": [
                AIMessage(
                    content=final_answer
                )
            ]
        }

    except Exception as exc:

        return {
            "messages": [
                AIMessage(
                    content=(
                        "ETL operation failed.\n\n"
                        f"Error: {type(exc).__name__}: {exc}"
                    )
                )
            ]
        }


# =============================================================
# 3. SQL NODE
# =============================================================

def sql_node(
    state: DataAgentSchema,
) -> dict:
    """
    Execute the SQL Analyst subgraph.
    """

    user_message = get_user_message(
        state
    )

    sql_input = {
        "messages": [],

        "user_question": user_message,

        "curated_ques": "",

        "prompt_query_context": "",

        "generated_sql_query": "",

        "is_safe": "No",

        "comments": "",

        "sql_query_execution_result": "",

        "final_answer": "",
    }

    try:

        response = sql_analyst.invoke(
            sql_input
        )

        final_answer = extract_sql_result(
            response
        )

        return {
            "messages": [
                AIMessage(
                    content=final_answer
                )
            ]
        }

    except Exception as exc:

        return {
            "messages": [
                AIMessage(
                    content=(
                        "SQL operation failed.\n\n"
                        f"Error: {type(exc).__name__}: {exc}"
                    )
                )
            ]
        }


# =============================================================
# 4. ROUTING EDGE
# =============================================================

def route_edge(
    state: DataAgentSchema,
) -> Literal["sql_node", "etl_node"]:
    """
    Decide which specialist agent should execute.
    """

    route = normalize_route(
        state.route_response
    )

    if route == "sql":
        return "sql_node"

    return "etl_node"


# =============================================================
# 5. BUILD DATA AGENT GRAPH
# =============================================================

data_agent_graph = StateGraph(
    DataAgentSchema
)


# -------------------------------------------------------------
# Nodes
# -------------------------------------------------------------

data_agent_graph.add_node(
    "router_node",
    router_node,
)

data_agent_graph.add_node(
    "sql_node",
    sql_node,
)

data_agent_graph.add_node(
    "etl_node",
    etl_node,
)


# -------------------------------------------------------------
# Entry
# -------------------------------------------------------------

data_agent_graph.add_edge(
    START,
    "router_node",
)


# -------------------------------------------------------------
# Router
# -------------------------------------------------------------

data_agent_graph.add_conditional_edges(
    "router_node",
    route_edge,
    {
        "sql_node": "sql_node",
        "etl_node": "etl_node",
    },
)


# -------------------------------------------------------------
# Finish
# -------------------------------------------------------------

data_agent_graph.add_edge(
    "sql_node",
    END,
)

data_agent_graph.add_edge(
    "etl_node",
    END,
)


# -------------------------------------------------------------
# Compile
# -------------------------------------------------------------

data_agent = (
    data_agent_graph.compile()
)


# =============================================================
# GRAPH VISUALIZATION
# =============================================================

def generate_graph_image(
    output_path: str = "data_agent_graph.png",
):

    try:

        from IPython.display import Image

        image_data = (
            data_agent
            .get_graph()
            .draw_mermaid_png()
        )

        with open(
            output_path,
            "wb",
        ) as file:

            file.write(
                image_data
            )

        print(
            f"Graph image generated: {output_path}"
        )

    except Exception as exc:

        print(
            "Graph image generation skipped: "
            f"{type(exc).__name__}: {exc}"
        )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print()
    print("=" * 72)
    print("🤖 DATA ENGINEERING AGENT")
    print("=" * 72)

    print()
    print("Available specialist agents:")
    print("  • SQL Analyst")
    print("  • ETL Analyst")

    print()
    print("-" * 72)

    generate_graph_image(
        "data_agent_graph.png"
    )

    # ---------------------------------------------------------
    # User request
    # ---------------------------------------------------------

    user_request = (
        "I want to extract the data from the API endpoint "
        "'https://pokeapi.co/api/v2/pokemon' "
        "and save it to the data/extract folder as CSV."
    )

    print()
    print("USER REQUEST")
    print("-" * 72)
    print(user_request)

    print()
    print("PROCESSING...")
    print("-" * 72)

    # ---------------------------------------------------------
    # Invoke Data Agent
    # ---------------------------------------------------------

    try:

        response = data_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_request
                    )
                ],
                "route_response": "",
            }
        )

        # -----------------------------------------------------
        # Route
        # -----------------------------------------------------

        route = response.get(
            "route_response",
            "",
        )

        print()
        print(
            f"ROUTED TO: {route.upper()}"
        )

        # -----------------------------------------------------
        # Final response
        # -----------------------------------------------------

        messages = response.get(
            "messages",
            [],
        )

        print()
        print("=" * 72)
        print("FINAL RESPONSE")
        print("=" * 72)

        if messages:

            final_message = messages[-1]

            final_content = getattr(
                final_message,
                "content",
                str(final_message),
            )

            print(
                final_content
            )

        else:

            print(
                "No final response was returned."
            )

    except Exception as exc:

        print()
        print("=" * 72)
        print("❌ DATA AGENT ERROR")
        print("=" * 72)

        print(
            f"{type(exc).__name__}: {exc}"
        )

    print()
    print("=" * 72)

    