"""Command-line entry point for the AI Data Agent."""

from Agents.data_agent import data_agent
from langchain_core.messages import HumanMessage


def main() -> None:
    user_request = (
        "Extract data from the API endpoint "
        "'https://pokeapi.co/api/v2/pokemon' and save it "
        "to the data/extract folder as CSV."
    )
    response = data_agent.invoke(
        {
            "messages": [HumanMessage(content=user_request)],
            "route_response": "",
        }
    )
    print(response)


if __name__ == "__main__":
    main()
