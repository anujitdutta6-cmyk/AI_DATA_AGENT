# Run data_agent.py , optional use main.py
from Agents.data_agent import data_agent
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    response = data_agent.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)