import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from utils.llm_pick import pick_llm
from Models.schema import AgentSchema,JugdgeSchema
from langchain_core.messages import HumanMessage

llm = pick_llm("medium")
llm_judge = llm.with_structured_output(JugdgeSchema)

sql_query = "DELETE * FROM users WHERE age > 30;"

# prompt = f"""
# you are an SQL Judge for data security. Your task is to determine whether the SQL query is safe or not. The SQL Query should only be used for data retrieval and should not modify the database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the  database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with 'No'.Additionally, provide comments explaining your decision. 
# Here's the SQL query to evaluted:
# {sql_query}"""

# response = llm_judge.invoke(prompt).model_dump()
# print(response)
