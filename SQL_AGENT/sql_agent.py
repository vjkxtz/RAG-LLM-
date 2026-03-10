import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import pathlib
from langchain.agents import create_agent



template = """Question: {question}

Answer: Let's think step by step."""


prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(
    model="phi3:instruct",
    temperature=0.7,
    num_predict=256,
)

llm = ChatOllama(
    model="phi3:instruct",
   
)

#chain = prompt | model

# chain.invoke({"question": "What is LangChain?"})

def main():
    # engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={0}".format(params), connect_args={'attrs_before': attrs_before})
    # db = SQLDatabase(engine)
    
    client = sqlite3.connect("database/Body1_Masterdata_2022-2026.db") 
    cursor = client.cursor()

    local_path = pathlib.Path("database/Body1_Masterdata_2022-2026.db")
    print(local_path)
    #sql_query = """SELECT name FROM sqlite_master  WHERE type='table';"""
    #cursor.execute(sql_query)
    #print(cursor.fetchall())

    db = SQLDatabase.from_uri(f"sqlite:///{local_path}")
    #print(f"Dialect: {db.dialect}")
    #print(f"Available tables: {db.get_usable_table_names()}")
    #print(f'Sample output: {db.run("SELECT * FROM workout LIMIT 5;")}')

    system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()
    # result = tools[0].func("select * FROM station LIMIT 5;")
    # print(result)
    for tool in tools:
        print(f"{tool.name}: {tool.description}\n")
    

    # create agent will not work with ollama models if open ai then can use it

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )

    question = "Which is the longest problem?"

    # for step in agent.stream(
    #     {"messages": [{"role": "user", "content": question}]},
    #     stream_mode="values",
    # ):
    #     step["messages"][-1].pretty_print()

    for step in agent.stream({"messages": [{"role": "user", "content": question}]}):
        print(step, end="")


if __name__ == "__main__":
    main() 
