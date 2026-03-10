"""
SQL Agent using LangChain + Ollama + SQLite3
=============================================
A conversational SQL agent that understands natural language questions
and queries a SQLite database to return accurate answers.

Requirements:
    pip install langchain langchain-community langchain-ollama
    Ollama must be running locally: https://ollama.com
    Pull a model first: ollama pull llama3.2 (or mistral, codellama, etc.)
"""

import sqlite3
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ollama import ChatOllama


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DB_PATH = "company.db"          # SQLite database file
OLLAMA_MODEL = "llama3.2"       # Change to any model you have pulled
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0                 # 0 = deterministic SQL generation


# ─────────────────────────────────────────────
# Step 1: Create & Seed the SQLite Database
# ─────────────────────────────────────────────

def create_sample_database(db_path: str) -> None:
    """Create a sample company database with realistic data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Departments ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            location    TEXT NOT NULL,
            budget      REAL NOT NULL
        )
    """)

    # --- Employees ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            department_id INTEGER NOT NULL,
            role          TEXT NOT NULL,
            salary        REAL NOT NULL,
            hire_date     TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)

    # --- Projects ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            status        TEXT NOT NULL,   -- active | completed | on_hold
            budget        REAL NOT NULL,
            start_date    TEXT NOT NULL,
            end_date      TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)

    # --- Seed data (only if tables are empty) ---
    cursor.execute("SELECT COUNT(*) FROM departments")
    if cursor.fetchone()[0] == 0:
        departments = [
            ("Engineering",  "New York",    1_500_000),
            ("Marketing",    "Los Angeles", 800_000),
            ("Sales",        "Chicago",     1_200_000),
            ("HR",           "New York",    500_000),
            ("Data Science", "San Francisco", 1_000_000),
        ]
        cursor.executemany(
            "INSERT INTO departments (name, location, budget) VALUES (?, ?, ?)",
            departments,
        )

        employees = [
            ("Alice Johnson",  "alice@co.com",   1, "Senior Engineer",   120_000, "2020-03-15"),
            ("Bob Smith",      "bob@co.com",     1, "Junior Engineer",    75_000, "2022-07-01"),
            ("Carol White",    "carol@co.com",   2, "Marketing Manager", 105_000, "2019-01-20"),
            ("David Brown",    "david@co.com",   3, "Sales Lead",         95_000, "2021-05-10"),
            ("Eva Martinez",   "eva@co.com",     4, "HR Specialist",      70_000, "2023-02-28"),
            ("Frank Lee",      "frank@co.com",   5, "Data Scientist",    130_000, "2020-11-03"),
            ("Grace Kim",      "grace@co.com",   5, "ML Engineer",       140_000, "2021-08-17"),
            ("Henry Clark",    "henry@co.com",   1, "DevOps Engineer",   110_000, "2022-03-22"),
            ("Iris Patel",     "iris@co.com",    2, "Content Strategist", 80_000, "2023-01-09"),
            ("James Wilson",   "james@co.com",   3, "Account Executive",  85_000, "2021-12-01"),
        ]
        cursor.executemany(
            "INSERT INTO employees (name, email, department_id, role, salary, hire_date) VALUES (?, ?, ?, ?, ?, ?)",
            employees,
        )

        projects = [
            ("Platform Rebuild",    1, "active",    500_000, "2024-01-01", None),
            ("Brand Refresh",       2, "completed", 150_000, "2023-06-01", "2023-12-31"),
            ("CRM Integration",     3, "active",    200_000, "2024-03-01", None),
            ("Talent Pipeline",     4, "on_hold",    80_000, "2024-02-01", None),
            ("ML Forecasting",      5, "active",    300_000, "2024-01-15", None),
            ("API Gateway",         1, "completed", 120_000, "2023-09-01", "2024-02-28"),
            ("Social Campaign",     2, "active",     90_000, "2024-04-01", None),
            ("Sales Analytics",     3, "active",    110_000, "2024-02-01", None),
        ]
        cursor.executemany(
            "INSERT INTO projects (name, department_id, status, budget, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?)",
            projects,
        )

    conn.commit()
    conn.close()
    print(f"✅ Database ready: {db_path}\n")


# ─────────────────────────────────────────────
# Step 2: Build the SQL Agent
# ─────────────────────────────────────────────

def build_agent(db_path: str):
    """Construct and return the LangChain SQL agent."""

    # Connect LangChain to the SQLite DB
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=3,   # Include sample rows so the LLM understands data shape
    )

    # Ollama LLM (must be running locally)
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

    # Create the SQL agent (uses ReAct reasoning loop)
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",   # Works well with Ollama chat models
        verbose=True,                # Shows the SQL queries being generated
        max_iterations=10,
        handle_parsing_errors=True,
    )

    return agent


# ─────────────────────────────────────────────
# Step 3: Interactive Chat Loop
# ─────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "How many employees are in each department?",
    "Who are the top 3 highest-paid employees?",
    "What is the total salary budget per department?",
    "List all active projects and their budgets.",
    "Which department has the highest average salary?",
    "How many employees were hired after 2021?",
    "What is the total budget allocated to active projects?",
]

def run_agent():
    """Main entry point — sets up DB, builds agent, starts chat loop."""

    print("=" * 60)
    print("       SQL Agent  |  LangChain + Ollama + SQLite3")
    print("=" * 60)
    print(f"  Model : {OLLAMA_MODEL}")
    print(f"  DB    : {DB_PATH}")
    print("=" * 60)
    print()

    # Create the database
    create_sample_database(DB_PATH)

    # Build the agent
    print("🔧 Loading agent...")
    agent = build_agent(DB_PATH)
    print("✅ Agent ready!\n")

    # Show sample questions
    print("💡 Try asking:")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"   {i}. {q}")
    print()
    print("Type 'quit' or 'exit' to stop.\n")
    print("-" * 60)

    # Conversation loop
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye!")
            break

        print()
        try:
            result = agent.invoke({"input": user_input})
            print(f"\n🤖 Agent: {result['output']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Make sure Ollama is running and the model is pulled.")

        print("-" * 60)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()
