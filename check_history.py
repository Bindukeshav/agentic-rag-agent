import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import graph  # reuses your compiled graph

thread_id = input("Which thread ID do you want to check? ")
config = {"configurable": {"thread_id": thread_id}}

state = graph.get_state(config)
print("\n--- SAVED STATE FOR THIS THREAD ---")
print(f"Last question: {state.values.get('question')}")
print(f"Last answer: {state.values.get('generation')}")