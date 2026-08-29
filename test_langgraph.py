from typing import TypedDict
from langgraph.graph import StateGraph, END


# This defines the "shape" of the shared data that flows through the graph.
# Every node can read from this, and add to it.
class GraphState(TypedDict):
    message: str


# A node is just a normal Python function.
# It receives the current state, and returns a dictionary of updates to it.
def first_node(state: GraphState) -> GraphState:
    print("Running first_node...")
    return {"message": state["message"] + " -> visited first_node"}


def second_node(state: GraphState) -> GraphState:
    print("Running second_node...")
    return {"message": state["message"] + " -> visited second_node"}


# Build the graph
builder = StateGraph(GraphState)

# Register both functions as nodes, giving each a name
builder.add_node("first", first_node)
builder.add_node("second", second_node)

# Define the flow: start at "first", then go to "second", then stop
builder.set_entry_point("first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

# Compile it into something runnable
graph = builder.compile()

# Run it with an initial state
result = graph.invoke({"message": "Start"})
print("\nFinal state:", result)