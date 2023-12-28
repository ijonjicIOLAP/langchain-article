from agent import create_agent
from langchain.schema.messages import AIMessage, HumanMessage


agent = create_agent()
chat_history = []

result = agent.invoke(
    {
        "input": "Use the tool at your disposal",
        "chat_history": chat_history
    }
)
chat_history.extend(
    [
        HumanMessage(content=result["input"]),
        AIMessage(content=result["output"]),
    ]
)

