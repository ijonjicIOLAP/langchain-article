from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv
import os


_ = load_dotenv(find_dotenv())
MEMORY_KEY = os.environ['MEMORY_KEY']

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful assistant for using tools at your disposal.

            When using tool called match_all_query() write the summary about the data and show few citations from that data to support your summary.
            """,
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)