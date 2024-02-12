Chat-gpt locally using python and Elasticsearch

The purpose of this article is to introduce one of the most popular tools in the world of programming today - large language model (LLM). 
The term "black box" is often used when talking about artificial intelligence, and it is defined as: "a complex system or device whose internal workings are hidden or not readily understood." 
Artificial intelligence is truly complex, whether we are talking about models in the world of computer vision or models responsible for natural language processing (NLP). I will try to bring this complexity closer to you through a practical example in which we will create a basic architecture for using NLP models on text summarization.
In this way, without going into the black box and settings of models, like chat-gpt-turbo, we will gain insight into their capabilities and limitations through the preparation and application.

The architecture we will create will be the following:

 
Picture1

In steps it looks like this:
1.	Create Elasticsearch locally
2.	Upload documents to Elasticsearch
3.	Create an LLM agent using the Langchain library in python
4.	Conversation with agent



Before we start, I would like to clarify a few more things about our architecture.
1.	What is Elasticsearch (ES) and why do we use it for the database?
Briefly, Elasticsearch is the distributed search and analytics engine at the heart of the Elastic Stack.
For more information visit https://www.elastic.co/ 
I chose ES because it's used in projects, it's easy to set up with Docker, and it integrates well with Langchain.

2.	If you are wondering what Langchain is; that's another thing I'd like to clear up before we get into the code.
Briefly, LangChain is a framework for developing applications powered by language models.
For more information visit https://python.langchain.com/docs/get_started 
With the help of Langchain, we will initialize our artificial intelligence model and prepare it for custom use.

One last thing before we start.
We'll be using Docker to run ES. If you haven't used Docker before, don't worry, we'll simplify things so that everything will be set up with 2 commands in cmd. Of course, you will need to install Docker if you don't have it, and you can do that here.
https://docs.docker.com/desktop/install/windows-install/

So, let’s start.









Step 1. Setup Elasticsearch with the help of Docker
1.	install Docker Desktop and run it
2.	open cmd and run: git clone https://github.com/ijonjicIOLAP/langchain-article.git
3.	navigate to folder langhchain-article that you cloned and run docker compose up 
4.	in your web browser go to localhost:9200 and check if running 
(user: elastic, pass: changeme) 
You should see something like this.

 Picture2
That’s it, our base is ready.
To set up for the next steps you should create a new folder with python virtual environment that is running python 3.10, 4 empty .py files, .env file and folder for our data. I included all my files inside repo in folder code so you can use Raven.pdf to follow along. You can name files as you would like but I suggest naming them like I did. You can see the names and folder structure in this picture.
  Picture3

Activate your environment and run 
pip install langchain elasticsearch python-dotenv openai pypdf tiktoken

Next, we will look at the content of .env file. 
MEMORY_KEY="chat_history"
OPENAI_API_KEY=”PUT_YOUR_API_KEY”
ELASTIC_HOST="http://elastic:changeme@localhost:9200"

As you can see in .env file we will need OPENAI_API_KEY, so now is a good time to sign up to https://openai.com/ . You can use your google acc and then create your own API key. Other variables should be the same as mine. 







Step 2. Upload documents to ES
To begin with, in utils.py, we need to import a few things and load env variables. 
# utils.py
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from elasticsearch import Elasticsearch
from langchain.agents import tool
import os
from dotenv import load_dotenv, find_dotenv
import openai

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
ELASTIC_HOST = os.environ['ELASTIC_HOST']
elastic_client = Elasticsearch(hosts=[ELASTIC_HOST])

Then we create 2 functions, one to connect and upload to ES and another to load pdf, split it, embed it and upload it to ES using the first function.
#utils.py
def upload_to_es(docs, embeddings, index_name):
    db = ElasticsearchStore.from_documents(
        docs,
        embeddings,
        es_url=ELASTIC_HOST,
        index_name=index_name,
    )

    db.client.indices.refresh(index=index_name)

def pdf_to_es(file: str, index_name: str):
    loader = PyPDFLoader(file)
    data = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()

    upload_to_es(docs=docs, embeddings=embeddings, index_name=index_name)

Loading and splitting are familiar to you, but embedding might not be. Simply put, to embed some text means to translate them to numbers, or more precisely vectors. We are using OpenAI embeddings but there are many more that you can use.
I decided to upload the poem Raven by Edgar Allan Poe that I put in my data folder.
# utils.py
file_dir = os.path.dirname(os.path.realpath('__file__'))
file = os.path.join(file_dir, 'data\Raven.pdf')
pdf_to_es(file, 'poe')

After calling the function pdf_to_es() we check that everything went well by printing indices, and if we see that index 'poe' is mentioned, we are sure that the data has been loaded. 
# utils.py
print(elastic_client.indices.get_alias(index="*"))

Just a note, make sure to comment out these 4 lines after you run them.
Of course, you can upload any other file by using other loaders, play with the chunk_size and chunk_overlap parameter when splitting the text, experiment with embeddings depending on your needs. Just make sure to use the same embeddings for embedding documents and for your LLM model.

Step 3. Create Agent
In this case, we broke the agent into several parts.
We will use utils.py to write the functions that our agent will use.
The match_all_query() function is the simplest search we can do on ES, it simply returns all the data we imported under index “poe”. We will later pass that function to the agent as a tool that he will be able to use. Using the decorator @tool we can transform any python function into a tool for our agent, and one agent can use many tools. We are using a function without any arguments to make things simple but, we could use the function with arguments and even let the agent decide from user input what arguments to use. Here we could really personalize agent for our needs, e.g. create tools for searching and manipulating contents of a given CSV, use Google or Wikipedia API to conduct searches and fetch page summaries of top-k results, run shell commands, run some basic or advanced math. If we want, we could prompt our agent to use one function after another and perform some complex tasks. Sky is the limit here but let’s get back to earth and our basic function. The important thing here, or any other function, is to write DOC string that explains the function and arguments.
# utils.py
@tool 
def match_all_query():
    """
    Query's elasticsreach document and returns result based on match_all query
    """
    results = elastic_client.search(index="poe", body={"query": {"match_all": {}}}, size=999)
    data = []
    i = 0
    while i < results['hits']['total']['value']:
        for result in results['hits']['hits'][i]['_source']['text']:
            data.append(result)
        i+=1
    return ''.join(data)

In prompt.py will write the prompt that is needed to explain to the agent its purpose and tasks, and how it can solve them. At this point we are customizing our agent for our needs and tools. There are few ways to write a prompt that you can find in Langchain docs but basically, we are setting up our agent to do specific tasks. You can look at it like fine-tuning a model for us, using full power of LLM like chat-gpt-3-turbo to summarize Edgar Allan Poe poems maybe isn’t the most optimal use for it but it could easily be used on some science papers, financial and other reports and more. The important thing here is the text in triple quotes, explanation should be clear and precise to avoid mistakes and hallucinations. Again, for this example prompt is short and simple but you can imagine that it can get much longer when we try to utilize many tools and solve complex tasks.  
# prompt.py
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

Agent.py is the file in which we will combine what we have done so far in utils.py and prompt.py into one object that will become our chat agent. Function create_agent() is doing exactly that, for LLM we are using OpenAI model gpt-3.5-turbo but you could experiment with many other models that you can find in Langchain docs, we are also passing the tools to the agent in form of a list, then combining the tools and model, putting it all together in our agent and returning AgentExecutor that is the runtime for an agent. This is what actually calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats. 
# agent.py
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from prompt import prompt
from utils import match_all_query

def create_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [ match_all_query]

    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


Step 4. Let’s talk 
Conversation.py is the final step where we can talk to our agent and test it. For my example I can just write “Use the tool at your disposal” as input because I explained in the prompt what is the tool and how to use it, and I get a short summary of a poem and few citations just to see if the answer is really coming from my data and not from general knowledge that comes with LLM-s. 
# conversation.py
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

And here is the answer.  
Picture4
In the blue we can see the data returned by the tool and in the green we can see a summary of that data.

Conclusion
I highly recommend reading the Langchain documentation for more information about many functions and objects that we used here, and I didn’t explain, but that wasn’t the purpose of this article. If you followed the steps or just copied the code from git hub repo you have a nice starting point to play and learn about LLM-s. Next you can try and change the model that was used here, try some other embeddings or try the same thing on other documents. Good luck, have fun. 





 


