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




# file_dir = os.path.dirname(os.path.realpath('__file__'))
# file = os.path.join(file_dir, 'data\Raven.pdf')
# pdf_to_es(file, 'poe')

# print(elastic_client.indices.get_alias(index="*"))


