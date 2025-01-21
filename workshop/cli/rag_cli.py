import os
import pathlib

import typer
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from data_prep import load_data

app = typer.Typer()

load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME")


@app.command()
def upload(file: pathlib.Path, index_name: str = INDEX_NAME):
    """Chunks and uploads a document to the index"""
    typer.echo(f"Uploading {file} to index: {index_name}")
    load_data(
        file,
        index_name=index_name,
    )
    typer.echo("Done!")


@app.command()
def search(query: str):
    """Perform a RAG search using llama3.3"""

    typer.echo("Preparing Embeddings")
    embeddings = HuggingFaceEmbeddings()
    typer.echo("DONE")
    vector_search = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        opensearch_url=os.getenv("OPENSEARCH_SERVICE_URI"),
    )

    results = vector_search.similarity_search(
        query,
        vector_field="content_vector",
        text_field="content",
        metadata_field="*",
    )

    episodes = set()
    for result in results:
        episodes.add(f"{result.metadata["title"]} - {result.metadata["url"]}")
    print(f'Here are some episodes that might help you with "{query}":')
    print("\n".join(episodes))
    print("-------------------")

    llm = ChatOllama(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(
        """
            Offer supportive advice for the question {query} with supporting quotes from 
            "{docs}".

            If there are no documents to quote, say "I don't have any information on that."

            Mention the quote you're pulling from                                                                     
            Don't include quotes from other sources
            make responses about 600 characters
        """
    )

    chain = prompt | llm | StrOutputParser()
    topic = {
        "query": query,
        "docs": "\n".join([result.page_content for result in results]),
    }
    for chunks in chain.stream(topic):
        print(chunks, end="", flush=True)


if __name__ == "__main__":
    app()
