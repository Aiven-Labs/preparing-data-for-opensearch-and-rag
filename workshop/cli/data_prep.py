import frontmatter
import os
import pathlib
import uuid

import arrow
from rich.prompt import Confirm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy import helpers

from create_index import client

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")

fmt = r"MMMM[\s+]D[\w+,\s+]YYYY"

# define splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separators=[".", "!", "?", "\n"],
)

# define embeddings. These options are all the defaults and not explicitly needed.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)


def load_data(file: pathlib.Path, index_name: str = INDEX_NAME):
    """Chunk data, create embeddings, and index in OpenSearch."""
    frontmatter_post = frontmatter.loads(
        file.read_text()
    )  # loads the metadata from the file
    base_data = {
        "_index": index_name,
        "title": frontmatter_post["title"],
        "description": frontmatter_post["description"],
        "url": frontmatter_post["url"],
        "pub_date": arrow.get(frontmatter_post["pub_date"], fmt).date().isoformat(),
    }

    docs = []

    post_chunks = splitter.create_documents([frontmatter_post.content])
    for post_chunk in post_chunks:
        doc = {
            **base_data,
            **{
                "_id": str(uuid.uuid4()),
                "content": post_chunk.page_content,
                "content_vector": embeddings.embed_documents([post_chunk.page_content])[
                    0
                ],
            },
        }
        docs.append(doc)
    print(f"{file.name} - len(docs) chunks")
    response = helpers.bulk(client, docs)
    return response


if __name__ == "__main__":
    if Confirm("Do you want to flush the index and generate a new index"):

        for file in pathlib.Path("transcripts_complete").iterdir():
            load_data(file)
