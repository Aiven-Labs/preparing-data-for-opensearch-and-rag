import os

from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

connection_string = os.getenv("OPENSEARCH_SERVICE_URI")
client = OpenSearch(connection_string, use_ssl=True, timeout=100)
client.info()

index_settings = {
    "settings": {
        "index": {"knn": True},
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "description": {"type": "text"},
            "url": {"type": "keyword"},
            "content": {"type": "text"},
            "content_vector": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "faiss",
                },
            },
            "pub_date": {"type": "date"},
        }
    },
}

os_index = os.getenv("INDEX_NAME")

if __name__ == "__main__":
    client.indices.create(index=index_name, body=index_settings, ignore=400)
