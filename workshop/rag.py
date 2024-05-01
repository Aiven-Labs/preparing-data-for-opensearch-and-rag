import os
from dotenv import load_dotenv

from rich.console import Console
from opensearchpy import OpenSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community

load_dotenv()

CONNECTION_STRING  = os.getenv("OPENSEARCH_SERVICE_URI")
INDEX_NAME ="embedded_transcripts"
client = OpenSearch(CONNECTION_STRING, use_ssl=True, timeout=100)
client.info()