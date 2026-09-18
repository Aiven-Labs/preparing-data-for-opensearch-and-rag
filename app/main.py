"""RAG document uploader service.

Accepts text documents over HTTP, chunks them, generates embeddings, and
indexes them into Aiven for OpenSearch so they can be retrieved for RAG.
"""

import io
import mimetypes
import os
import uuid

import boto3
import frontmatter
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, helpers
from pypdf import PdfReader

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

OPENSEARCH_SERVICE_URI = os.environ["OPENSEARCH_SERVICE_URI"]
INDEX_NAME = os.environ.get("INDEX_NAME", "rag-documents")
OPENSEARCH_VERIFY_CERTS = os.environ.get("OPENSEARCH_VERIFY_CERTS", "true").lower() != "false"

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")

app = FastAPI(title="RAG Document Uploader")
client = OpenSearch(
    OPENSEARCH_SERVICE_URI,
    use_ssl=True,
    verify_certs=OPENSEARCH_VERIFY_CERTS,
    timeout=100,
)

s3_client = (
    boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        config=BotoConfig(s3={"addressing_style": "path"}) if S3_ENDPOINT_URL else None,
    )
    if S3_BUCKET
    else None
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separators=[".", "!", "?", "\n"],
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

INDEX_SETTINGS = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "description": {"type": "text"},
            "url": {"type": "keyword"},
            "content": {"type": "text"},
            "content_vector": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
            },
            "pub_date": {"type": "date"},
            "page": {"type": "integer"},
        }
    },
}


@app.on_event("startup")
def ensure_index() -> None:
    client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS, ignore=400)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _extract_pages(filename: str, raw: bytes) -> tuple[list[tuple[int | None, str]], dict]:
    """Return ([(page_number, text), ...], metadata) for a text or PDF upload.

    page_number is 1-indexed for PDFs (so chunks can link back to a specific
    page) and None for plain text/markdown, which has no page concept.
    """
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        return pages, {}

    post = frontmatter.loads(raw.decode("utf-8"))
    return [(None, post.content)], dict(post.metadata)


def _require_s3() -> None:
    if s3_client is None:
        raise HTTPException(400, "S3_BUCKET is not configured")


def _s3_key(filename: str) -> str:
    return f"{S3_PREFIX}{filename}"


def _index_content(filename: str, raw: bytes, index_name: str) -> dict:
    """Chunk, embed, and index a text or PDF document's content."""
    pages, metadata = _extract_pages(filename, raw)

    if not any(text.strip() for _, text in pages):
        raise HTTPException(400, "Document has no content to index")

    base_data = {
        "_index": index_name,
        "title": metadata.get("title", filename),
        "description": metadata.get("description", ""),
        "url": metadata.get("url", filename),
        "pub_date": metadata.get("pub_date"),
    }

    docs = []
    for page_number, text in pages:
        if not text.strip():
            continue
        for chunk in splitter.create_documents([text]):
            docs.append(
                {
                    **base_data,
                    "_id": str(uuid.uuid4()),
                    "content": chunk.page_content,
                    "content_vector": embeddings.embed_documents([chunk.page_content])[0],
                    "page": page_number,
                }
            )

    indexed, errors = helpers.bulk(client, docs)
    return {"file": filename, "index": index_name, "chunks_indexed": indexed, "errors": errors}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), index_name: str = INDEX_NAME) -> dict:
    """Save a text or PDF document to S3 (if configured), then chunk, embed, and index it."""
    raw = await file.read()

    result = _index_content(file.filename, raw, index_name)

    if s3_client is not None:
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=_s3_key(file.filename),
            Body=raw,
            ContentType=content_type,
        )
        result["s3_key"] = _s3_key(file.filename)

    return result


@app.get("/files")
def list_files() -> dict:
    """List documents previously saved to S3."""
    _require_s3()
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    keys = [obj["Key"] for obj in response.get("Contents", [])]
    return {"bucket": S3_BUCKET, "files": keys}


@app.get("/files/{key:path}")
def get_file(key: str) -> Response:
    """Stream a document previously saved to S3, for viewing/downloading in the browser."""
    _require_s3()
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as exc:
        raise HTTPException(404, f"Could not load '{key}' from S3: {exc}") from exc

    filename = key.removeprefix(S3_PREFIX)
    return Response(
        content=obj["Body"].read(),
        media_type=obj.get("ContentType", "application/octet-stream"),
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/index-from-s3")
def index_from_s3(key: str, index_name: str = INDEX_NAME) -> dict:
    """Load a previously saved document from S3 and (re)index it."""
    _require_s3()
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as exc:
        raise HTTPException(404, f"Could not load '{key}' from S3: {exc}") from exc

    raw = obj["Body"].read()
    filename = key.removeprefix(S3_PREFIX)
    return _index_content(filename, raw, index_name)


@app.get("/search")
def search_documents(query: str, k: int = 5, index_name: str = INDEX_NAME) -> dict:
    """Run a KNN similarity search against indexed document chunks."""
    vector = embeddings.embed_query(query)
    body = {
        "size": k,
        "query": {"knn": {"content_vector": {"vector": vector, "k": k}}},
    }
    response = client.search(index=index_name, body=body)

    results = [
        {
            "score": hit["_score"],
            "title": hit["_source"].get("title"),
            "url": hit["_source"].get("url"),
            "content": hit["_source"].get("content"),
            "page": hit["_source"].get("page"),
        }
        for hit in response["hits"]["hits"]
    ]
    return {"query": query, "results": results}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def spa_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
