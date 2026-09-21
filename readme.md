# Prepare your Data for AI Using Aiven for OpenSearch and LangChain

This workshop aims to take unprepared data and make it usable with a Retrieval Augementation Generation (RAG) Pattern for a chat bot.

### Learn more about regularly occuring Aiven workshops

[![Click to Get Started](https://img.shields.io/badge/Learn%20about%20Upcoming%20Workshops!-005eb8?style=for-the-badge)](https://go.aiven.io/github-to-workshops-opensearch-langchain)

In this workshop, we'll be using [Aiven for OpenSearch](https://aiven.io/opensearch) and [LangChain](https://www.langchain.com/) to:

* Chunk transcription data and generate embeddings
* Configure our OpenSearch index for _Known Nearest Neighbors_ (KNN) and perform a similarity search
* Connect our search responses to an Large Language Model (LLM) to generate informed answers using LangChain
* Compare the performance of multiple LLMs

The repo has two parts:

* **[`workshop/`](./workshop/README.md)** — step-by-step Jupyter notebooks that walk through the RAG pattern by hand, from setting up your Aiven for OpenSearch service to chunking data and querying an LLM.
* **[`app/`](./app/main.py)** — a small FastAPI service that packages the same ideas (chunk → embed → index → search) into a running web app, backed by a local OpenSearch and S3-compatible (RustFS) stack via Docker Compose.

## Getting Started with the Workshop

Our instructions and notebooks are in the `workshop` folder.

[![Click to Get Started](https://img.shields.io/badge/Get%20Started!-005eb8?style=for-the-badge)](./workshop/README.md)

## Running the App

`app/` is a FastAPI service with a small web UI for uploading documents, indexing them into OpenSearch with embeddings, and running similarity search — the workshop's concepts, running as a real app.

[![Aiven Runtime: Ready](https://img.shields.io/badge/Aiven%20Runtime-Ready-8A2BE2?style=for-the-badge&labelColor=000000)](https://templates.aiven.io)

This app can also be deployed straight to [Aiven Runtime](https://aiven.io/runtime), which builds it from the included `Dockerfile` and `docker-compose.yml` and wires up its OpenSearch and S3-compatible storage dependencies for you, without having to manage the underlying infrastructure yourself.

### Run it with Docker Compose

This spins up OpenSearch, a local S3-compatible store (RustFS), and the app itself.

```bash
cp .env.example .env
docker compose up --build
```

Once the stack is healthy, open [http://localhost:8080](http://localhost:8080) for the web UI, or use the API directly:

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/upload` | Upload a text/Markdown/PDF file — saves it to S3, chunks it, embeds it, and indexes it into OpenSearch |
| `GET` | `/files` | List documents previously saved to S3 |
| `GET` | `/files/{key}` | Stream a saved document back for viewing/downloading |
| `POST` | `/index-from-s3` | (Re)index a document already saved to S3 |
| `GET` | `/search?query=...` | Run a KNN similarity search against indexed chunks |
| `GET` | `/health` | Health check |

Services and ports:

* **App / UI** — `http://localhost:8080`
* **OpenSearch** — `https://localhost:9200` (admin / `OPENSEARCH_INITIAL_ADMIN_PASSWORD` from `.env`)
* **RustFS console** — `http://localhost:9001`

### Configuration

Copy `.env.example` to `.env` and adjust as needed. By default the app talks to the local OpenSearch and RustFS containers; to point it at a real Aiven for OpenSearch service and AWS S3 bucket instead, set `OPENSEARCH_SERVICE_URI` to your Aiven service URI and provide standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` values (leaving `S3_ENDPOINT_URL` unset so boto3 talks to AWS instead of RustFS).

## License

The text and materials for this workshop are licensed under the Apache license, version 2.0. Full license text is available in the [LICENSE](LICENSE) file.

Please note that the project explicitly does not require a CLA (Contributor License Agreement) from its contributors.

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/kjaymiller/conduit-transcripts">Conduit Podcast Transcripts</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://relay.fm/conduit">Jay Miller, Kathy Campbell, original downloads from whisper work done by Pilix</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution-NonCommercial-ShareAlike 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>

## Contact

Bug reports and patches are very welcome, please post them as GitHub issues and pull requests at https://github.com/Aiven-Labs/preparing-data-for-opensearch-and-rag

To report any possible vulnerabilities or other serious issues please see our [security](SECURITY.md) policy.

Report Code of Conduct issues according to [our policy](https://github.com/Aiven-Labs/.github/blob/main/CODE_OF_CONDUCT.md)
