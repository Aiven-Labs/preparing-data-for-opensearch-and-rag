# Streamlit RAG Application

This Streamlit app provides an interactive interface for the RAG (Retrieval-Augmented Generation) functionality demonstrated in the `3-implement-in-rag.ipynb` notebook.

## Features

- 🔍 **Semantic Search**: Query your OpenSearch index using similarity search
- 🤖 **Multiple LLM Support**: Choose between Ollama (local) or OpenAI (cloud) models
- 📊 **Interactive Results**: View search results with similarity scores
- 📚 **Source Citations**: See which episodes/documents were used to generate responses
- 💬 **Streaming Responses**: Watch AI responses generate in real-time

## Prerequisites

1. **Environment Setup**: Make sure you have your `.env` file configured with:
   - `OPENSEARCH_SERVICE_URI`: Your OpenSearch service URL
   - `INDEX_NAME`: The name of your OpenSearch index
   - `OPENAI_API_KEY`: (Optional) If you want to use OpenAI models

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ollama** (Optional): If you want to use Ollama models locally:
   - Install Ollama from https://ollama.ai
   - Pull a model: `ollama pull llama3`
   - Install the Python package: `pip install langchain-ollama`

## Running the App

From the `workshop` directory, run:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Configure Settings** (Sidebar):
   - Select your preferred LLM model (Ollama or OpenAI)
   - Choose a specific model variant
   - Adjust the number of search results (k)

2. **Enter a Query**:
   - Type your question in the text input field
   - The app will automatically:
     - Search for relevant documents
     - Display source episodes
     - Generate an AI response

3. **View Results**:
   - See the relevant episodes/sources at the top
   - Expand "View Search Results" to see detailed matches with scores
   - Read the AI-generated response below

## Example Queries

- "how do I create healthy boundaries?"
- "how can I get better at saying no?"
- "what are some productivity tips?"

## Troubleshooting

- **"No LLM models available"**: Make sure you have either Ollama installed or an OpenAI API key set
- **"Please set OPENSEARCH_SERVICE_URI"**: Check your `.env` file configuration
- **Ollama connection errors**: Ensure Ollama is running and the model is pulled

