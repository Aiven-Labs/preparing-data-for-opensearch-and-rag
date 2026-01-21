import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Try to import Ollama, but make it optional
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Vector Podcast Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vector_search" not in st.session_state:
    st.session_state.vector_search = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


def get_base_episode_url(metadata: dict) -> str:
    """Get the base episode URL (without timestamp) for uniqueness.
    
    Returns the original url field, or youtube_url if available, or youtube_video_id as fallback.
    """
    # Prefer original url field
    base_url = metadata.get('url')
    if base_url and base_url != 'N/A':
        return base_url
    
    # Fall back to youtube_url if available
    youtube_url = metadata.get('youtube_url')
    if youtube_url:
        return youtube_url
    
    # Last resort: construct from youtube_video_id
    youtube_video_id = metadata.get('youtube_video_id')
    if youtube_video_id:
        return f"https://youtu.be/{youtube_video_id}"
    
    return 'N/A'


def construct_episode_url(metadata: dict) -> str:
    """Construct episode URL with timestamp support.
    
    If timestamp and youtube_video_id exist, constructs a timestamped YouTube URL.
    Otherwise, falls back to the original url field.
    """
    timestamp = metadata.get('timestamp')
    youtube_video_id = metadata.get('youtube_video_id')
    
    # Check if both timestamp and youtube_video_id exist and are not None
    if timestamp is not None and youtube_video_id:
        try:
            # Ensure timestamp is an integer
            timestamp_int = int(timestamp)
            return f"https://youtu.be/{youtube_video_id}?t={timestamp_int}"
        except (ValueError, TypeError):
            # If timestamp conversion fails, fall through to original URL
            pass
    
    # Fall back to base episode URL
    return get_base_episode_url(metadata)


@st.cache_resource
def initialize_vector_search():
    """Initialize the OpenSearch vector search connection"""
    opensearch_url = os.getenv("OPENSEARCH_SERVICE_URI")
    base_index_name = os.getenv("INDEX_NAME")
    
    # Use INDEX_NAME_WITH_TIMESTAMPS if set, otherwise default to {INDEX_NAME}_timestamps
    index_name_with_timestamps = os.getenv("INDEX_NAME_WITH_TIMESTAMPS")
    if index_name_with_timestamps:
        index_name = index_name_with_timestamps
    elif base_index_name:
        index_name = f"{base_index_name}_timestamps"
    else:
        index_name = None
    
    if not opensearch_url or not index_name:
        st.error("Please set OPENSEARCH_SERVICE_URI and INDEX_NAME in your .env file")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings()
    vector_search = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings,
        opensearch_url=opensearch_url,
    )
    return vector_search, embeddings


def perform_search(query: str, k: int = 6):
    """Perform similarity search"""
    if st.session_state.vector_search is None:
        st.session_state.vector_search, st.session_state.embeddings = initialize_vector_search()
    
    results = st.session_state.vector_search.similarity_search_with_score(
        query,
        vector_field="content_vector",
        text_field="content",
        metadata_field="*",
        k=k,
    )
    return results


def get_episodes_from_results(results):
    """Extract unique episodes from search results with their metadata"""
    episodes_dict = {}
    for result in results:
        if isinstance(result, tuple):
            doc = result[0]
        else:
            doc = result
        title = doc.metadata.get('title', 'Unknown')
        # Use base URL (without timestamp) for uniqueness and display in episode list
        base_url = get_base_episode_url(doc.metadata)
        description = doc.metadata.get('description', '')
        # Try multiple possible field names for image URL
        img_url = (
            doc.metadata.get('img_url') or 
            doc.metadata.get('image_url') or 
            doc.metadata.get('image') or 
            doc.metadata.get('episode_image') or
            None
        )
        # Use base URL for episode key to ensure uniqueness across chunks
        episode_key = f"{title} - {base_url}"
        # Store the first occurrence of each episode with its img_url
        # Use base URL for episode list (represents the whole episode, not a specific chunk)
        if episode_key not in episodes_dict:
            episodes_dict[episode_key] = {
                'title': title,
                'url': base_url,  # Use base URL for episode list (unique episodes)
                'description': description,
                'img_url': img_url
            }
    return episodes_dict


def generate_response(query: str, docs: str, model_type: str, model_name: Optional[str] = None):
    """Generate response using the selected LLM"""
    if model_type == "OpenAI":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please set OPENAI_API_KEY in your .env file")
            return None
        llm = ChatOpenAI(model=model_name or "gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful assistant answering questions about a podcast using the provided documents.
            
            Use the following documents to answer the question. Each document has a Title, URL, and Content.
            
            Instructions:
            - Answer the question using information from the provided documents
            - Include supporting quotes from the documents, wrapped in quotation marks
            - Mention which episode/document you're quoting from
            - If there are no relevant documents, say "I don't have any information on that."
            - Don't include quotes from other sources not in the provided documents
            - Keep responses under 1000 characters but use multiple paragraphs for readability
            """),
            ("user", "Question: {query}\n\nDocuments:\n{docs}"),
        ])
    elif model_type == "Ollama":
        if not OLLAMA_AVAILABLE:
            st.error("Ollama is not available. Please install langchain-ollama")
            return None
        llm = ChatOllama(model=model_name or "llama3")
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions about a podcast using the provided documents.

Question: {query}

Documents:
{docs}

Instructions:
- Answer the question using information from the provided documents above
- Each document has a "Title:" field, a "URL:" field, and a "Content:" field
- Include supporting quotes from the documents
- When citing sources, format citations as clickable markdown links: [Title](URL)
- Use the title from the "Title:" field and the URL from the "URL:" field of the document you're quoting from
- Example: "As mentioned in [Episode Title](https://youtu.be/VIDEO_ID?t=123)..."
- If there are no relevant documents, say "I don't have any information on that."
- Don't include quotes from sources not in the provided documents
- Keep responses around 800 characters
            """
        )
    else:
        st.error(f"Unknown model type: {model_type}")
        return None
    
    chain = prompt | llm | StrOutputParser()
    topic = {"query": query, "docs": docs}
    return chain, topic


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_options = []
    if OLLAMA_AVAILABLE:
        model_options.append("Ollama")
    if os.getenv("OPENAI_API_KEY"):
        model_options.append("OpenAI")
    
    if not model_options:
        st.error("No LLM models available. Please configure Ollama or OpenAI API key.")
        st.stop()
    
    model_type = st.selectbox("Select LLM Model", model_options)
    
    # Model-specific settings
    ollama_model = None
    openai_model = None
    if model_type == "Ollama":
        ollama_model = st.selectbox(
            "Ollama Model",
            ["llama3", "llama3.2", "llama3.1", "mistral", "neural-chat"],
            index=0
        )
    elif model_type == "OpenAI":
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
    
    if not OLLAMA_AVAILABLE and model_type == "Ollama":
        st.warning("⚠️ Ollama not available. Install with: `pip install langchain-ollama`")
    
    # Search parameters
    st.header("🔍 Search Parameters")
    k_results = st.slider("Number of results (k)", min_value=1, max_value=20, value=6)
    
    st.header("ℹ️ About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) application:
    - Searches your OpenSearch index using semantic similarity
    - Generates responses using LLMs (Ollama or OpenAI)
    - Provides source citations from retrieved documents
    """)

# Main content
st.title("🔍 RAG Search Application")
st.markdown("Ask questions and get AI-powered answers based on your indexed documents.")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., how do I create healthy boundaries?",
    key="query_input"
)

if query:
    # Perform search
    with st.spinner("Searching for relevant documents..."):
        results = perform_search(query, k=k_results)
    
    if results:
        # Display episodes/sources with images
        st.subheader("📚 Relevant Episodes/Sources")
        episodes = get_episodes_from_results(results)
        
        # Display episode images with descriptions
        episode_list = list(episodes.values())
        if episode_list:
            for episode in episode_list:
                # Create a container for each episode
                with st.container():
                    # Create columns: image on left, description on right
                    img_col, desc_col = st.columns([1, 3])
                    
                    with img_col:
                        # Display smaller image (150px width)
                        if episode['img_url']:
                            try:
                                st.image(
                                    episode['img_url'], 
                                    width=150
                                )
                            except Exception as e:
                                # Fallback: Try using HTML/markdown
                                try:
                                    st.markdown(
                                        f'<img src="{episode["img_url"]}" alt="{episode["title"]}" style="width:150px; border-radius:8px;">',
                                        unsafe_allow_html=True
                                    )
                                except Exception as e2:
                                    st.warning(f"Could not load image")
                                    st.info(f"📄\n{episode['title']}")
                        else:
                            # Placeholder if no image
                            st.info(f"📄\n{episode['title']}")
                    
                    with desc_col:
                        # Display episode title (clickable if URL is available)
                        if episode['url'] != 'N/A':
                            st.markdown(f"**[{episode['title']}]({episode['url']})**")
                        else:
                            st.markdown(f"**{episode['title']}**")
                        
                        # Display description with ellipsis if too long
                        max_desc_length = 250
                        description = episode.get('description', '')
                        if description:
                            if len(description) > max_desc_length:
                                truncated_desc = description[:max_desc_length].rsplit(' ', 1)[0] + '...'
                                st.markdown(f"*{truncated_desc}*")
                            else:
                                st.markdown(f"*{description}*")
                        else:
                            st.markdown("*No description available*")
                        
                        # Display episode link
                        if episode['url'] != 'N/A':
                            st.markdown(f"[🔗 View Episode]({episode['url']})")
                    
                    st.markdown("---")
        
        st.divider()
        
        # Generate response - right after episodes
        st.subheader("💬 AI Response")
        
        # Prepare documents for LLM
        formatted_docs = []
        for result in results:
            doc = result[0] if isinstance(result, tuple) else result
            title = doc.metadata.get('title', 'Unknown')
            # Use helper function to construct URL with timestamp support
            url = construct_episode_url(doc.metadata)
            content = doc.page_content
            formatted_docs.append(f"Title: {title}\nURL: {url}\nContent: {content}")
        docs_text = "\n\n".join(formatted_docs)
        
        # Get model name
        model_name = None
        if model_type == "Ollama":
            model_name = ollama_model
        elif model_type == "OpenAI":
            model_name = openai_model
        
        # Generate and stream response
        response_container = st.empty()
        full_response = ""
        
        try:
            chain, topic = generate_response(query, docs_text, model_type, model_name)
            if chain:
                status_text = st.empty()
                status_text.text("Generating response...")
                for chunk in chain.stream(topic):
                    full_response += chunk
                    response_container.markdown(full_response)
                status_text.empty()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.exception(e)
        
        st.divider()
        
        # Debug and additional information at the bottom
        # Debug: Show available metadata fields
        with st.expander("🔍 Debug: View Metadata Fields", expanded=False):
            if results:
                sample_doc = results[0][0] if isinstance(results[0], tuple) else results[0]
                st.json(sample_doc.metadata)
        
        # Also show a simple list view
        with st.expander("📋 View as List", expanded=False):
            for episode_key in sorted(episodes.keys()):
                st.markdown(f"- {episode_key}")
        
        # Display search results with scores
        with st.expander("View Search Results (with similarity scores)", expanded=False):
            for i, (doc, score) in enumerate(results, 1):
                st.markdown(f"**Result {i}** (Score: {score:.4f})")
                st.text_area(
                    f"Content {i}",
                    value=doc.page_content,
                    height=100,
                    key=f"result_{i}",
                    label_visibility="collapsed"
                )
                st.markdown(f"*Metadata: {doc.metadata}*")
                st.markdown("---")
    else:
        st.warning("No results found for your query.")

else:
    st.info("👆 Enter a question above to start searching and generating responses.")

