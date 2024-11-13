from components.assets import arrow_circle_icon, github_icon
from components.chat import chat, chat_form, chat_message
import asyncio
import modal
from fasthtml.common import *
import fastapi
import logging
from transformers import AutoTokenizer
import uuid
from modal import Secret  # Import Secret
from fastlite import Database  # For database operations
from starlette.middleware.sessions import SessionMiddleware  # For session handling
import aiohttp  # For asynchronous HTTP requests
import os
import sqlite3

# Constants
MODELS_DIR = "/llama_mini"
MODEL_NAME = "Llama-3.2-3B-Instruct"  
FAISS_DATA_DIR = "/faiss_data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
USERNAME = "c123ian"
APP_NAME = "rag-chatbot"
DATABASE_DIR = "/db_data"  # Database directory

db_path = os.path.join(DATABASE_DIR, 'chat_history.db')

# Ensure the directory exists
os.makedirs(DATABASE_DIR, exist_ok=True)

# Create the 'conversations' table if it does not exist
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    DROP TABLE IF EXISTS conversations
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations_history_table_sqlalchemy (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()

# Step 2: Initialize FastLite Database connection
db = Database(db_path)
conversations = db['conversations']  # Access the existing table

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Download the model weights
try:
    volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",
    "python-fasthtml==0.4.3",
    "aiohttp",          # For asynchronous HTTP requests
    "faiss-cpu",        # For FAISS
    "sentence-transformers",
    "pandas",
    "numpy",
    "huggingface_hub",
    "transformers",
    'sqlalchemy'     
)

# Define the FAISS volume
try:
    faiss_volume = modal.Volume.lookup("faiss_data", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Create the FAISS data volume first with download_faiss.py")

# Define the database volume
try:
    db_volume = modal.Volume.lookup("db_data", create_if_missing=True)
except modal.exception.NotFoundError:
    db_volume = modal.Volume.persisted("db_data")


# Define the Modal app
app = modal.App(APP_NAME)

# vLLM server implementation with model path handling
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve_vllm():
    # ... (vLLM server code remains unchanged)
    pass  # For brevity, assuming this part remains the same as your code


# FastHTML web interface implementation with RAG
@app.function(
    image=image,
    volumes={FAISS_DATA_DIR: faiss_volume, DATABASE_DIR: db_volume},
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    import faiss
    import os
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    from starlette.middleware.sessions import SessionMiddleware
    from fastapi.middleware import Middleware
    from starlette.websockets import WebSocket
    import uuid
    import asyncio
    from sqlalchemy import create_engine, Column, String, DateTime
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    import datetime


    # Retrieve the secret key from environment variables
    SECRET_KEY = os.environ.get('YOUR_KEY')
    if not SECRET_KEY:
        raise Exception("YOUR_KEY environment variable not set.")

    # Paths to FAISS index and documents
    FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
    DATA_PICKLE_PATH = os.path.join(FAISS_DATA_DIR, "data.pkl")

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Load documents (DataFrame)
    df = pd.read_pickle(DATA_PICKLE_PATH)
    # Ensure 'combined_text' column exists
    if 'combined_text' not in df.columns:
        df['combined_text'] = df['Question']  # Use 'Question' column for embeddings

    docs = df['Answer'].tolist()  # Use corresponding 'Answer' column as the context

    # Load embedding model
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize FastHTML app with session middleware
    fasthtml_app, rt = fast_app(
        hdrs=(
            Script(src="https://cdn.tailwindcss.com"),
            Link(
                rel="stylesheet",
                href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
            ),
        ),
        ws_hdr=True,
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=SECRET_KEY,
                session_cookie="secure_session",
                max_age=86400,  # 24 hours
                same_site="strict",
                https_only=True
            )
        ]
    )

    # Add a dictionary to store session-specific messages
    session_messages = {}


    # Define the SQLAlchemy base and model
    Base = declarative_base()

    class Conversation(Base):
        __tablename__ = 'conversations_history_table_sqlalchemy'
        message_id = Column(String, primary_key=True)
        session_id = Column(String, nullable=False)
        role = Column(String, nullable=False)
        content = Column(String, nullable=False)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Create a SQLAlchemy engine and session
    engine = create_engine(f'sqlite:///{db_path}')
    Session = sessionmaker(bind=engine)
    sqlalchemy_session = Session()

    # Update the load_chat_history function
    async def load_chat_history(session_id):
        """Load chat history for a session from the database"""
        if session_id not in session_messages:
            # Query the database for this session's messages using SQLAlchemy
            session_history = sqlalchemy_session.query(Conversation).filter_by(session_id=session_id).order_by(Conversation.created_at).all()
            
            # Initialize the session's message list
            session_messages[session_id] = [
                {"role": msg.role, "content": msg.content}
                for msg in session_history
            ]

        return session_messages[session_id]

    @rt("/")
    async def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

        # Load chat history when rendering the page
        messages = await load_chat_history(session_id)

        return Div(
            H1(
                "Chat with Agony Aunt",
                cls="text-3xl font-bold mb-4 text-white"
            ),
            chat(session_id=session_id, messages=messages),  # Pass messages to chat component
            # Model status indicator
            Div(
                Span("Model status: "),
                Span("⚫", id="model-status-emoji"),
                cls="model-status text-white mt-4"
            ),
            Div(id="top-sources"),  # Placeholder for Top Sources
            cls="flex flex-col items-center min-h-screen bg-black",
        )

    # Placeholder implementation for arrow_circle_icon
    # (Assuming arrow_circle_icon is imported correctly)

    def chat_top_sources(top_sources):
        return Div(
            Div(
                Div("Top Sources", cls="text-zinc-400 text-sm"),
                Div(
                    *[
                        A(
                            Span(
                                source['data_headline'][:31],
                                cls=(
                                    "font-mono text-green-500 group-hover:"
                                    "text-green-400 text-sm"
                                ),
                            ),
                            Div(
                                arrow_circle_icon(),
                                cls=(
                                    "flex items-center justify-center text-green-500"
                                    " group-hover:text-green-400 size-5 -rotate-45"
                                ),
                            ),
                            href=source['url'],
                            target="_blank",
                            cls=(
                                "justify-between items-center pl-2 pr-1 flex border"
                                " border-green-500 w-48 rounded-md group"
                            ),
                        )
                        for source in top_sources
                    ],
                    cls="flex items-center justify-start gap-2",
                ),
                cls="flex flex-col items-center gap-1",
            ),
            cls="flex flex-col items-center gap-1",
        )

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, session, send):
        if session is None:
            session = {}

        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

        messages = await load_chat_history(session_id)

        response_received = asyncio.Event()  # Event to indicate if response has been received

        # Increase max_tokens as per your request
        max_tokens = 6000

        async def update_model_status():
            # Wait for 3 seconds
            await asyncio.sleep(3)
            if not response_received.is_set():
                # Start switching between ⚫ and 🟡 every second
                for _ in range(25):  # 25 times * 2 seconds = 50 seconds
                    if response_received.is_set():
                        break
                    # Toggle to 🟡
                    await send(
                        Span(
                            "🟡",
                            id="model-status-emoji",
                            hx_swap_oob="innerHTML"
                        )
                    )
                    await asyncio.sleep(1)
                    if response_received.is_set():
                        break
                    # Toggle to ⚫
                    await send(
                        Span(
                            "⚫",
                            id="model-status-emoji",
                            hx_swap_oob="innerHTML"
                        )
                    )
                    await asyncio.sleep(1)
                else:
                    # After 50 seconds, set status to 🔴 if no response
                    if not response_received.is_set():
                        await send(
                            Span(
                                "🔴",
                                id="model-status-emoji",
                                hx_swap_oob="innerHTML"
                            )
                        )
            if response_received.is_set():
                # Set status to 🟢
                await send(
                    Span(
                        "🟢",
                        id="model-status-emoji",
                        hx_swap_oob="innerHTML"
                    )
                )
                # Wait for 10 minutes, then set back to ⚫
                await asyncio.sleep(600)
                await send(
                    Span(
                        "⚫",
                        id="model-status-emoji",
                        hx_swap_oob="innerHTML"
                    )
                )

        # Start the update_model_status coroutine
        asyncio.create_task(update_model_status())

        # Add user's message to session-specific chat history
        messages.append({"role": "user", "content": msg})
        message_index = len(messages) - 1

        # Store in database
        new_message = Conversation(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role='user',
            content=msg
        )
        sqlalchemy_session.add(new_message)
        sqlalchemy_session.commit()

        await send(chat_form(disabled=True))
        await send(
            Div(
                chat_message(message_index, messages=messages),  # Pass messages
                id="messages",
                hx_swap_oob="beforeend"
            )
        )

        # Compute embedding of user's message
        question_embedding = emb_model.encode([msg], normalize_embeddings=True)
        question_embedding = question_embedding.astype('float32')

        # Retrieve top K similar documents
        K = 3
        distances, indices = index.search(question_embedding, K)
        retrieved_docs = [docs[idx] for idx in indices[0]]

        # Extract 'data-headline' and 'URL' of the top documents
        top_sources = []
        for idx in indices[0][:2]:  # Top 2 documents
            data_headline = df.iloc[idx]['data-headline']
            url = df.iloc[idx]['URL']  # Ensure 'URL' is a column in your DataFrame
            top_sources.append({'data_headline': data_headline, 'url': url})

        # Construct context from retrieved documents
        context = "\n\n".join(retrieved_docs)

        # Build conversation history
        def build_conversation(messages, max_length=2000):
            conversation = ''
            total_length = 0
            # Start from the latest messages
            for message in reversed(messages):
                role = message['role']
                content = message['content']
                message_text = f"{role.capitalize()}: {content}\n"
                total_length += len(message_text)
                if total_length > max_length:
                    break
                conversation = message_text + conversation
            return conversation

        conversation_history = build_conversation(messages)

        # System prompt
        system_prompt = (
            "You are an 'Agony Aunt' who helps individuals clarify their options and think through their choices. "
            "Provide thoughtful, empathetic, and helpful responses based on the user's concerns or questions."
            "Review provided context information for guidance. Do not mention conversation history directly. Avoid 'boyfriend/her' pronouns where possible."
            "Elaborate in detail using context if needed."
        )

        # Limit context to the most relevant snippets
        context = "\n\n".join(retrieved_docs[0:2])  # Take TOP 2

        # Build the prompt
        prompt = (
            f"{system_prompt}\n\nContext Information:\n{context}\n\n"
            f"Conversation History:\n{conversation_history}\nAssistant:"
        )

        # Log the final prompt for debugging purposes
        print(f"Final Prompt being passed to the LLM:\n{prompt}\n")

        # Send prompt to vLLM server using aiohttp
        vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
        params = {"prompt": prompt, "max_tokens": 2000, "stream": "true"}

        # Add assistant's response to session-specific chat history
        messages.append({"role": "assistant", "content": ""})
        message_index = len(messages) - 1
        await send(
            Div(
                chat_message(message_index, messages=messages),
                id="messages",
                hx_swap_oob="beforeend"
            )
        )

        async with aiohttp.ClientSession() as client_session:
            async with client_session.get(vllm_url, params=params) as response:
                if response.status == 200:
                    # Indicate that response has been received
                    response_received.set()

                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            text = chunk.decode('utf-8')
                            messages[message_index]["content"] += text
                            await send(
                                Span(
                                    text,
                                    id=f"msg-content-{message_index}",
                                    hx_swap_oob="beforeend"
                                )
                            )

                    # Store the assistant's response in the database
                    assistant_content = messages[message_index]["content"]
                    new_assistant_message = Conversation(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role='assistant',
                        content=assistant_content
                    )
                    sqlalchemy_session.add(new_assistant_message)
                    sqlalchemy_session.commit()
                else:
                    # Handle error
                    error_message = "Error: Unable to get response from LLM."
                    messages.append({"role": "assistant", "content": error_message})
                    await send(
                        Div(
                            chat_message(len(messages) - 1, messages=messages),
                            id="messages",
                            hx_swap_oob="beforeend"
                        )
                    )

        # Send the 'Top Sources' to the 'top-sources' div below the chat window
        await send(
            Div(
                chat_top_sources(top_sources),
                id="top-sources",
                hx_swap_oob="innerHTML"
            )
        )

        await send(chat_form(disabled=False))

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()      # Serve the vLLM server
    serve_fasthtml()  # Serve the FastHTML web interface















