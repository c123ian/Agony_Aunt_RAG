from components.assets import arrow_circle_icon, github_icon
from components.chat import chat, chat_form, chat_message, chat_messages
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

# Constants
MODELS_DIR = "/llama_mini"
MODEL_NAME = "Llama-3.2-3B-Instruct"  
FAISS_DATA_DIR = "/faiss_data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
USERNAME = "c123ian"
APP_NAME = "rag-chatbot"
DATABASE_DIR = "/db_data"  # Database directory

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
    "transformers"      # Added for the tokenizer
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
    import os
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.logger import RequestLogger
    import fastapi
    from fastapi.responses import StreamingResponse, JSONResponse
    import uuid
    import asyncio
    from typing import Optional, AsyncGenerator

    # Function to find the model path by searching for 'config.json'
    def find_model_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    # Function to find the tokenizer path by searching for 'tokenizer_config.json'
    def find_tokenizer_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "tokenizer_config.json" in files:
                return root
        return None

    # Check if model files exist
    model_path = find_model_path(MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MODELS_DIR}")

    # Check if tokenizer files exist
    tokenizer_path = find_tokenizer_path(MODELS_DIR)
    if not tokenizer_path:
        raise Exception(f"Could not find tokenizer files in {MODELS_DIR}")

    print(f"Initializing AsyncLLMEngine with model path: {model_path} and tokenizer path: {tokenizer_path}")

    # Create a FastAPI app
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # Create an `AsyncLLMEngine`, the core of the vLLM server.
    engine_args = AsyncEngineArgs(
        model=model_path,      # Use the found model path
        tokenizer=tokenizer_path,  # Use the found tokenizer path
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Get model config using the robust event loop handling
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())

    # Initialize OpenAIServingChat
    request_logger = RequestLogger(max_log_len=256)  # Adjust max_log_len as needed
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        [MODEL_NAME],  # served_model_names
        "assistant",   # response_role
        lora_modules=None,  # Adjust if you're using LoRA
        prompt_adapters=None,  # Adjust if you're using prompt adapters
        request_logger=request_logger,
        chat_template=None,  # Adjust if you have a specific chat template
    )

    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 6000, stream: bool = False):
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.5,  # Encourage the model to use new tokens
            stop=["User:", "\n\n"]
        )

        async def completion_generator() -> AsyncGenerator[str, None]:
            try:
                full_response = ""
                assistant_prefix_removed = False
                last_yielded_position = 0
                async for result in engine.generate(prompt, sampling_params, request_id):
                    if len(result.outputs) > 0:
                        new_text = result.outputs[0].text
                        if not assistant_prefix_removed:
                            new_text = new_text.split("Assistant:")[-1].lstrip()
                            assistant_prefix_removed = True

                        if len(new_text) > last_yielded_position:
                            new_part = new_text[last_yielded_position:]
                            yield new_part
                            last_yielded_position = len(new_text)

                        full_response = new_text
            except Exception as e:
                yield str(e)

        if stream:
            return StreamingResponse(
                completion_generator(), media_type="text/event-stream"
            )
        else:
            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            return JSONResponse(content={"choices": [{"text": completion.strip()}]})

    return web_app

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

    # Set up the database using MiniDataAPI
    db_path = os.path.join(DATABASE_DIR, 'chat_history.db')
    db = Database(db_path)

    # Define the Conversation class for the database table
    class Conversation:
        id: int
        session_id: str
        role: str
        content: str

    # Create the conversations table
    conversations = db.create(Conversation)

    @rt("/")
    async def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

        return Div(
            H1(
                "Chat with Agony Aunt",
                cls="text-3xl font-bold mb-4 text-white"
            ),
            chat(session_id=session_id),  # Pass session_id here
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
        # Check if session is None and initialize if necessary
        if session is None:
            session = {}
            # Optionally, store the session back in some persistent storage if needed
        
        # Ensure session_id is set
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

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

        conversations.insert(
            session_id=session_id,
            role='user',
            content=msg
        )

        # Add user's message to chat history
        chat_messages.append({"role": "user", "content": msg})
        await send(chat_form(disabled=True))
        await send(
            Div(
                chat_message(len(chat_messages) - 1),
                id="messages",
                hx_swap_oob="beforeend"
            )
        )

        # Compute embedding of user's message
        question_embedding = emb_model.encode([msg], normalize_embeddings=True)
        question_embedding = question_embedding.astype('float32')

        # Retrieve top K similar documents, but later we take only the most relevant retrieved document
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

        # Build conversation history (limit to last N messages to stay within context window)
        def build_conversation(chat_messages, max_length=2000):
            conversation = ''
            total_length = 0
            # Start from the latest messages
            for message in reversed(chat_messages):
                role = message['role']
                content = message['content']
                message_text = f"{role.capitalize()}: {content}\n"
                total_length += len(message_text)
                if total_length > max_length:
                    break
                conversation = message_text + conversation
            return conversation

        conversation_history = build_conversation(chat_messages)

        # System prompt
        system_prompt = (
            "You are an 'Agony Aunt' who helps individuals clarify their options and think through their choices. "
            "Provide thoughtful, empathetic, and helpful responses based on the user's concerns or questions."
            "Review provided context information for guidance. Do not mention conversation history directly. Avoid 'boyfriend/her' pronouns where possible."
            "Elaborate in detail using context if needed."
        )

        # Limit context to the most relevant snippet (e.g., the top result)
        context = "\n\n".join(retrieved_docs[0:2])  # Take TOP 2

        # Build the prompt by concatenating the system prompt, context, and conversation history
        prompt = (
            f"{system_prompt}\n\nContext Information:\n{context}\n\n"
            f"Conversation History:\n{conversation_history}\nAssistant:"
        )

        # Log the final prompt for debugging purposes
        print(f"Final Prompt being passed to the LLM:\n{prompt}\n")

        # Send prompt to vLLM server using aiohttp
        vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
        params = {"prompt": prompt, "max_tokens": 2000, "stream": "true"}

        async with aiohttp.ClientSession() as client_session:
            async with client_session.get(vllm_url, params=params) as response:
                if response.status == 200:
                    # Indicate that response has been received
                    response_received.set()

                    # Add assistant's response to chat history
                    chat_messages.append({"role": "assistant", "content": ""})
                    message_index = len(chat_messages) - 1
                    await send(
                        Div(
                            chat_message(message_index),
                            id="messages",
                            hx_swap_oob="beforeend"
                        )
                    )

                    # Stream response to the user
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            text = chunk.decode('utf-8')
                            chat_messages[message_index]["content"] += text
                            await send(
                                Span(
                                    text,
                                    id=f"msg-content-{message_index}",
                                    hx_swap_oob="beforeend"
                                )
                            )

                    # Store the assistant's response in the database
                    assistant_content = chat_messages[message_index]["content"]
                    conversations.insert(
                        session_id=session_id,
                        role='assistant',
                        content=assistant_content
                    )
                else:
                    # Handle error
                    message = "Error: Unable to get response from LLM."
                    chat_messages.append({"role": "assistant", "content": message})
                    await send(
                        Div(
                            chat_message(len(chat_messages) - 1),
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







