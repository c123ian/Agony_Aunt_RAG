import modal

# Define the Modal app
app = modal.App("download_model")

# Define the Modal image with necessary packages
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # Download models from the Hugging Face Hub
            "hf-transfer",      # Download models faster with Rust
            "transformers",     # For tokenizer and model handling
            "torch",            # PyTorch for model computations
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Directory to store models
MODELS_DIR = "/llama_mini"

# Default model name
DEFAULT_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Create or access the volume
volume = modal.Volume.from_name("llama_mini", create_if_missing=True)

# Time constants
MINUTES = 60
HOURS = 60 * MINUTES

# Define the app with image and secrets
app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface")])

# Function to download the model
@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Reload the volume to get the latest state
    volume.reload()

    # Download the model weights and config
    snapshot_download(
        model_name,
        local_dir=MODELS_DIR,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        force_download=force_download,
    )

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODELS_DIR)

    # Commit the changes to the volume
    volume.commit()

# Local entrypoint to trigger the download
@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
