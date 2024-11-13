import modal
import os

# Define Modal image with faiss-cpu, pandas, numpy, huggingface_hub, and sentence-transformers
pandas_faiss_image = modal.Image.debian_slim().pip_install(
    "faiss-cpu", 
    "pandas", 
    "numpy", 
    "huggingface_hub", 
    "sentence-transformers"  # Add sentence-transformers
)

# Define volume and constants
app = modal.App("download_faiss_data")
FAISS_DATA_DIR = "/faiss_data"
DATASET_NAME = "c123ian/dear_deidre_agony_aunt"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
faiss_volume = modal.Volume.from_name("faiss_data", create_if_missing=True)

# Import pandas, numpy, and faiss inside Modal function context
with pandas_faiss_image.imports():
    from huggingface_hub import snapshot_download
    import pandas as pd
    import numpy as np
    import faiss  # Import faiss-cpu
    from sentence_transformers import SentenceTransformer  # Now it should work

@app.function(image=pandas_faiss_image, volumes={FAISS_DATA_DIR: faiss_volume}, timeout=4 * 60 * 60)
def download_and_store_data(force_download=False):
    # Download the dataset from Hugging Face
    dataset_dir = snapshot_download(
        DATASET_NAME,
        local_dir=FAISS_DATA_DIR,
        force_download=force_download,
        repo_type="dataset"  # Specify it's a dataset, not a model
    )
    
    # Adjust the path to the correct file name based on your volume listing
    csv_file = os.path.join(FAISS_DATA_DIR, "dear-deidre_procesed_fast_post.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at {csv_file}")

    # Load dataset using pandas
    df = pd.read_csv(csv_file)

    # Combine relevant text fields
    df['combined_text'] = df['data-original-text'] + " " + df['data-headline']

    # Initialize embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Generate embeddings for the combined text
    embeddings = embedding_model.encode(df['combined_text'].tolist(), convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Normalize embeddings for FAISS
    faiss.normalize_L2(embeddings)

    # Create FAISS index and add embeddings
    # Using IndexFlatIP (Inner Product) with normalized vectors, which means you're using cosine similarity.
    # cosine similarity scores, ranging from -1 (completely dissimilar) to 1 (identical).
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index to volume
    faiss.write_index(index, f"{FAISS_DATA_DIR}/faiss_index.bin")

    # Save the DataFrame to a pickle file for later retrieval
    df.to_pickle(f"{FAISS_DATA_DIR}/data.pkl")

    # Save the embedding model to the volume
    embedding_model.save(f"{FAISS_DATA_DIR}/embedding_model")

@app.local_entrypoint()
def main(force_download: bool = False):
    download_and_store_data.remote(force_download)



