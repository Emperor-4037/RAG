#!/usr/bin/env python3
import os
import urllib.request
import argparse
from pathlib import Path

def download_file(url: str, dest: Path):
    print(f"Downloading {dest.name}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open(dest, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Successfully downloaded {dest.name}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        # Remove partial files
        if dest.exists():
            dest.unlink()

def main():
    parser = argparse.ArgumentParser(description="Download required ML models.")
    parser.add_argument("--embedding", type=str, default="Xenova/bge-base-en-v1.5", 
                        help="HuggingFace repository for the ONNX embedding model")
    args = parser.parse_args()

    # We use Xenova's ONNX exports as they are standard and reliable for ONNX Runtime.
    # The config in .env should match this directory format.
    model_name = args.embedding.replace("/", "_")
    emb_dir = Path("models") / model_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for HuggingFace Hub raw files
    base_url = f"https://huggingface.co/{args.embedding}/resolve/main"

    files_to_download = [
        ("onnx/model.onnx", "model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("config.json", "config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
    ]

    print(f"--- Downloading Embedding Model ({args.embedding}) ---")
    for remote_path, local_name in files_to_download:
        dest = emb_dir / local_name
        if not dest.exists():
            download_file(f"{base_url}/{remote_path}", dest)
        else:
            print(f"{local_name} already exists, skipping.")
            
    print("\nModel download complete. Ensure .env matches this structure:")
    print(f"EMBEDDING_MODEL_NAME={args.embedding}")

if __name__ == "__main__":
    main()
