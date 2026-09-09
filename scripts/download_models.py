#!/usr/bin/env python3
import os
import urllib.request
import argparse
from pathlib import Path

def download_file(url: str, dest: Path):
    print(f"  Downloading {dest.name}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open(dest, 'wb') as out_file:
                out_file.write(response.read())
        print(f"  -> Success: {dest.name}")
    except Exception as e:
        print(f"  -> Failed to download {url}: {e}")
        if dest.exists():
            dest.unlink()

def fetch_hf_onnx_model(repo_id: str):
    model_name = repo_id.replace("/", "_")
    model_dir = Path("models") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    files_to_download = [
        ("onnx/model.onnx", "model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("config.json", "config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
    ]

    print(f"\n=== Downloading Model: {repo_id} ===")
    for remote_path, local_name in files_to_download:
        dest = model_dir / local_name
        if not dest.exists():
            download_file(f"{base_url}/{remote_path}", dest)
        else:
            print(f"  -> Skipped: {local_name} (already exists)")

def main():
    parser = argparse.ArgumentParser(description="Download required ML models.")
    parser.add_argument("--embedding", type=str, default="Xenova/bge-base-en-v1.5", 
                        help="HuggingFace ONNX embedding model")
    parser.add_argument("--reranker", type=str, default="Xenova/ms-marco-MiniLM-L-6-v2", 
                        help="HuggingFace ONNX cross-encoder model")
    args = parser.parse_args()

    fetch_hf_onnx_model(args.embedding)
    fetch_hf_onnx_model(args.reranker)
            
    print("\nModel downloads complete.")
    print("Ensure your .env contains:")
    print(f"EMBEDDING_MODEL_NAME={args.embedding}")
    print(f"RERANKER_MODEL_NAME={args.reranker}")

if __name__ == "__main__":
    main()
