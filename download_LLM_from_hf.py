# Load model directly
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen3-0.6B", cache_dir='/data/LLMs/')