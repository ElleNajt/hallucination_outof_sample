#!/usr/bin/env python3
"""
Deploy Modal backend configured for our experiments.

This script deploys the hallucination probe backend with settings
optimized for the 8B model experiments.
"""

import subprocess
import sys
import os
from pathlib import Path


def deploy_backend(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", n_gpu: int = 1):
    """
    Deploy the Modal backend with specified configuration.

    Args:
        model_name: Model to use as default
        n_gpu: Number of GPUs to use
    """
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Try to load from .env
        env_file = Path(__file__).parent.parent / "hallucination_probes" / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip()
                        break

    if not hf_token:
        print("❌ Error: HF_TOKEN not found. Set it as an environment variable or in hallucination_probes/.env")
        sys.exit(1)

    # Path to modal_backend.py
    backend_path = Path(__file__).parent.parent / "hallucination_probes" / "demo" / "modal_backend.py"

    if not backend_path.exists():
        print(f"❌ Error: Backend not found at {backend_path}")
        sys.exit(1)

    print(f"🚀 Deploying Modal backend...")
    print(f"   Model: {model_name}")
    print(f"   GPUs: {n_gpu}")
    print()

    # Read the backend file
    with open(backend_path) as f:
        backend_code = f.read()

    # Check current settings
    import re
    current_model = re.search(r'DEFAULT_MODEL = "(.*?)"', backend_code)
    current_gpu = re.search(r'N_GPU = (\d+)', backend_code)

    if current_model:
        current_model = current_model.group(1)
        print(f"   Current DEFAULT_MODEL: {current_model}")

    if current_gpu:
        current_gpu = int(current_gpu.group(1))
        print(f"   Current N_GPU: {current_gpu}")

    print()

    # Deploy
    try:
        env = os.environ.copy()
        env["HF_TOKEN"] = hf_token

        # Find modal in venv
        venv_modal = Path(__file__).parent.parent / "venv" / "bin" / "modal"
        modal_cmd = str(venv_modal) if venv_modal.exists() else "modal"

        result = subprocess.run(
            [modal_cmd, "deploy", str(backend_path)],
            cwd=backend_path.parent,
            env=env,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print("\n✅ Backend deployed successfully!")
            print(f"\nNote: Backend will start with {current_model if current_model else 'default model'}")
            print(f"Experiments will call switch_model() to use {model_name} if different.")
        else:
            print(f"\n❌ Deployment failed with exit code {result.returncode}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during deployment: {e}")
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy Modal backend for hallucination probe experiments"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)"
    )

    args = parser.parse_args()

    deploy_backend(args.model, args.gpus)


if __name__ == "__main__":
    main()