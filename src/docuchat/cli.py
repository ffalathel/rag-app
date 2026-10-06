"""`docuchat`: start the app, with flags for the common setups.

    docuchat                  API key from .env, full pipeline
    docuchat --no-ocr         PDFs with real text (not scans): much faster uploads
    docuchat --cpu            no GPU: the small reranker (~3 s instead of ~2 min per question)
    docuchat --fallback       a local model answers when the API call fails
    docuchat --local          the local model only, no API key needed
    docuchat --public         listen on all interfaces, behind a proxy (Codespace, server)

Flags combine. Each one only sets DOCUCHAT_* variables (see config.Settings),
so anything here can also be set by hand or in .env; a flag wins over .env.
"""

import argparse
import os

LOCAL_MODEL = ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf")
SMALL_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def flag_env(args: argparse.Namespace) -> dict[str, str]:
    """The DOCUCHAT_* variables the flags stand for (the local model path is
    added by main(), since resolving it downloads the model)."""
    env = {}
    if args.no_ocr:
        env["DOCUCHAT_DO_OCR"] = "false"
    if args.cpu:
        env["DOCUCHAT_CROSS_ENCODER_NAME"] = SMALL_RERANKER
        env["DOCUCHAT_CROSS_ENCODER_MAX_LENGTH"] = "512"
    if args.local:
        env["DOCUCHAT_LLM_PROVIDER"] = "llamacpp"
    if args.local or args.fallback:
        env["DOCUCHAT_CONTEXT_WINDOW"] = "8192"
    return env


def _local_model_path() -> str:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        raise SystemExit('The local model needs an extra install: pip install -e ".[local-llm]"')
    from huggingface_hub import hf_hub_download

    print(f"Fetching the local model ({LOCAL_MODEL[1]}, ~1.1 GB on first run)...")
    return hf_hub_download(*LOCAL_MODEL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="docuchat", description="Start docuchat. The UI is at http://localhost:<port>.")
    parser.add_argument("--no-ocr", action="store_true",
                        help="skip OCR; faster uploads, but scanned PDFs come out empty")
    parser.add_argument("--cpu", action="store_true",
                        help="use the small reranker, for machines without a GPU")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--fallback", action="store_true",
                       help="answer with a local model when the API call fails")
    model.add_argument("--local", action="store_true",
                       help="answer only with the local model; no API key needed")
    parser.add_argument("--public", action="store_true",
                        help="listen on all interfaces and trust proxy headers")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not os.path.isdir("eval/nodes"):
        raise SystemExit("Run docuchat from the repo folder: it loads the sample documents from eval/.")

    os.environ.update(flag_env(args))
    if args.local or args.fallback:
        os.environ["DOCUCHAT_GGUF_PATH"] = _local_model_path()

    # after the flags, so .env never overrides them (load_dotenv keeps set vars)
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    import uvicorn
    uvicorn.run("docuchat.api:app", port=args.port,
                host="0.0.0.0" if args.public else "127.0.0.1",
                proxy_headers=args.public, forwarded_allow_ips="*" if args.public else None)


if __name__ == "__main__":
    main()
