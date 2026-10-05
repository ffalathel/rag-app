"""Entry point for the Hugging Face Gradio Space: the Space runs this file,
which serves the FastAPI app (Gradio UI mounted at /) the Dockerfile runs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import uvicorn

uvicorn.run("docuchat.api:app", host="0.0.0.0", port=7860, proxy_headers=True, forwarded_allow_ips="*")
