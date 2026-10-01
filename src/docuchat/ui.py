"""Gradio UI. Every handler calls one Service method; there is no pipeline
logic here. Layout follows the notebook's Gradio app."""

from pathlib import Path

import gradio as gr

from docuchat.service import Service, ServiceError

EXAMPLES = [
    "What is the loan amount and interest rate on the fixed-rate sample loan?",
    "What are the total closing costs?",
    "Is there a prepayment penalty?",
]


def format_answer(result: dict) -> str:
    parts = []
    if result["expired"]:
        parts.append("_Your session expired, so this answer uses the sample documents._\n")
    parts.append(result["answer"])
    if result["sources"]:
        parts.append("\n**Sources**")
        parts += [f"- {s['filename']}, page {s['page_number']}: {s['preview']}"
                  for s in result["sources"]]
    return "\n".join(parts)


def build(service: Service) -> gr.Blocks:
    def chat(message, history, request: gr.Request):
        try:
            return format_answer(service.answer(request.session_hash, message))
        except ServiceError as exc:
            return f"⚠️ {exc}"

    def process(paths, request: gr.Request):
        if not paths:
            return "Choose one or more PDFs first."
        files = [(Path(p).name, Path(p).read_bytes()) for p in paths]
        try:
            chunks = service.ingest_upload(request.session_hash, files)
        except ServiceError as exc:
            return f"⚠️ {exc}"
        return f"Indexed {chunks} chunks from {len(files)} file(s). Questions now use your documents."

    def back_to_sample(request: gr.Request):
        service.reset(request.session_hash)
        return "Using the sample documents."

    with gr.Blocks(title="docuchat") as demo:
        gr.Markdown("# docuchat\nAsk questions about sample CFPB mortgage documents, "
                    "or upload your own PDFs.")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Upload PDF(s)", file_types=[".pdf"],
                                 file_count="multiple")
                process_btn = gr.Button("Process & Index", variant="primary")
                sample_btn = gr.Button("Back to sample documents")
                status = gr.Textbox(label="Status", interactive=False, lines=3,
                                    value="Using the sample documents.")
            with gr.Column(scale=3):
                gr.ChatInterface(chat, examples=EXAMPLES)
        process_btn.click(process, inputs=upload, outputs=status)
        sample_btn.click(back_to_sample, outputs=status)
    return demo
