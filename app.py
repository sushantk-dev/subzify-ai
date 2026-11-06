"""
Subzify AI - web UI for Whisper transcription.

Files needed in the same folder: app.py, worker.py, transcribe.py

Setup (inside your venv):
    pip install -U gradio openai-whisper

Run:
    python app.py
Then open http://127.0.0.1:7860 in your browser.
"""

import multiprocessing as mp
import queue
import tempfile
import time
from pathlib import Path

import gradio as gr

from transcribe import WRITERS  # reuse the txt / srt / vtt writers
from worker import transcribe_job

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}
LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}

ctx = mp.get_context("spawn")  # same behaviour on Windows, macOS and Linux
current = {"proc": None, "stopped": False}  # the running job, so Stop can kill it


def fmt_eta(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"


def run(file_path, model_name, language, translate, formats, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Upload an audio or video file first.")
    if not formats:
        raise gr.Error("Pick at least one output format.")
    if current["proc"] is not None and current["proc"].is_alive():
        raise gr.Error("A job is already running. Press Stop first.")

    src = Path(file_path)
    q = ctx.Queue()
    proc = ctx.Process(
        target=transcribe_job,
        args=(str(src), model_name, LANGUAGES[language],
              "translate" if translate else "transcribe", q),
        daemon=True,
    )
    current.update(proc=proc, stopped=False)
    proc.start()

    progress(0, desc="Starting...")
    result, started = None, None
    while True:
        try:
            kind, payload = q.get(timeout=0.5)
        except queue.Empty:
            if not proc.is_alive():
                break  # killed by Stop, or crashed
            continue

        if kind == "status":
            progress(0, desc=payload)
        elif kind == "progress":
            started = started or time.time()
            elapsed = time.time() - started
            eta = f" · about {fmt_eta(elapsed / payload - elapsed)} left" if payload > 0.02 else ""
            progress(payload, desc=f"Transcribing{eta}")
        elif kind == "done":
            result = payload
            break
        elif kind == "error":
            raise gr.Error(payload)

    proc.join(timeout=5)
    current["proc"] = None

    if result is None:
        if current["stopped"]:
            return "", None, gr.Video(value=None), "**Stopped.**"
        raise gr.Error("Transcription process ended unexpectedly.")

    outdir = Path(tempfile.mkdtemp(prefix="subzify_"))
    files = []
    for fmt in formats:
        out = outdir / f"{src.stem}.{fmt}"
        WRITERS[fmt](result, out)
        files.append(str(out))

    # Video preview with the subtitles shown as a track
    preview = gr.Video(value=None)
    if src.suffix.lower() in VIDEO_EXTS:
        sub = outdir / f"{src.stem}.vtt"
        if not sub.exists():
            WRITERS["vtt"](result, sub)
        preview = gr.Video(value=str(src), subtitles=str(sub))

    info = f"Detected language: **{result.get('language', '?')}** · {len(result['segments'])} segments"
    return result["text"].strip(), files, preview, info


def stop():
    proc = current["proc"]
    if proc is not None and proc.is_alive():
        current["stopped"] = True
        proc.terminate()
        gr.Info("Stopping transcription...")
    else:
        gr.Info("Nothing is running.")


with gr.Blocks(title="Subzify AI") as demo:
    gr.Markdown("# Subzify AI\nUpload a video or audio file and get a transcript and subtitles.")

    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(
                label="Audio or video file",
                file_types=["audio", "video"],
                type="filepath",
            )
            model_in = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large", "turbo"],
                value="small",
                label="Model",
                info="Bigger = more accurate but slower",
            )
            lang_in = gr.Dropdown(list(LANGUAGES), value="Auto-detect", label="Language")
            translate_in = gr.Checkbox(label="Translate to English")
            formats_in = gr.CheckboxGroup(
                ["txt", "srt", "vtt"], value=["txt", "srt"], label="Output formats"
            )
            with gr.Row():
                btn = gr.Button("Transcribe", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")

        with gr.Column(scale=2):
            info_out = gr.Markdown()
            text_out = gr.Textbox(label="Transcript", lines=12)
            files_out = gr.File(label="Download", file_count="multiple")
            video_out = gr.Video(label="Preview with subtitles")

    btn.click(
        run,
        inputs=[file_in, model_in, lang_in, translate_in, formats_in],
        outputs=[text_out, files_out, video_out, info_out],
        show_progress_on=[text_out],  # one progress bar instead of one per output
    )
    stop_btn.click(stop, queue=False)


if __name__ == "__main__":
    demo.queue().launch()