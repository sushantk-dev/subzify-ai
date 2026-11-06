# Subzify AI

Generate transcripts and subtitles from any video or audio file using [OpenAI Whisper](https://github.com/openai/whisper), from the command line or a simple web UI. Runs fully on your own machine.

## Features

- Transcribe video (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, …) and audio (`.mp3`, `.wav`, `.m4a`, …)
- Export as plain text (`.txt`) or subtitles (`.srt`, `.vtt`)
- Auto-detect the spoken language, or pick it yourself (Hindi, Tamil, English and more)
- Translate speech from any language into English
- Web UI with live progress, time-left estimate and a Stop button
- Video preview with the generated subtitles

## Project structure

```
subzify-ai/
├── app.py          # Web UI (Gradio)
├── worker.py       # Runs Whisper in a separate process so jobs can be stopped
├── transcribe.py   # Command-line tool + txt/srt/vtt writers
└── README.md
```

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) installed and on your PATH
- Optional: an NVIDIA GPU with CUDA for much faster transcription

## Installation

1. **Install ffmpeg**

   | OS      | Command                    |
   |---------|----------------------------|
   | Windows | `winget install ffmpeg`    |
   | macOS   | `brew install ffmpeg`      |
   | Ubuntu  | `sudo apt install ffmpeg`  |

   Close and reopen your terminal, then check it works:

   ```bash
   ffmpeg -version
   ```

2. **Create a virtual environment and install packages**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

   pip install -U openai-whisper gradio
   ```

3. **(Optional) Enable GPU acceleration**

   Check whether PyTorch can see your GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   If it prints `False` but you have an NVIDIA GPU, install the CUDA build of PyTorch using the command from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Web UI

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser, then:

1. Upload a video or audio file
2. Choose the model, language, and output formats
3. Click **Transcribe** (click **Stop** at any time to cancel)
4. Copy the transcript, download the files, or preview the video with subtitles

### Command line

```bash
python transcribe.py <file> [options]
```

| Option         | Description                                         | Default     |
|----------------|-----------------------------------------------------|-------------|
| `--model`      | `tiny`, `base`, `small`, `medium`, `large`, `turbo` | `small`     |
| `--language`   | Language code, e.g. `en`, `hi`, `ta`                | auto-detect |
| `--translate`  | Translate speech into English                       | off         |
| `--formats`    | One or more of `txt`, `srt`, `vtt`                  | `txt`       |
| `--outdir`     | Folder to save output files                         | `.`         |

**Examples**

```bash
python transcribe.py video.mp4
python transcribe.py video.mp4 --formats txt srt vtt
python transcribe.py lecture.mp3 --model medium --language hi
python transcribe.py movie.mkv --translate --formats srt --outdir subs
```

## Choosing a model

| Model    | Size    | Speed     | Accuracy | Good for                         |
|----------|---------|-----------|----------|----------------------------------|
| `tiny`   | ~75 MB  | Fastest   | Low      | Quick tests                      |
| `base`   | ~140 MB | Very fast | Fair     | Clear speech                     |
| `small`  | ~460 MB | Fast      | Good     | Everyday use on CPU (default)    |
| `medium` | ~1.5 GB | Slow      | Better   | Accents, non-English audio       |
| `large`  | ~2.9 GB | Slowest   | Best     | Maximum accuracy, needs a GPU    |
| `turbo`  | ~1.6 GB | Fast      | Very good| Best speed/accuracy on a GPU     |

Models are downloaded automatically the first time you use them.

> **Tip:** Long videos on CPU can take hours with larger models. Try `small` on a short clip first, and use a GPU for full-length movies.

## How it works

1. **ffmpeg** extracts the audio track and converts it to 16 kHz mono
2. Whisper splits the audio into 30-second chunks and converts each to a log-mel spectrogram
3. The **encoder** processes the spectrogram; the **decoder** generates text with timestamps
4. The segments are written out as `.txt`, `.srt`, or `.vtt`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: [WinError 2] The system cannot find the file specified` | ffmpeg isn't installed or isn't on PATH. Install it, then **open a new terminal**. |
| Progress stuck on "Loading model" | The model is downloading on first use, so wait for it to finish. |
| Very slow transcription | Use a smaller model, or set up GPU acceleration. |
| Wrong language detected | Set the language manually in the UI or with `--language`. |
| Port 7860 already in use | Close the other running instance, or run `demo.queue().launch(server_port=7861)` in `app.py`. |

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition model
- [Gradio](https://www.gradio.app/) — web UI
- [FFmpeg](https://ffmpeg.org/) — audio/video decoding
