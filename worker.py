"""
Runs Whisper in a separate process so the UI can stop it instantly.
Sends messages back to the UI through a queue:
    ("status", text)      - stage update
    ("progress", 0..1)    - fraction of audio transcribed
    ("done", result)      - finished transcription
    ("error", text)       - something went wrong
"""


def transcribe_job(file_path, model_name, language, task, q):
    try:
        import tqdm

        base_tqdm = tqdm.tqdm

        class ProgressTqdm(base_tqdm):
            """Whisper uses tqdm for its progress; forward it to the UI."""

            _last = -1

            def update(self, n=1):
                out = super().update(n)
                if self.total:
                    pct = int(self.n * 100 / self.total)
                    if pct != self._last:
                        self._last = pct
                        q.put(("progress", self.n / self.total))
                return out

        tqdm.tqdm = ProgressTqdm  # whisper calls tqdm.tqdm(...) during transcribe

        import whisper

        q.put(("status", f"Loading '{model_name}' model (first time downloads it)..."))
        model = whisper.load_model(model_name)

        q.put(("status", "Transcribing..."))
        result = model.transcribe(
            file_path,
            language=language,
            task=task,
            fp16=model.device.type != "cpu",
            verbose=False,  # must be False so the progress bar is used
        )
        q.put(("done", result))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))
