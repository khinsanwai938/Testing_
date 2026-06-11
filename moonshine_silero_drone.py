"""
Drone Voice Control — Moonshine v2 + Silero VAD
================================================
Moonshine v2 has VAD built-in inside MicTranscriber.
Silero VAD is added as an EXTRA pre-gate to filter drone motor noise
before audio even reaches Moonshine.

Architecture:
  Microphone → Silero VAD (motor noise filter) → Moonshine MicTranscriber
                                                   (built-in VAD + streaming ASR)
                                                        ↓
                                                  DroneCommandListener
                                                        ↓
                                               Console + transcript file

Install:
    pip install moonshine-voice torch torchaudio pyaudio numpy

Download model (first run only):
    python -m moonshine_voice.download --language en
"""

import os
import sys
import time
import queue
import warnings
import numpy as np
import pyaudio
import torch
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

from moonshine_voice import (
    MicTranscriber,
    TranscriptEventListener,
    get_model_for_language,
    ModelArch,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Moonshine model size — choose based on your hardware:
#   ModelArch.TINY_STREAMING   →  34MB,  12.00% WER  (Raspberry Pi / Nano)
#   ModelArch.SMALL_STREAMING  →  123MB,  7.84% WER  (Jetson Nano / laptop)
#   ModelArch.MEDIUM_STREAMING →  245MB,  6.65% WER  (Jetson Orin / desktop)
MOONSHINE_MODEL = ModelArch.SMALL_STREAMING

# Silero VAD pre-gate settings
# Set USE_SILERO_PREGATING = False to rely only on Moonshine's built-in VAD
USE_SILERO_PREGATING = True

# Speech probability threshold for Silero (0.0 - 1.0)
# Raise to 0.65-0.75 for heavy drone motor environments
SILERO_VAD_THRESHOLD = 0.55

# Silero processes 512 samples at 16kHz (~32ms per window)
SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512
CHUNK_SIZE = 512   # Match VAD window so every chunk is evaluated immediately


# ─────────────────────────────────────────────
# SILERO VAD LOADER
# ─────────────────────────────────────────────

def load_silero_vad():
    print("  Loading Silero VAD pre-gate...")
    try:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        print("  Silero VAD ready.")
        return model
    except Exception as e:
        print(f"[WARNING] Could not load Silero VAD: {e}")
        print("[WARNING] Falling back to Moonshine built-in VAD only.")
        return None


def get_speech_prob(vad_model, audio_float32):
    """Run Silero VAD on a 512-sample audio window. Returns 0.0-1.0."""
    if len(audio_float32) < VAD_WINDOW_SAMPLES:
        padded = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)
        padded[:len(audio_float32)] = audio_float32
        audio_float32 = padded
    tensor = torch.from_numpy(audio_float32[:VAD_WINDOW_SAMPLES])
    with torch.no_grad():
        return vad_model(tensor, SAMPLE_RATE).item()


# ─────────────────────────────────────────────
# MOONSHINE TRANSCRIPT LISTENER
# ─────────────────────────────────────────────

class DroneCommandListener(TranscriptEventListener):
    """
    Moonshine calls:
      on_line_text_changed() → while user is still speaking (partial update)
      on_line_completed()    → when speech segment ends (final result)
    """

    def __init__(self, output_file):
        self.output_file = output_file
        self.last_partial_len = 0

    def on_line_text_changed(self, event):
        """Live preview — updates on same line while speaking."""
        text = event.line.text.strip()
        if text:
            display = f"\r🎙  Listening : {text}"
            sys.stdout.write(display.ljust(100))
            sys.stdout.flush()
            self.last_partial_len = len(display)

    def on_line_completed(self, event):
        """Final transcription — print cleanly and save to file."""
        text = event.line.text.strip()
        if text:
            # Clear the partial line
            sys.stdout.write("\r" + " " * 100 + "\r")
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"✅ [{timestamp}] Command : {text}")
            sys.stdout.flush()

            self.output_file.write(f"[{timestamp}] Command : {text}\n")
            self.output_file.flush()


# ─────────────────────────────────────────────
# SILERO PRE-GATE WRAPPER
# (feeds filtered audio into Moonshine manually)
# ─────────────────────────────────────────────

class SileroGatedMicCapture:
    """
    Captures audio from PyAudio, runs Silero VAD per chunk,
    and only passes speech chunks into the provided Moonshine Transcriber.
    Used when USE_SILERO_PREGATING = True.
    """

    def __init__(self, transcriber, silero_model):
        self.transcriber = transcriber
        self.silero_model = silero_model
        self.audio_queue = queue.Queue()
        self.running = False
        self.last_prob = 0.0

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._callback,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start(self):
        self.running = True
        self.stream.start_stream()
        print("\n[READY] Drone Voice Control active. Speak a command...\n")

        try:
            while self.running and self.stream.is_active():
                try:
                    raw_data = self.audio_queue.get(timeout=0.1)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    prob = get_speech_prob(self.silero_model, audio_float32)
                    self.last_prob = prob

                    if prob >= SILERO_VAD_THRESHOLD:
                        # Speech detected → send to Moonshine
                        bar = int(prob * 20)
                        sys.stdout.write(
                            f"\r🎙  Silero VAD: [{'█' * bar}{'░' * (20 - bar)}]"
                            f" {prob:.2f} — SPEECH "
                        )
                        sys.stdout.flush()
                        # Feed raw int16 bytes to Moonshine transcriber
                        self.transcriber.add_audio(audio_int16)
                    else:
                        bar = int(prob * 20)
                        sys.stdout.write(
                            f"\r⏸   Silero VAD: [{'░' * 20}]"
                            f" {prob:.2f} — silence "
                        )
                        sys.stdout.flush()

                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("   DRONE VOICE CONTROL — MOONSHINE v2 + SILERO VAD")
    print("=" * 60)
    print(f"  Model          : {MOONSHINE_MODEL.name}")
    print(f"  Silero Pre-gate: {'ON' if USE_SILERO_PREGATING else 'OFF (Moonshine VAD only)'}")
    if USE_SILERO_PREGATING:
        print(f"  Silero Threshold: {SILERO_VAD_THRESHOLD}  (raise for noisy motors)")
    print("=" * 60)

    # Load Silero VAD if enabled
    silero_model = None
    if USE_SILERO_PREGATING:
        silero_model = load_silero_vad()
        if silero_model is None:
            print("[INFO] Proceeding with Moonshine built-in VAD only.")

    # Load Moonshine model
    print("  Loading Moonshine v2 model (downloads on first run)...")
    try:
        model_path, model_arch = get_model_for_language(
            wanted_language="en",
            wanted_model_arch=MOONSHINE_MODEL,
        )
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Moonshine model: {e}")
        print("Run: pip install moonshine-voice && python -m moonshine_voice.download --language en")
        sys.exit(1)

    print("  Moonshine v2 ready.\n")

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"drone_moonshine_{timestamp}.txt"
    print(f"  Saving to: {output_filename}")
    print("  Press Ctrl+C to stop.\n")

    with open(output_filename, "a", encoding="utf-8") as f:

        # ── MODE A: Silero pre-gate + Moonshine Transcriber ──
        if USE_SILERO_PREGATING and silero_model is not None:
            from moonshine_voice import Transcriber

            transcriber = Transcriber(model_path=model_path, model_arch=model_arch)
            listener = DroneCommandListener(output_file=f)
            transcriber.add_listener(listener)
            transcriber.start()

            capture = SileroGatedMicCapture(
                transcriber=transcriber,
                silero_model=silero_model,
            )
            try:
                capture.start()   # blocks until Ctrl+C
            finally:
                transcriber.stop()

        # ── MODE B: Moonshine MicTranscriber (built-in VAD only) ──
        else:
            mic_transcriber = MicTranscriber(
                model_path=model_path,
                model_arch=model_arch,
            )
            listener = DroneCommandListener(output_file=f)
            mic_transcriber.add_listener(listener)

            print("\n[READY] Drone Voice Control active. Speak a command...\n")
            mic_transcriber.start()

            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                mic_transcriber.stop()

    print("\n\nSystem safely disconnected.")


if __name__ == "__main__":
    main()
