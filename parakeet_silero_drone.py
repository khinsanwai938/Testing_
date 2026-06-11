"""
Drone Voice Control — NVIDIA Parakeet TDT 1.1B + Silero VAD
============================================================
Architecture:
  Microphone → PyAudio chunks
      → Silero VAD (motor/wind noise filter)
          → speech buffer accumulates
              → Parakeet TDT 1.1B (NeMo) transcribes on silence
                  → Console print + transcript file saved

Why Parakeet TDT 1.1B for drones?
  - Trained on 64K hours of English speech
  - RTFx > 2000 (processes audio far faster than real-time)
  - No 30s window constraint (unlike Whisper)
  - Lowercase output — clean for command parsing
  - CC-BY-4.0 license — free for commercial use

Install:
    pip install nemo_toolkit[asr] torch torchaudio pyaudio numpy

    NOTE: nemo_toolkit is large (~2GB). Use a venv.
    If nemo install is slow, try:
        pip install 'nemo_toolkit[asr]' --find-links https://download.pytorch.org/whl/torch_stable.html

Model download: happens automatically on first run (~4.4GB for 1.1B).
For lighter option swap MODEL_NAME to "nvidia/parakeet-tdt-0.6b-v2" (~2.2GB).
"""

import os
import sys
import queue
import warnings
import numpy as np
import pyaudio
import torch
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Model selection:
#   "nvidia/parakeet-tdt-1.1b"    → 1.1B params, highest accuracy, ~4.4GB
#   "nvidia/parakeet-tdt-0.6b-v2" → 600M params, faster, ~2.2GB (recommended for Jetson)
MODEL_NAME = "nvidia/parakeet-tdt-1.1b"

SAMPLE_RATE = 16000
CHUNK_SIZE = 512           # 32ms per chunk — matches Silero VAD window perfectly

# ── Silero VAD ──
# Raise threshold to 0.65-0.75 if drone motor noise causes false triggers
VAD_THRESHOLD = 0.55
VAD_WINDOW_SAMPLES = 512   # Silero requires exactly 512 samples at 16kHz

# ── Speech buffering ──
# Minimum speech duration (seconds) before transcribing — filters noise bursts
MIN_SPEECH_DURATION = 0.4

# Seconds of silence after speech to trigger transcription (Parakeet is batch-style,
# so we collect full utterance then transcribe — unlike streaming models)
SILENCE_TRIGGER_TIME = 1.0

# Maximum buffer duration (seconds) before force-transcribing
# Prevents memory growth during very long continuous speech
MAX_BUFFER_DURATION = 15.0


# ─────────────────────────────────────────────
# SILERO VAD
# ─────────────────────────────────────────────

def load_silero_vad():
    print("  Loading Silero VAD...")
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
        print(f"[CRITICAL] Silero VAD failed to load: {e}")
        sys.exit(1)


def get_speech_prob(vad_model, audio_float32):
    """Score a 512-sample window. Returns probability 0.0-1.0."""
    if len(audio_float32) < VAD_WINDOW_SAMPLES:
        padded = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)
        padded[:len(audio_float32)] = audio_float32
        audio_float32 = padded
    tensor = torch.from_numpy(audio_float32[:VAD_WINDOW_SAMPLES])
    with torch.no_grad():
        return vad_model(tensor, SAMPLE_RATE).item()


# ─────────────────────────────────────────────
# PARAKEET TDT LOADER
# ─────────────────────────────────────────────

def load_parakeet(model_name):
    print(f"  Loading Parakeet TDT ({model_name})...")
    print("  (First run downloads ~4.4GB — subsequent runs use cache)\n")
    try:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=model_name
        )
        model.eval()
        # Move to GPU if available — massive speed boost on Jetson
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"  Parakeet TDT ready on {device.upper()}.")
        return model, device
    except ImportError:
        print("[CRITICAL] NeMo not installed.")
        print("Run: pip install 'nemo_toolkit[asr]'")
        sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Failed to load Parakeet: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
# TRANSCRIBE WITH PARAKEET
# ─────────────────────────────────────────────

def transcribe_buffer(parakeet_model, audio_buffer_float32):
    """
    Transcribe a numpy float32 audio array using Parakeet TDT.
    Parakeet expects a list of audio arrays (batch).
    Returns transcribed text string.
    """
    try:
        # NeMo's transcribe() accepts raw numpy arrays directly
        output = parakeet_model.transcribe(
            [audio_buffer_float32],   # batch of 1
            batch_size=1,
        )
        # Output is a list of Hypothesis objects or strings depending on NeMo version
        if hasattr(output[0], 'text'):
            return output[0].text.strip()
        else:
            return str(output[0]).strip()
    except Exception as e:
        print(f"\n[WARN] Transcription error: {e}")
        return ""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 62)
    print("   DRONE VOICE CONTROL — PARAKEET TDT 1.1B + SILERO VAD")
    print("=" * 62)
    print(f"  Model           : {MODEL_NAME}")
    print(f"  VAD Threshold   : {VAD_THRESHOLD}  (raise to 0.7 for noisy rotors)")
    print(f"  Min Speech      : {MIN_SPEECH_DURATION}s")
    print(f"  Silence Trigger : {SILENCE_TRIGGER_TIME}s")
    print(f"  Max Buffer      : {MAX_BUFFER_DURATION}s")
    print("=" * 62)

    # Load models
    silero_model = load_silero_vad()
    parakeet_model, device = load_parakeet(MODEL_NAME)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"drone_parakeet_{timestamp}.txt"
    print(f"\n  Saving to : {output_filename}")
    print("  Press Ctrl+C to stop.\n")

    # ── PyAudio setup ──
    audio_queue = queue.Queue()

    def stream_callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=stream_callback,
        )
    except Exception as e:
        print(f"[CRITICAL] Could not open microphone: {e}")
        p.terminate()
        sys.exit(1)

    # ── VAD + buffer state ──
    speech_buffer = np.zeros(0, dtype=np.float32)  # accumulates confirmed speech
    speech_duration = 0.0      # seconds of speech in current segment
    silence_time = 0.0         # seconds of silence since last speech
    is_in_speech = False       # are we currently in a speech segment?
    last_prob = 0.0

    print("=" * 62)
    print("[READY] Listening for drone commands...\n")

    try:
        stream.start_stream()

        with open(output_filename, "a", encoding="utf-8") as f:
            while stream.is_active():
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)

                    if len(audio_int16) == 0:
                        continue

                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    chunk_duration = len(audio_float32) / SAMPLE_RATE

                    # ── Silero VAD ──
                    prob = get_speech_prob(silero_model, audio_float32)
                    last_prob = prob
                    chunk_is_speech = prob >= VAD_THRESHOLD

                    # ── Update state ──
                    if chunk_is_speech:
                        is_in_speech = True
                        silence_time = 0.0
                        speech_duration += chunk_duration
                        speech_buffer = np.concatenate((speech_buffer, audio_float32))

                        bar = int(prob * 20)
                        sys.stdout.write(
                            f"\r🎙  VAD [{('█' * bar).ljust(20, '░')}]"
                            f" {prob:.2f}  speech {speech_duration:.1f}s"
                        )
                        sys.stdout.flush()

                    else:
                        if is_in_speech:
                            silence_time += chunk_duration
                            # Keep buffering short silences (catches trailing phonemes)
                            speech_buffer = np.concatenate((speech_buffer, audio_float32))

                        bar = int(prob * 20)
                        sys.stdout.write(
                            f"\r⏸   VAD [{('░' * 20)}]"
                            f" {prob:.2f}  silence {silence_time:.1f}s "
                        )
                        sys.stdout.flush()

                    # ── Trigger transcription on silence OR buffer overflow ──
                    should_transcribe = (
                        is_in_speech and (
                            silence_time >= SILENCE_TRIGGER_TIME or
                            speech_duration >= MAX_BUFFER_DURATION
                        )
                    )

                    if should_transcribe:
                        if speech_duration >= MIN_SPEECH_DURATION:
                            sys.stdout.write("\r" + " " * 80 + "\r")
                            sys.stdout.write("⚡ Transcribing...\n")
                            sys.stdout.flush()

                            text = transcribe_buffer(parakeet_model, speech_buffer)

                            if text:
                                ts = datetime.now().strftime("%H:%M:%S")
                                sys.stdout.write("\r" + " " * 80 + "\r")
                                print(f"✅ [{ts}] Command : {text}")
                                sys.stdout.flush()

                                f.write(f"[{ts}] Command : {text}\n")
                                f.flush()
                        else:
                            sys.stdout.write(
                                f"\r[VAD] Ignored burst "
                                f"({speech_duration:.2f}s < {MIN_SPEECH_DURATION}s min)"
                                f"{'  ' * 20}\n"
                            )
                            sys.stdout.flush()

                        # ── COLD RESET ──
                        speech_buffer = np.zeros(0, dtype=np.float32)
                        speech_duration = 0.0
                        silence_time = 0.0
                        is_in_speech = False

                        # Flush stale audio
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break

                        print("[READY] Listening for next command...\n")

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n\nShutting down Drone Voice Control...")
    finally:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()
        print("System safely disconnected.")


if __name__ == "__main__":
    main()
