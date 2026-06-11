import os
import sys
import json
import queue
import numpy as np
import pyaudio
import torch
import warnings
from datetime import datetime
from vosk import Model, KaldiRecognizer, SetLogLevel

warnings.filterwarnings("ignore", category=UserWarning)

# Mute Vosk/Kaldi internal C++ logging chatter to keep console clean
SetLogLevel(-1)

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# --- SILERO VAD CONFIGURATION ---
# Speech probability threshold (0.0 - 1.0)
# Raise to 0.65-0.75 if drone motor noise causes false triggers
VAD_THRESHOLD = 0.5

# Silero VAD requires exactly 512 samples at 16kHz (~32ms per window)
VAD_WINDOW_SAMPLES = 512

# Minimum continuous speech duration (seconds) before passing to Vosk
# Filters out short motor noise bursts
MIN_SPEECH_DURATION = 0.3

# Seconds of silence after speech before resetting — lets Vosk finalize
SILENCE_RESET_TIME = 1.0


def load_silero_vad():
    """Load Silero VAD model from torch hub."""
    print("Loading Silero VAD model...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False
        )
        print("Silero VAD loaded successfully.")
        return model
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Silero VAD: {e}")
        print("Try: pip install torch torchaudio")
        sys.exit(1)


def get_speech_prob(vad_model, audio_float32):
    """
    Run Silero VAD on a 512-sample window.
    Returns speech probability (0.0 to 1.0).
    """
    if len(audio_float32) < VAD_WINDOW_SAMPLES:
        padded = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)
        padded[:len(audio_float32)] = audio_float32
        audio_float32 = padded
    else:
        audio_float32 = audio_float32[:VAD_WINDOW_SAMPLES]

    tensor = torch.from_numpy(audio_float32)
    with torch.no_grad():
        prob = vad_model(tensor, SAMPLE_RATE).item()
    return prob


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("  DRONE VOICE CONTROL — VOSK + SILERO VAD")
    print("=" * 60)

    # Load Silero VAD
    vad_model = load_silero_vad()

    # Load Vosk
    print("Loading Vosk engine...")
    model = Model(lang="en-us")
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"drone_transcript_{timestamp}.txt"

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
            stream_callback=stream_callback
        )
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not open microphone: {e}")
        p.terminate()
        sys.exit(1)

    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("  DRONE VOICE CONTROL — VOSK + SILERO VAD")
    print("=" * 60)
    print(f"  VAD Threshold  : {VAD_THRESHOLD}  (raise to 0.7 for noisy drone env)")
    print(f"  Min Speech     : {MIN_SPEECH_DURATION}s (bursts shorter than this ignored)")
    print(f"  Silence Reset  : {SILENCE_RESET_TIME}s (resets Vosk after pause)")
    print(f"  Saving to      : {output_filename}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # --- VAD state ---
    vad_chunk_buffer = np.zeros(0, dtype=np.float32)  # staging buffer for 512-sample windows
    is_in_speech = False           # currently inside a speech segment?
    silence_time = 0.0             # accumulated silence duration
    speech_duration = 0.0          # accumulated speech duration in current phrase
    last_speech_prob = 0.0

    try:
        stream.start_stream()
        print("\n[READY] Listening for drone commands...\n")

        with open(output_filename, "a", encoding="utf-8") as f:
            while stream.is_active():
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)

                    if len(audio_int16) == 0:
                        continue

                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    chunk_duration = len(audio_float32) / SAMPLE_RATE

                    # --- Run Silero VAD in 512-sample windows ---
                    vad_chunk_buffer = np.concatenate((vad_chunk_buffer, audio_float32))
                    chunk_is_speech = False

                    while len(vad_chunk_buffer) >= VAD_WINDOW_SAMPLES:
                        window = vad_chunk_buffer[:VAD_WINDOW_SAMPLES]
                        vad_chunk_buffer = vad_chunk_buffer[VAD_WINDOW_SAMPLES:]
                        prob = get_speech_prob(vad_model, window)
                        last_speech_prob = prob
                        if prob >= VAD_THRESHOLD:
                            chunk_is_speech = True

                    # --- Update VAD state ---
                    if chunk_is_speech:
                        silence_time = 0.0
                        is_in_speech = True
                        speech_duration += chunk_duration

                        # Show live VAD bar
                        bar = int(last_speech_prob * 20)
                        print(
                            f"\r🎙  VAD: [{'█' * bar}{'░' * (20 - bar)}] {last_speech_prob:.2f}",
                            end="", flush=True
                        )
                    else:
                        if is_in_speech:
                            silence_time += chunk_duration
                        bar = int(last_speech_prob * 20)
                        print(
                            f"\r⏸   VAD: [{'░' * 20}] {last_speech_prob:.2f}"
                            f" (silence {silence_time:.1f}s)",
                            end="", flush=True
                        )

                    # --- Gate: only feed audio to Vosk during active speech ---
                    if is_in_speech:

                        # Pass raw bytes to Vosk (it expects int16 PCM)
                        if recognizer.AcceptWaveform(raw_data):
                            result_json = json.loads(recognizer.Result())
                            transcript = result_json.get("text", "").strip()

                            if transcript and speech_duration >= MIN_SPEECH_DURATION:
                                sys.stdout.write("\r" + " " * 100 + "\r")
                                sys.stdout.write(f"✅ Command : {transcript}\n")
                                sys.stdout.flush()
                                f.write(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                                    f"Command : {transcript}\n"
                                )
                                f.flush()

                        else:
                            partial_json = json.loads(recognizer.PartialResult())
                            partial_text = partial_json.get("partial", "").strip()

                            if partial_text:
                                sys.stdout.write(
                                    f"\rListening : {partial_text}...".ljust(100)
                                )
                                sys.stdout.flush()

                        # --- Reset after sustained silence ---
                        if silence_time >= SILENCE_RESET_TIME:
                            # Force Vosk to emit any buffered final result
                            final_json = json.loads(recognizer.FinalResult())
                            final_text = final_json.get("text", "").strip()

                            if final_text and speech_duration >= MIN_SPEECH_DURATION:
                                sys.stdout.write("\r" + " " * 100 + "\r")
                                sys.stdout.write(f"✅ Command : {final_text}\n")
                                sys.stdout.flush()
                                f.write(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                                    f"Command : {final_text}\n"
                                )
                                f.flush()
                            elif speech_duration < MIN_SPEECH_DURATION and speech_duration > 0:
                                sys.stdout.write(
                                    f"\r[VAD] Ignored short burst "
                                    f"({speech_duration:.2f}s){' ' * 40}\n"
                                )
                                sys.stdout.flush()

                            # --- COLD RESET ---
                            recognizer = KaldiRecognizer(model, SAMPLE_RATE)  # fresh Vosk state
                            vad_chunk_buffer = np.zeros(0, dtype=np.float32)
                            is_in_speech = False
                            silence_time = 0.0
                            speech_duration = 0.0
                            last_speech_prob = 0.0

                            # Flush stale audio from queue
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except queue.Empty:
                                    break

                            print("\n[READY] Listening for drone commands...\n")

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n\nShutting down Drone Voice Control Pipeline...")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("System safely disconnected.")


if __name__ == "__main__":
    main()
