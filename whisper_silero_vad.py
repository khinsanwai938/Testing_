import os
import sys
import queue
import numpy as np
import pyaudio
import warnings
import torch
from datetime import datetime
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# --- SILERO VAD CONFIGURATION ---
# Probability threshold: 0.0 - 1.0
# Higher = less sensitive (fewer false triggers from noise/motor hum)
# Lower  = more sensitive (catches quiet speech)
# Recommended for drone: 0.6 - 0.75 (motor noise environment)
VAD_THRESHOLD = 0.5

# How many seconds of non-speech to wait before finalizing a sentence
SILENCE_WINDOW = 1.2

# Minimum speech duration (seconds) to bother transcribing — filters out short noise bursts
MIN_SPEECH_DURATION = 0.3

# Silero VAD processes 512 samples at 16kHz (~32ms per chunk)
VAD_WINDOW_SAMPLES = 512


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
        (get_speech_timestamps, _, read_audio, _, _) = utils
        print("Silero VAD loaded successfully.")
        return model, get_speech_timestamps
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Silero VAD: {e}")
        print("Try: pip install torch torchaudio silero-vad")
        sys.exit(1)


def is_speech_chunk(vad_model, audio_chunk_float32):
    """
    Run Silero VAD on a single audio chunk.
    Returns speech probability (0.0 to 1.0).
    """
    # Silero needs exactly VAD_WINDOW_SAMPLES; pad or trim if needed
    if len(audio_chunk_float32) < VAD_WINDOW_SAMPLES:
        padded = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)
        padded[:len(audio_chunk_float32)] = audio_chunk_float32
        audio_chunk_float32 = padded
    else:
        audio_chunk_float32 = audio_chunk_float32[:VAD_WINDOW_SAMPLES]

    tensor = torch.from_numpy(audio_chunk_float32)
    with torch.no_grad():
        speech_prob = vad_model(tensor, SAMPLE_RATE).item()
    return speech_prob


def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load Silero VAD
    vad_model, _ = load_silero_vad()

    # Load Whisper
    print("Loading Faster-Whisper model...")
    try:
        whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Whisper engine: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"whisper_silero_transcript_{timestamp}.txt"

    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== WHISPER + SILERO VAD TRANSCRIPTION ===")
    print(f" VAD Threshold  : {VAD_THRESHOLD}  (raise if motor noise triggers false starts)")
    print(f" Silence Window : {SILENCE_WINDOW}s (pause needed to finalize sentence)")
    print(f" Min Speech     : {MIN_SPEECH_DURATION}s (shorter bursts ignored)")
    print(f" Saving to      : {output_filename}")
    print(" Press Ctrl+C to stop.\n")
    print("=" * 60)

    audio_queue = queue.Queue()
    audio_buffer = np.zeros(0, dtype=np.float32)       # Accumulates speech audio
    vad_chunk_buffer = np.zeros(0, dtype=np.float32)   # Accumulates chunks for VAD windowing

    continuous_silence_time = 0.0
    has_spoken_in_phrase = False
    speech_duration = 0.0  # Track total speech time in current phrase

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

    try:
        stream.start_stream()
        print("Listening...\n")

        with open(output_filename, "a", encoding="utf-8") as f:
            while stream.is_active():
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)

                    if len(audio_int16) == 0:
                        continue

                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    chunk_duration = len(audio_float32) / SAMPLE_RATE

                    # --- Feed into VAD chunk buffer ---
                    vad_chunk_buffer = np.concatenate((vad_chunk_buffer, audio_float32))

                    # Run Silero VAD when we have enough samples (512 samples = ~32ms)
                    is_speech = False
                    speech_prob = 0.0
                    while len(vad_chunk_buffer) >= VAD_WINDOW_SAMPLES:
                        window = vad_chunk_buffer[:VAD_WINDOW_SAMPLES]
                        vad_chunk_buffer = vad_chunk_buffer[VAD_WINDOW_SAMPLES:]
                        speech_prob = is_speech_chunk(vad_model, window)
                        if speech_prob >= VAD_THRESHOLD:
                            is_speech = True

                    # --- Update state based on VAD result ---
                    if is_speech:
                        continuous_silence_time = 0.0
                        has_spoken_in_phrase = True
                        speech_duration += chunk_duration
                        audio_buffer = np.concatenate((audio_buffer, audio_float32))

                        # Show live VAD indicator
                        bar = int(speech_prob * 20)
                        print(f"\r🎙  VAD: [{'█' * bar}{'░' * (20 - bar)}] {speech_prob:.2f}", end="", flush=True)

                    else:
                        continuous_silence_time += chunk_duration
                        # Still buffer audio during short silences (captures trailing phonemes)
                        if has_spoken_in_phrase:
                            audio_buffer = np.concatenate((audio_buffer, audio_float32))

                        # Show silence indicator
                        print(f"\r⏸   VAD: [{'░' * 20}] {speech_prob:.2f} (silence {continuous_silence_time:.1f}s)", end="", flush=True)

                    # --- Scenario A: Live preview while speaking ---
                    if has_spoken_in_phrase and continuous_silence_time < SILENCE_WINDOW:
                        if len(audio_buffer) > 0 and len(audio_buffer) % (SAMPLE_RATE // 2) < CHUNK_SIZE:
                            segments, _ = whisper_model.transcribe(
                                audio_buffer,
                                language="en",
                                beam_size=5,
                                vad_filter=True,
                                condition_on_previous_text=False,
                                temperature=0.0
                            )
                            text = " ".join([seg.text for seg in segments]).strip()
                            if text:
                                print(f"\rListening: {text:<80}", end="", flush=True)

                    # --- Scenario B: Silence detected -> finalize ---
                    elif has_spoken_in_phrase and continuous_silence_time >= SILENCE_WINDOW:

                        # Skip if speech was too short (likely noise burst)
                        if speech_duration < MIN_SPEECH_DURATION:
                            print(f"\r[VAD] Ignored short burst ({speech_duration:.2f}s < {MIN_SPEECH_DURATION}s min){' ' * 40}")
                        else:
                            segments, _ = whisper_model.transcribe(
                                audio_buffer,
                                language="en",
                                beam_size=5,
                                vad_filter=True,
                                condition_on_previous_text=False,
                                temperature=0.0
                            )
                            final_text = " ".join([seg.text for seg in segments]).strip()

                            if final_text:
                                sys.stdout.write("\r" + " " * 120 + "\r")
                                print(f"✅ Sentence: {final_text}")
                                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {final_text}\n")
                                f.flush()

                        # --- COLD RESET ---
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        vad_chunk_buffer = np.zeros(0, dtype=np.float32)
                        has_spoken_in_phrase = False
                        continuous_silence_time = 0.0
                        speech_duration = 0.0

                        # Flush stale audio from queue
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break

                        print("\nListening...\n")

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n\nStream stopped by user.")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Disconnected safely.")


if __name__ == "__main__":
    main()
