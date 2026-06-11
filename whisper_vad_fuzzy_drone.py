import os
import sys
import queue
import numpy as np
import pyaudio
import warnings
import torch
from datetime import datetime
from faster_whisper import WhisperModel
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SAMPLE_RATE    = 16000
CHUNK_SIZE     = 512       # Match Silero VAD window (512 samples = 32ms)
MODEL_SIZE     = "small"
DEVICE         = "cpu"
COMPUTE_TYPE   = "int8"

# ── Silero VAD ──
# Raise to 0.65–0.75 if drone motor noise causes false triggers
VAD_THRESHOLD     = 0.55
VAD_WINDOW_SAMPLES = 512

# Seconds of silence after speech to finalize a sentence
SILENCE_WINDOW    = 1.2

# Minimum speech duration to bother transcribing (filters motor noise bursts)
MIN_SPEECH_DURATION = 0.3

# ── Fuzzy Command Matching ──
MATCH_THRESHOLD = 75.0   # Scores above 75% trigger the command action

# ─────────────────────────────────────────────
# DRONE COMMAND DICTIONARY
# ─────────────────────────────────────────────

COMMAND_DICTIONARY = {
    "ARM": [
        "arm the drone",
        "arm engines",
        "turn on motors",
        "system unlock",
        "engage motors",
        "start up the drone",
        "unlock flight controller",
        "system arm",
        "arm",
        "turn on the drone",
        "let's ready the drone for flight",
        "get the props spinning"
    ],
    "DISARM": [
        "disarm the drone",
        "disarm engines",
        "cut power",
        "turn off motors",
        "kill the motors",
        "stop the propellers",
        "lock system",
        "cut the engines",
        "shut down the engines",
        "power down the drone",
        "okay you can turn off the power now"
    ],
    "TAKEOFF": [
        "take off",
        "launch drone",
        "start flying",
        "go up",
        "lift off",
        "begin flight",
        "fly up",
        "fly up into the air",
        "lift off the ground",
        "go ahead"
    ],
    "RTL": [
        "return to launch",
        "come back home",
        "go home",
        "return home",
        "rtl",
        "land at base",
        "fly back",
        "return to base",
        "go to home position",
        "fly back to where you started",
        "bring the drone back to me",
        "time to come home",
        "bring it back",
        "return and land",
        "go back and land"
    ]
}

# ─────────────────────────────────────────────
# SILERO VAD
# ─────────────────────────────────────────────

def load_silero_vad():
    print("Loading Silero VAD...")
    try:
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        print("Silero VAD ready.")
        return model
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Silero VAD: {e}")
        print("Try: pip install torch torchaudio")
        sys.exit(1)


def get_speech_prob(vad_model, audio_float32):
    """Run Silero VAD on a 512-sample window. Returns probability 0.0–1.0."""
    if len(audio_float32) < VAD_WINDOW_SAMPLES:
        padded = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)
        padded[:len(audio_float32)] = audio_float32
        audio_float32 = padded
    tensor = torch.from_numpy(audio_float32[:VAD_WINDOW_SAMPLES])
    with torch.no_grad():
        return vad_model(tensor, SAMPLE_RATE).item()


# ─────────────────────────────────────────────
# DRONE COMMAND EXECUTION
# ─────────────────────────────────────────────

def execute_drone_command(command_type, matched_phrase, score):
    """Placeholder — replace pass blocks with your actual drone API calls."""
    print("\n" + "=" * 52)
    print(f"  [ACTION TRIGGERED] -> Command  : {command_type}")
    print(f"  Matched phrase     : '{matched_phrase}'")
    print(f"  Confidence         : {score:.1f}%")
    print("=" * 52 + "\n")

    if command_type == "ARM":
        pass   # vehicle.armed = True
    elif command_type == "DISARM":
        pass   # vehicle.armed = False
    elif command_type == "TAKEOFF":
        pass   # drone.arm_and_takeoff()
    elif command_type == "RTL":
        pass   # drone.set_mode("RTL")


def process_voice_command(text):
    """Fuzzy-match transcribed text against COMMAND_DICTIONARY using RapidFuzz WRatio."""
    best_command      = None
    best_score        = 0.0
    best_match_phrase = ""

    for command_action, target_phrases in COMMAND_DICTIONARY.items():
        result = process.extractOne(text, target_phrases, scorer=fuzz.WRatio)
        if result:
            matched_phrase, score, _ = result
            if score > best_score:
                best_score        = score
                best_command      = command_action
                best_match_phrase = matched_phrase

    if best_score >= MATCH_THRESHOLD:
        execute_drone_command(best_command, best_match_phrase, best_score)
    else:
        print(f"  -> [No Match] Best attempt: {best_command} at {best_score:.1f}%\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load Silero VAD
    vad_model = load_silero_vad()

    # Load Whisper
    print("Loading Faster-Whisper engine...")
    try:
        whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Whisper engine: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"drone_commands_{timestamp}.txt"

    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("      FUZZY DRONE VOICE COMMAND SYSTEM + SILERO VAD")
    print("=" * 60)
    print(f"  VAD Threshold   : {VAD_THRESHOLD}  (raise to 0.7 for noisy motors)")
    print(f"  Silence Window  : {SILENCE_WINDOW}s")
    print(f"  Min Speech      : {MIN_SPEECH_DURATION}s")
    print(f"  Match Threshold : {MATCH_THRESHOLD}%")
    print(f"  Saving to       : {output_filename}")
    print()
    print("  Try: 'Arm the drone' / 'Cut power' / 'Lift off' / 'Go home'")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

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
        print(f"[CRITICAL ERROR] Could not open microphone: {e}")
        p.terminate()
        sys.exit(1)

    # ── State variables ──
    audio_buffer        = np.zeros(0, dtype=np.float32)  # accumulates speech audio
    vad_chunk_buffer    = np.zeros(0, dtype=np.float32)  # staging for 512-sample windows
    has_spoken          = False      # currently inside a speech segment?
    continuous_silence  = 0.0        # seconds of silence since last speech chunk
    speech_duration     = 0.0        # total speech seconds in current phrase
    last_prob           = 0.0

    try:
        stream.start_stream()
        print("[READY] Listening for drone commands...\n")

        with open(output_filename, "a", encoding="utf-8") as f:
            while stream.is_active():
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                    audio_int16   = np.frombuffer(raw_data, dtype=np.int16)

                    if len(audio_int16) == 0:
                        continue

                    audio_float32  = audio_int16.astype(np.float32) / 32768.0
                    chunk_duration = len(audio_float32) / SAMPLE_RATE

                    # ── Feed into VAD staging buffer ──
                    vad_chunk_buffer = np.concatenate((vad_chunk_buffer, audio_float32))

                    # Run Silero on every 512-sample window available
                    chunk_is_speech = False
                    while len(vad_chunk_buffer) >= VAD_WINDOW_SAMPLES:
                        window          = vad_chunk_buffer[:VAD_WINDOW_SAMPLES]
                        vad_chunk_buffer = vad_chunk_buffer[VAD_WINDOW_SAMPLES:]
                        prob            = get_speech_prob(vad_model, window)
                        last_prob       = prob
                        if prob >= VAD_THRESHOLD:
                            chunk_is_speech = True

                    # ── Update speech / silence state ──
                    if chunk_is_speech:
                        continuous_silence = 0.0
                        has_spoken         = True
                        speech_duration   += chunk_duration
                        audio_buffer       = np.concatenate((audio_buffer, audio_float32))

                        bar = int(last_prob * 20)
                        sys.stdout.write(
                            f"\r🎙  VAD [{'█' * bar}{'░' * (20 - bar)}]"
                            f" {last_prob:.2f} — SPEECH {speech_duration:.1f}s"
                        )
                        sys.stdout.flush()

                    else:
                        if has_spoken:
                            continuous_silence += chunk_duration
                            # Keep buffering to capture trailing phonemes
                            audio_buffer = np.concatenate((audio_buffer, audio_float32))

                        bar = int(last_prob * 20)
                        sys.stdout.write(
                            f"\r⏸   VAD [{'░' * 20}]"
                            f" {last_prob:.2f} — silence {continuous_silence:.1f}s "
                        )
                        sys.stdout.flush()

                    # ── SCENARIO A: Live preview while speaking ──
                    if has_spoken and continuous_silence < SILENCE_WINDOW:
                        if len(audio_buffer) > 0 and \
                                len(audio_buffer) % (SAMPLE_RATE // 2) < CHUNK_SIZE:
                            segments, _ = whisper_model.transcribe(
                                audio_buffer,
                                language="en",
                                beam_size=2,
                                vad_filter=True,
                                condition_on_previous_text=False,
                                temperature=0.0,
                            )
                            preview = " ".join([s.text for s in segments]).strip()
                            if preview:
                                sys.stdout.write(
                                    f"\rListening : {preview:<70}"
                                )
                                sys.stdout.flush()

                    # ── SCENARIO B: Silence detected → finalize ──
                    elif has_spoken and continuous_silence >= SILENCE_WINDOW:

                        if speech_duration < MIN_SPEECH_DURATION:
                            # Too short — likely motor noise burst, skip
                            sys.stdout.write(
                                f"\r[VAD] Ignored burst "
                                f"({speech_duration:.2f}s){' ' * 40}\n"
                            )
                            sys.stdout.flush()

                        else:
                            # Transcribe full utterance
                            segments, _ = whisper_model.transcribe(
                                audio_buffer,
                                language="en",
                                beam_size=3,
                                vad_filter=True,
                                condition_on_previous_text=False,
                                temperature=0.0,
                            )
                            final_text = " ".join([s.text for s in segments]).strip()

                            if final_text:
                                sys.stdout.write("\r" + " " * 100 + "\r")
                                print(f"🗣  Spoken  : {final_text}")

                                # ── FUZZY COMMAND MATCH ──
                                process_voice_command(final_text.lower())

                                # Save to file
                                f.write(
                                    f"[{datetime.now().strftime('%H:%M:%S')}]"
                                    f" {final_text}\n"
                                )
                                f.flush()

                        # ── COLD RESET ──
                        audio_buffer       = np.zeros(0, dtype=np.float32)
                        vad_chunk_buffer   = np.zeros(0, dtype=np.float32)
                        has_spoken         = False
                        continuous_silence = 0.0
                        speech_duration    = 0.0

                        # Flush stale audio from queue
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break

                        print("[READY] Listening for next command...\n")

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
