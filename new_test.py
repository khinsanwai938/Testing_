import queue
import threading
import time

import numpy as np
import pyaudio
import pyttsx3

from test_nlp_brain import DroneNLPBrain
from test_drone_driver import MAVLinkDroneDriver


# ============================================================
# Configuration
# ============================================================

TEST_MODE = False

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

SILENCE_THRESHOLD = 600
SILENCE_WINDOW = 1.2
MAX_AUDIO_SECONDS = 10

CONNECTION_STRING = "udp:127.0.0.1:14551"

MIN_TAKEOFF_ALTITUDE = 2.0
MAX_TAKEOFF_ALTITUDE = 120.0
MIN_VOICE_CHUNKS = 3


# ============================================================
# TTS System — with mic-mute coordination
# ============================================================

tts_queue: queue.Queue = queue.Queue()

# This flag is SET while TTS is speaking.
# The audio processing loop checks it and discards mic input during that window,
# preventing the speaker output from being re-transcribed as a command.
tts_speaking = threading.Event()


def tts_worker():
    """
    Dedicated TTS thread.
    Sets tts_speaking for the duration of each utterance so the
    voice-capture loop knows to discard mic input.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    while True:
        text = tts_queue.get()
        if text is None:
            tts_queue.task_done()
            break
        try:
            print(f"[System Speak] -> {text!r}")
            tts_speaking.set()          # mic mute ON
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            tts_speaking.clear()        # mic mute OFF  (always cleared)
            tts_queue.task_done()


def confirmation_voice(text: str):
    """Queue a line for speech output."""
    tts_queue.put(text)


def wait_for_tts():
    """Block until the TTS queue is fully drained and speaker is silent."""
    tts_queue.join()


# ============================================================
# Audio State
# ============================================================

class AudioState:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.silence_timer: float = 0.0
        self.has_voice_activity: bool = False
        self.voice_chunk_count: int = 0

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.silence_timer = 0.0
        self.has_voice_activity = False
        self.voice_chunk_count = 0


# ============================================================
# Drone Command Execution
# ============================================================

def execute_intent(drone: MAVLinkDroneDriver, brain: DroneNLPBrain,
                   intent: str, text: str):
    """
    Speak confirmation first, then execute.
    Each branch announces what it is about to do BEFORE the blocking
    driver call, so the pilot hears immediate feedback even if the
    maneuver takes several seconds.
    """
    try:
        if intent == "arm":
            confirmation_voice("Arming vehicle.")
            wait_for_tts()
            ok = drone.arm_vehicle()
            confirmation_voice("Vehicle armed." if ok else "Arming failed.")

        elif intent == "disarm":
            confirmation_voice("Disarming vehicle.")
            wait_for_tts()
            ok = drone.disarm_vehicle()
            confirmation_voice("Vehicle disarmed." if ok else "Disarm failed.")

        elif intent == "takeoff":
            altitude = brain.extract_number(text)
            if altitude is None:
                altitude = 5.0
            else:
                altitude = max(MIN_TAKEOFF_ALTITUDE,
                               min(altitude, MAX_TAKEOFF_ALTITUDE))
            confirmation_voice(f"Taking off to {altitude:.0f} metres.")
            wait_for_tts()                          # pilot hears this before motors spin
            if not drone.execute_takeoff(target_altitude=altitude):
                confirmation_voice("Takeoff failed.")

        elif intent == "land":
            confirmation_voice("Landing now.")
            wait_for_tts()
            if not drone.land():
                confirmation_voice("Landing failed.")

        elif intent == "rtl":
            confirmation_voice("Returning to launch.")
            wait_for_tts()
            drone.change_flight_mode("RTL")

        elif intent == "forward":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving forward for {duration:.0f} seconds.")
            wait_for_tts()
            drone.move_forward(duration=duration)

        elif intent == "backward":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving backward for {duration:.0f} seconds.")
            wait_for_tts()
            drone.move_backward(duration=duration)

        elif intent == "left":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving left for {duration:.0f} seconds.")
            wait_for_tts()
            drone.move_left(duration=duration)

        elif intent == "right":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving right for {duration:.0f} seconds.")
            wait_for_tts()
            drone.move_right(duration=duration)

        elif intent == "hover":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 0.0
            msg = f"Hovering for {duration:.0f} seconds." if duration > 0 else "Holding position."
            confirmation_voice(msg)
            wait_for_tts()
            drone.hover(duration=duration)

        elif intent == "ascend":
            metres = brain.extract_number(text) or 3.0
            confirmation_voice(f"Ascending {metres:.0f} metres.")
            wait_for_tts()
            drone.ascend(metres=metres)

        elif intent == "descend":
            metres = brain.extract_number(text) or 3.0
            confirmation_voice(f"Descending {metres:.0f} metres.")
            wait_for_tts()
            drone.descend(metres=metres)

        elif intent == "rotate_left":
            degrees = brain.extract_number(text) or 90.0
            confirmation_voice(f"Rotating left {degrees:.0f} degrees.")
            wait_for_tts()
            drone.rotate_left(degrees=degrees)

        elif intent == "rotate_right":
            degrees = brain.extract_number(text) or 90.0
            confirmation_voice(f"Rotating right {degrees:.0f} degrees.")
            wait_for_tts()
            drone.rotate_right(degrees=degrees)

        else:
            confirmation_voice("Command not recognised.")

        brain.reset_duplicate_guard()

    except Exception as e:
        print(f"[EXECUTION ERROR] {e}")
        confirmation_voice("Command execution failed.")


# ============================================================
# Audio Callback
# ============================================================

def create_audio_callback(audio_queue: queue.Queue):
    def callback(in_data, frame_count, time_info, status):
        if status:
            print(f"[AUDIO WARNING] {status}")
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    return callback


# ============================================================
# Voice Processing
# ============================================================

def process_voice_chunk(raw: bytes, audio_state: AudioState):
    """
    Accumulates one PyAudio chunk.
    Returns a ready numpy buffer when an utterance is complete, else None.
    """
    audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    volume = float(np.sqrt(np.mean(audio_data ** 2)))
    audio_float = audio_data / 32768.0

    if volume > SILENCE_THRESHOLD:
        audio_state.has_voice_activity = True
        audio_state.silence_timer = 0.0
        audio_state.voice_chunk_count += 1
    else:
        if audio_state.has_voice_activity:
            audio_state.silence_timer += len(audio_data) / SAMPLE_RATE

    audio_state.buffer = np.concatenate((audio_state.buffer, audio_float))

    if len(audio_state.buffer) > SAMPLE_RATE * MAX_AUDIO_SECONDS:
        print("[WARNING] Audio buffer reset — exceeded max duration.")
        audio_state.reset()
        return None

    if (audio_state.has_voice_activity
            and audio_state.silence_timer >= SILENCE_WINDOW
            and audio_state.voice_chunk_count >= MIN_VOICE_CHUNKS):
        ready = audio_state.buffer.copy()
        audio_state.reset()
        return ready

    return None


# ============================================================
# Main
# ============================================================

def main():
    print("Initializing NLP brain...")
    brain = DroneNLPBrain()

    print("Connecting to drone...")
    drone = MAVLinkDroneDriver(connection_string=CONNECTION_STRING)

    audio_queue: queue.Queue = queue.Queue()
    audio_state = AudioState()

    last_command_time = 0.0
    command_cooldown = 2.0

    # Drone execution runs on a separate thread so:
    # (a) the mic loop keeps collecting audio while the drone is moving
    # (b) TTS plays immediately without waiting for the maneuver to finish
    execution_thread: threading.Thread | None = None

    stream = None
    p = None

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    if not TEST_MODE:
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=create_audio_callback(audio_queue),
            )
            stream.start_stream()
            print("Voice control active. Speak a command.")
        except Exception as e:
            print(f"[MIC ERROR] {e}")
            confirmation_voice("Microphone initialization failed.")
            raise

    try:
        while True:

            # =================================================
            # TEST MODE
            # =================================================

            if TEST_MODE:
                text = input("Enter command: ").strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit"):
                    break

            # =================================================
            # VOICE MODE
            # =================================================

            else:
                try:
                    raw = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # KEY FIX: discard mic input while TTS is playing.
                # Without this, the speaker output reaches the mic,
                # gets buffered, and is transcribed as a new command.
                if tts_speaking.is_set():
                    audio_state.reset()     # also clear any partial buffer
                    continue

                ready_buffer = process_voice_chunk(raw, audio_state)
                if ready_buffer is None:
                    continue

                try:
                    text = brain.transcribe_audio(ready_buffer, final=True)
                except Exception as e:
                    print(f"[TRANSCRIPTION ERROR] {e}")
                    confirmation_voice("Speech recognition failed.")
                    continue

                if not text:
                    continue

                print(f"[STT] Transcribed: {text!r}")

            # =================================================
            # NLP
            # =================================================

            try:
                intent, confidence = brain.analyze_phrase(text)
            except Exception as e:
                print(f"[NLP ERROR] {e}")
                confirmation_voice("Command analysis failed.")
                continue

            if intent == "invalid_command":
                print(f"[NLP] No match (confidence {confidence:.2f}) for: {text!r}")
                confirmation_voice("Command not recognised.")
                continue

            # =================================================
            # Cooldown
            # =================================================

            now = time.time()
            if now - last_command_time < command_cooldown:
                print("[INFO] Cooldown active — command ignored.")
                continue

            # Block new commands while the previous maneuver is still running.
            # e.g. don't accept "move left" while "move forward 10 seconds" is active.
            if execution_thread and execution_thread.is_alive():
                print("[INFO] Drone busy — command ignored.")
                confirmation_voice("Busy. Please wait.")
                continue

            last_command_time = now
            print(f"[NLP] Intent: {intent.upper()} | Confidence: {confidence:.2f}")

            # Run execution on its own thread so the mic loop is never blocked
            execution_thread = threading.Thread(
                target=execute_intent,
                args=(drone, brain, intent, text),
                daemon=True,
            )
            execution_thread.start()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Wait for any running maneuver to finish cleanly
        if execution_thread and execution_thread.is_alive():
            print("[INFO] Waiting for current maneuver to complete...")
            execution_thread.join(timeout=15)

        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"[STREAM ERROR] {e}")

        if p is not None:
            try:
                p.terminate()
            except Exception as e:
                print(f"[PYAUDIO ERROR] {e}")

        tts_queue.put(None)
        tts_thread.join(timeout=5)

        try:
            drone.close_connection()
        except Exception as e:
            print(f"[DRONE CLOSE ERROR] {e}")

        print("System shutdown complete.")


if __name__ == "__main__":
    main()