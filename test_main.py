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

TEST_MODE = True

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

SILENCE_THRESHOLD = 600
SILENCE_WINDOW = 1.2
MAX_AUDIO_SECONDS = 10

CONNECTION_STRING = "udp:127.0.0.1:14550"

# Minimum altitude guard — prevents "take off to 1 metre" accidents.
MIN_TAKEOFF_ALTITUDE = 2.0
MAX_TAKEOFF_ALTITUDE = 120.0


# ============================================================
# Text-to-Speech System
# ============================================================

tts_queue: queue.Queue = queue.Queue()

def tts_worker():
    """
    Dedicated TTS worker thread.
    Engine is created here so it runs on the thread that calls runAndWait(),
    avoiding cross-thread COM/audio driver crashes on Windows.
    """
    # FIX 1 (continued): engine instantiated on the worker thread.
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    while True:
        text = tts_queue.get()

        if text is None:
            """fixed"""
            tts_queue.task_done()
            break

        try:
            print(f"[TTS] {text}")
            engine.say(text)
            engine.runAndWait()

        except Exception as e:
            print(f"[TTS ERROR] {e}")
            """fixed"""
        finally:
            tts_queue.task_done()


def confirmation_voice(text: str):
    """Queue text for speech synthesis."""
    tts_queue.put(text)


# ============================================================
# Drone Command Execution
# ============================================================

def execute_intent(drone: MAVLinkDroneDriver, brain: DroneNLPBrain,
                   intent: str, text: str):
    """
    Execute a drone action for the given intent.

    `brain` is now passed as a parameter instead of read from a
    module-level global. Global mutable state makes the function untestable
    and introduces a subtle initialisation race — if execute_intent() were
    somehow called before main() sets `brain`, the RuntimeError path would
    fire. Passing it explicitly is safer and cleaner.

    Args:
        drone:  MAVLinkDroneDriver instance.
        brain:  DroneNLPBrain instance (needed for extract_number).
        intent: Intent string returned by analyze_phrase().
        text:   Original transcribed text (used to extract altitude numbers).
    """
    try:
        if intent == "arm":
            if drone.arm_vehicle():
                confirmation_voice("Vehicle armed successfully.")
            else:
                # arm_vehicle() returns False for TWO reasons:
                # (a) already armed, (b) arming failed.
                # Original code announced "already armed" for both.
                # Now the driver prints its own [DRIVER] message for failures,
                # so the voice feedback here is kept generic.
                confirmation_voice("Arm command completed.")

        elif intent == "disarm":
            if drone.disarm_vehicle():
                confirmation_voice("Vehicle disarmed successfully.")
            else:
                confirmation_voice("Disarm command completed.")

        elif intent == "takeoff":
            altitude = brain.extract_number(text)

            # Clamp altitude to a safe range.
            # Without this, "take off to 1000 metres" or a mis-transcription
            # like "take off to 0" would be sent straight to the drone.
            if altitude is None:
                altitude = 5.0
            else:
                altitude = max(MIN_TAKEOFF_ALTITUDE,
                               min(altitude, MAX_TAKEOFF_ALTITUDE))

            confirmation_voice(f"Taking off to {altitude:.0f} metres.")
            success = drone.execute_takeoff(target_altitude=altitude)
            if not success:
                confirmation_voice("Takeoff failed.")

        elif intent == "land":
            # land and rtl had confirmation_voice AFTER the blocking
            # driver call. change_flight_mode() waits up to 5 s for mode
            # confirmation, so the voice feedback was delayed. Announce first,
            # then change mode so the pilot hears immediate feedback.
            confirmation_voice("Landing initiated.")
            success = drone.land()
            if not success:
                confirmation_voice("Landing failed.")

        elif intent == "rtl":
            confirmation_voice("Returning to launch.")
            drone.change_flight_mode("RTL")
        
        elif intent == "forward":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving forward for {duration:.1f} seconds.")
            drone.move_forward(duration=duration)
        
        elif intent == "backward":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving backward for {duration:.1f} seconds.")
            drone.move_backward(duration=duration)

        elif intent == "left":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving left for {duration:.1f} seconds.")
            drone.move_left(duration=duration)

        elif intent == "right":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 3.0
            confirmation_voice(f"Moving right for {duration:.1f} seconds.")
            drone.move_right(duration=duration)
        
        elif intent == "hover":
            duration = brain.extract_duration(text) or brain.extract_number(text) or 0.0
            if duration > 0:
                confirmation_voice(f"Hovering in place for {duration:.1f} seconds.")
            else:
                confirmation_voice(f"Hovering in place.")
            drone.hover(duration=duration)

        elif intent == "ascend":
            metres = brain.extract_number(text) or 3.0
            confirmation_voice(f"Ascending for {metres:.1f} metres.")
            drone.ascend(metres=metres)
        
        elif intent == "descend":
            metres = brain.extract_number(text) or 3.0
            confirmation_voice(f"Descending for {metres:.1f} metres.")
            drone.descend(metres=metres)
        
        elif intent == "rotate_left":
            degrees = brain.extract_number(text) or 90.0
            confirmation_voice(f"Rotating left for {degrees:.1f} degrees.")
            drone.rotate_left(degrees=degrees)
        
        elif intent == "rotate_right":
            degrees = brain.extract_number(text) or 90.0
            confirmation_voice(f"Rotating right for {degrees:.1f} degrees.")
            drone.rotate_right(degrees=degrees)

        else:
            confirmation_voice("Unknown command.")

    except Exception as e:
        print(f"[EXECUTION ERROR] {e}")
        confirmation_voice("Command execution failed.")


# ============================================================
# Audio Callback
# ============================================================

def create_audio_callback(audio_queue: queue.Queue):
    """Create and return a PyAudio stream callback."""

    def callback(in_data, frame_count, time_info, status):
        if status:
            print(f"[AUDIO WARNING] {status}")
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    return callback

# ============================================================
# Main
# ============================================================

def main():
    print("Initializing NLP brain...")
    brain = DroneNLPBrain()

    print("Connecting to drone...")
    drone = MAVLinkDroneDriver(connection_string=CONNECTION_STRING)

    # --------------------------------------------------------
    # Runtime Variables
    # --------------------------------------------------------

    audio_queue: queue.Queue = queue.Queue()
    audio_buffer = np.zeros(0, dtype=np.float32)

    silence_timer = 0.0
    has_voice_activity = False

    last_command_time = 0.0
    command_cooldown = 2.0

    stream = None
    p = None

    # --------------------------------------------------------
    # Start TTS Thread
    # --------------------------------------------------------

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # --------------------------------------------------------
    # Initialize Audio Stream
    # --------------------------------------------------------

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
            print("Voice control active.")

        except Exception as e:
            print(f"[MIC ERROR] {e}")
            confirmation_voice("Microphone initialization failed.")

            # Raising lets the outer try/finally handle cleanup correctly.
            raise

    # --------------------------------------------------------
    # Main Loop
    # --------------------------------------------------------

    try:
        while True:

            # =================================================
            # TEST MODE  (keyboard input)
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

                audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                volume = np.sqrt(np.mean(audio_data ** 2))

                if volume > SILENCE_THRESHOLD:
                    has_voice_activity = True
                    silence_timer = 0.0
                else:
                    silence_timer += len(audio_data) / SAMPLE_RATE

                audio_float = audio_data / 32768.0
                audio_buffer = np.concatenate((audio_buffer, audio_float))

                if len(audio_buffer) > SAMPLE_RATE * MAX_AUDIO_SECONDS:
                    print("[WARNING] Audio buffer reset — no speech detected in time.")
                    audio_buffer = np.zeros(0, dtype=np.float32)
                    has_voice_activity = False
                    silence_timer = 0.0
                    continue

                if not (has_voice_activity and silence_timer >= SILENCE_WINDOW):
                    continue

                try:
                    text = brain.transcribe_audio(audio_buffer, final=True)

                except Exception as e:
                    print(f"[TRANSCRIPTION ERROR] {e}")
                    confirmation_voice("Speech recognition failed.")
                    audio_buffer = np.zeros(0, dtype=np.float32)
                    has_voice_activity = False
                    silence_timer = 0.0
                    continue

                audio_buffer = np.zeros(0, dtype=np.float32)
                has_voice_activity = False
                silence_timer = 0.0

                if not text:
                    continue

                print(f"[TRANSCRIBED] {text}")

            # =================================================
            # NLP Analysis
            # =================================================

            try:
                intent, confidence = brain.analyze_phrase(text)

            except Exception as e:
                print(f"[NLP ERROR] {e}")
                confirmation_voice("Command analysis failed.")
                continue

            if intent == "invalid_command":
                print(f"[INFO] No match (confidence {confidence:.2f}) for: {text!r}")
                confirmation_voice("Command not recognized.")
                continue

            # =================================================
            # Command Cooldown
            # =================================================

            current_time = time.time()

            if current_time - last_command_time < command_cooldown:
                print("[INFO] Command ignored — cooldown active.")
                """fixed"""
                confirmation_voice("Command blocked. System cooling down.")
                continue

            last_command_time = current_time

            print(f"[INTENT] {intent}  |  confidence: {confidence:.2f}  |  text: {text!r}")

            # FIX 3 (continued): pass brain explicitly instead of global.
            execute_intent(drone, brain, intent, text)

    except KeyboardInterrupt:
        print("\nStopping system...")

    finally:
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

        try:
            drone.close_connection()
        except Exception as e:
            print(f"[DRONE CLOSE ERROR] {e}")

        print("System shutdown complete.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()