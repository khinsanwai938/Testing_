import os
import sys
import time
import queue
import threading
import pyaudio
import pyttsx3
import numpy as np
from datetime import datetime

from nlp_brain import DroneNLPBrain
from drone_driver import MAVLinkDroneDriver

# --- Global Configurations ---
TEST_MODE = False
is_tts_talking = False  # Global safety watchdog flag to gate microphone input

def voice_reply(text):
    """Triggers non-blocking text-to-speech output in a background thread."""
    global is_tts_talking
    print(f"[System Speak] -> \"{text}\"")
    
    def target_speak():
        global is_tts_talking
        try:
            is_tts_talking = True  # Signal that the speaker is active
            time.sleep(0.1)        # Small buffer for audio stream synchronization
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            
            # Explicitly stop the event loop before deleting to prevent threading leak
            try:
                engine.endLoop()
            except:
                pass
            del engine
            
            time.sleep(0.4)        # Cool-down window to let acoustic room echo dissipate
        except Exception as e:
            print(f"[TTS ERROR] Background speech failed: {e}")
        finally:
            is_tts_talking = False # Re-open microphone gate safely

    threading.Thread(target=target_speak, daemon=True).start()


def log_event(raw_text, intent, confidence, outcome):
    """Maintains an append-only system execution file log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("voice_flight_commands.log", "a") as f:
        f.write(
            f"[{timestamp}] TEXT: \"{raw_text}\" | INTENT: {intent.upper()} "
            f"| CONF: {confidence:.2f} | STATUS: {outcome}\n"
        )


def calibrate_silence_threshold(p, sample_rate=16000, chunk_size=1024, duration=1):
    """Measures environmental acoustic variations to establish a baseline VAD limit."""
    print("[CALIBRATION] Measuring your ambient noise floor...")
    
    stream = p.open(
        format=pyaudio.paInt16, channels=1,
        rate=sample_rate, input=True,
        frames_per_buffer=chunk_size
    )

    frames = []
    num_chunks = int(sample_rate / chunk_size * duration)
    for _ in range(num_chunks):
        raw = stream.read(chunk_size, exception_on_overflow=False) 
        frames.append(np.frombuffer(raw, dtype=np.int16))

    stream.stop_stream()
    stream.close()

    all_samples = np.concatenate(frames).astype(np.float64)
    noise_rms = np.sqrt(np.mean(all_samples ** 2)) if len(all_samples) > 0 else 0.0
    threshold = float(np.clip(noise_rms * 1.8, 300, 2000))

    print(f"[CALIBRATION] Noise floor RMS: {noise_rms:.1f} → Silence threshold: {threshold:.1f}")
    return threshold


def execute_drone_command(captured_text: str, brain: DroneNLPBrain, drone: MAVLinkDroneDriver) -> None:
    """
    Parses a recognized textual command, matches it against flight intents,
    and runs specific MAVLink/Drone actions.
    """
    if not captured_text:
        return

    intent, confidence = brain.match_intent(captured_text)

    if intent == "invalid_command":
        log_event(captured_text, "REJECTED", confidence, "USER_ALERTED_INVALID")
        voice_reply("Command not recognized. Please try again.")
        return

    # --- System State Adjustments ---
    if intent == "arm":
        if drone.arm_vehicle():
            log_event(captured_text, intent, confidence, "MOTORS_ARMED")
            voice_reply("Arming aircraft propulsion motors.")
        else:
            voice_reply("The aircraft failed safety checks or is already armed.")

    elif intent == "disarm":
        if drone.disarm_vehicle():
            log_event(captured_text, intent, confidence, "MOTORS_DISARMED")
            voice_reply("Disarming propulsion systems safely.")
        else:
            voice_reply("Disarm command failed. Verify vehicle safety locks.")

    elif intent == "takeoff":
        value = brain.extract_number(captured_text)
        target_alt = value if value else 5.0
        voice_reply(f"Taking off to target altitude of {int(target_alt)} meters.")
        if drone.execute_takeoff(target_alt):
            log_event(captured_text, intent, confidence, f"TAKEOFF_ACTIVE_ALT_{target_alt}")
        else:
            voice_reply("Takeoff sequence aborted.")

    # --- Standard Flight Modes ---
    elif intent == "rtl":
        drone.change_mode("GUIDED")
        time.sleep(0.1)
        drone.change_mode("RTL")
        log_event(captured_text, intent, confidence, "MODE_RTL")
        voice_reply("Command verified. Returning back to home base.")

    elif intent == "loiter":
        drone.change_mode("GUIDED")
        time.sleep(0.1)
        drone.change_mode("LOITER")
        log_event(captured_text, intent, confidence, "MODE_LOITER")
        voice_reply("Position hold engaged. Loitering.")

    elif intent == "land":
        drone.change_mode("LAND")
        log_event(captured_text, intent, confidence, "MODE_LAND")
        voice_reply("Landing immediately.")

    # --- Translational Movements (Requires Airborne Safety Check) ---
    else:
        if not drone.is_flying():
            log_event(captured_text, intent, confidence, "REJECTED_NOT_IN_AIR")
            voice_reply("Safety reject. The drone must be airborne first.")
            return

        value = brain.extract_number(captured_text) if brain.extract_number(captured_text) else 5.0

        if intent == "move_forward":
            drone.send_body_translation(value, 0, 0)
            voice_reply(f"Moving forward {int(value)} meters.")
        elif intent == "move_backward":
            drone.send_body_translation(-value, 0, 0)
            voice_reply(f"Moving backward {int(value)} meters.")
        elif intent == "move_left":
            drone.send_body_translation(0, -value, 0)
            voice_reply(f"Moving left {int(value)} meters.")
        elif intent == "move_right":
            drone.send_body_translation(0, value, 0)
            voice_reply(f"Moving right {int(value)} meters.")

        log_event(captured_text, intent, confidence, f"MOVE_EXEC_VAL_{value}")


def start_voice_flight_system():
    """
    Main orchestration loop. Collects incoming microphone stream vectors or 
    keyboard characters and handles real-time Voice Activity Detection (VAD).
    """
    os.system('cls' if os.name == 'nt' else 'clear')

    brain = DroneNLPBrain()
    # Connect driver to standard software-in-the-loop firmware endpoint
    drone = MAVLinkDroneDriver(connection_string='127.0.0.1:14552')

    print("\n=== VOX-FLIGHT SYSTEM INTEGRATION PIPELINE ACTIVE ===")

    SAMPLE_RATE = 16000
    CHUNK_SIZE  = 1024
    SILENCE_WINDOW = 0.8
    MAX_BUFFER_SAMPLES = SAMPLE_RATE * 6  
    MIN_BUFFER_SAMPLES = int(SAMPLE_RATE * 0.4)

    audio_queue  = queue.Queue()
    audio_buffer = np.zeros(0, dtype=np.float32)
    silence_time = 0.0
    has_spoken   = False

    def stream_callback(in_data, frame_count, time_info, status):
        # Insert zero-padded silence if the drone is talking to stop echo looping
        if is_tts_talking:
            audio_queue.put(b'\x00' * len(in_data)) 
        else:
            audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    if not TEST_MODE:
        p = pyaudio.PyAudio()
        SILENCE_THRESHOLD = calibrate_silence_threshold(p)

        try:
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE,
                stream_callback=stream_callback
            )
            stream.start_stream()
        except Exception as e:
            print(f"[CRITICAL ERROR] Could not bind microphone interface: {e}")
            p.terminate()
            sys.exit(1)
        print(">> RUNNING IN LIVE VOICE MODE <<")
        voice_reply("Voice flight pipeline active. Ready for hardware controls.")
    else:
        SILENCE_THRESHOLD = 600
        print(">> RUNNING IN KEYBOARD TEST MODE <<")

    while True:
        try:
            captured_text = ""

            if TEST_MODE:
                print("\n[ TEST MODE ]")
                captured_text = input("Enter command: ").lower().strip()
                if not captured_text: 
                    continue
            else:
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Abstracted safe backend status checking
                    if drone.is_flying():
                        alt_val = drone._get_altitude() if hasattr(drone, '_get_altitude') else 0.0
                        sys.stdout.write(f"\r[Telemetry Update] Altitude: {alt_val:.2f}m")
                        sys.stdout.flush()
                    continue

                audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                if len(audio_int16) == 0: 
                    continue

                volume = np.sqrt(np.mean(audio_int16.astype(np.float64) ** 2)) if np.mean(audio_int16**2) > 0 else 0.0
                chunk_dur = len(audio_int16) / SAMPLE_RATE

                if volume >= SILENCE_THRESHOLD:
                    silence_time = 0.0
                    has_spoken = True
                else:
                    silence_time += chunk_dur

                if has_spoken:
                    audio_buffer = np.concatenate((audio_buffer, audio_int16.astype(np.float32) / 32768.0))

                if len(audio_buffer) > MAX_BUFFER_SAMPLES:
                    audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]

                if has_spoken and silence_time >= SILENCE_WINDOW:
                    if len(audio_buffer) < MIN_BUFFER_SAMPLES:
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        has_spoken = False
                        silence_time = 0.0
                        continue

                    print("\n[STT] Processing utterance...")
                    captured_text = brain.transcribe_audio(audio_buffer, final=True)
                    
                    # Reset audio parameters for next command cycle
                    audio_buffer = np.zeros(0, dtype=np.float32)
                    has_spoken = False
                    silence_time = 0.0
                    
                    while not audio_queue.empty():
                        audio_queue.get_nowait()

                    if not captured_text:
                        print("[STT] No speech detected — skipping.")
                        continue

                    print(f'[STT] Transcribed: "{captured_text}"')

            # --- Execution Dispatcher ---
            if captured_text:
                execute_drone_command(captured_text, brain, drone)
                
                # Flush trailing microphone frame garbage generated during voice command delays
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break

        except KeyboardInterrupt:
            print("\nShutting down safety systems cleanly...")
            break
        except Exception as e:
            print(f"[ERROR] Pipeline runtime error: {e}")

    # --- Resource Cleanup Phase ---
    if not TEST_MODE:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

    drone.close()
    print("[SYSTEM] Pipeline terminated cleanly.")


if __name__ == "__main__":
    start_voice_flight_system()