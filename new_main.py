import os
import sys
import time
import queue
import threading
import pyaudio
import pyttsx3
import numpy as np
from datetime import datetime

from llm_con import DroneNLPEngine
from driver import MAVLinkDroneDriver

TEST_MODE = True
is_tts_talking = False  # Global flag to block microphone during TTS playback


# ======================================================================
# Text-to-speech
# ======================================================================

def voice_reply(text: str):
    global is_tts_talking
    print(f"[SYSTEM SPEAK] -> \"{text}\"")

    def _speak():
        global is_tts_talking
        try:
            is_tts_talking = True
            time.sleep(0.1)          # Audio stream sync buffer
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            del engine
            time.sleep(0.4)          # Room echo cool-down
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            is_tts_talking = False

    threading.Thread(target=_speak, daemon=True).start()


# ======================================================================
# Logging
# ======================================================================

def log_event(raw_text: str, intent: str, confidence: float, outcome: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("voice_flight_commands.log", "a") as f:
        f.write(
            f"[{timestamp}] TEXT: \"{raw_text}\" | "
            f"INTENT: {intent.upper()} | "
            f"CONF: {confidence:.2f} | "
            f"STATUS: {outcome}\n"
        )


# ======================================================================
# Microphone calibration
# ======================================================================

def calibrate_silence_threshold(p, sample_rate=16000, chunk_size=1024,
                                 duration=1.5) -> float:
    print("[CALIBRATION] Measuring ambient noise floor...")
    print("[CALIBRATION] Please stay silent for 1.5 seconds...")
    stream = p.open(
        format=pyaudio.paInt16, channels=1,
        rate=sample_rate, input=True,
        frames_per_buffer=chunk_size,
    )
    frames = []
    for _ in range(int(sample_rate / chunk_size * duration)):
        raw = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(np.frombuffer(raw, dtype=np.int16))
    stream.stop_stream()
    stream.close()

    all_samples = np.concatenate(frames).astype(np.float64)
    noise_rms   = np.sqrt(np.mean(all_samples ** 2))
    threshold   = float(np.clip(noise_rms * 1.8, 300, 2000))
    print(f"[CALIBRATION] Noise RMS: {noise_rms:.1f} → Threshold: {threshold:.1f}")
    return threshold


# ======================================================================
# Main pipeline
# ======================================================================

def run_central_pipeline():
    os.system('cls' if os.name == 'nt' else 'clear')

    brain = DroneNLPEngine()
    drone = MAVLinkDroneDriver(connection_string="udp:127.0.0.1:14551")

    print("\n=== VOX-FLIGHT SYSTEM INTEGRATION PIPELINE ACTIVE ===")

    SAMPLE_RATE         = 16000
    CHUNK_SIZE          = 1024
    SILENCE_WINDOW      = 0.8          # seconds of silence before processing
    MAX_BUFFER_SAMPLES  = SAMPLE_RATE * 6
    MIN_BUFFER_SAMPLES  = int(SAMPLE_RATE * 0.4)

    audio_queue  = queue.Queue()
    audio_buffer = np.zeros(0, dtype=np.float32)
    silence_time = 0.0
    has_spoken   = False

    def stream_callback(in_data, frame_count, time_info, status):
        # Drop microphone data while TTS is playing to prevent echo loops
        if is_tts_talking:
            audio_queue.put(b'\x00' * len(in_data))
        else:
            audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    # ── Audio setup ───────────────────────────────────────────────────
    if not TEST_MODE:
        p = pyaudio.PyAudio()
        SILENCE_THRESHOLD = calibrate_silence_threshold(p, SAMPLE_RATE, CHUNK_SIZE)
        try:
            stream = p.open(
                format=pyaudio.paInt16, channels=1,
                rate=SAMPLE_RATE, input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=stream_callback,
            )
            stream.start_stream()
        except Exception as e:
            print(f"[CRITICAL] Could not open microphone: {e}")
            p.terminate()
            sys.exit(1)
        print(">> RUNNING IN LIVE VOICE MODE <<")
        voice_reply("Voice flight pipeline active. Ready for commands.")
    else:
        SILENCE_THRESHOLD = 600
        print(">> RUNNING IN KEYBOARD TEST MODE <<")

    # ── Main loop ─────────────────────────────────────────────────────
    while True:
        try:
            captured_text = ""

            # ── Input: keyboard (TEST) or microphone (LIVE) ───────────
            if TEST_MODE:
                print("\n[ TEST MODE ]")
                captured_text = input("Enter command: ").lower().strip()
                if not captured_text:
                    continue

            else:
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Live telemetry display while waiting for voice input
                    alt = drone.get_altitude()
                    if alt is not None and drone.is_flying():
                        state = drone.get_safety_state()
                        sys.stdout.write(
                            f"\r[Telemetry] Alt: {alt:.2f} m | State: {state}    "
                        )
                        sys.stdout.flush()
                    continue

                audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                if len(audio_int16) == 0:
                    continue

                volume = (
                    np.sqrt(np.mean(audio_int16.astype(np.float64) ** 2))
                    if np.mean(audio_int16 ** 2) > 0 else 0.0
                )
                chunk_dur = len(audio_int16) / SAMPLE_RATE

                if volume >= SILENCE_THRESHOLD:
                    silence_time = 0.0
                    has_spoken   = True
                else:
                    silence_time += chunk_dur

                if has_spoken:
                    audio_buffer = np.concatenate(
                        (audio_buffer, audio_int16.astype(np.float32) / 32768.0)
                    )

                if len(audio_buffer) > MAX_BUFFER_SAMPLES:
                    audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]

                if has_spoken and silence_time >= SILENCE_WINDOW:
                    if len(audio_buffer) < MIN_BUFFER_SAMPLES:
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        has_spoken   = False
                        silence_time = 0.0
                        continue

                    print("\n[STT] Processing utterance...")
                    captured_text = brain.transcribe_audio(audio_buffer)
                    audio_buffer  = np.zeros(0, dtype=np.float32)
                    has_spoken    = False
                    silence_time  = 0.0

                    # Flush stale audio accumulated during STT processing
                    while not audio_queue.empty():
                        audio_queue.get_nowait()

                    if not captured_text:
                        print("[STT] No speech detected — skipping.")
                        continue
                    print(f'[STT] Transcribed: "{captured_text}"')

            # ── Intent dispatch ───────────────────────────────────────
            if not captured_text:
                continue

            intent, confidence = brain.match_intent(captured_text)

            # Flush any audio junk that built up during NLP processing
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            if not intent or intent == "invalid_command":
                log_event(captured_text, "REJECTED", confidence, "USER_ALERTED_INVALID")
                voice_reply("Command not recognized. Please try again.")
                continue

            # ── Emergency (highest priority — always allowed) ─────────
            if intent == "emergency_safe":
                action = drone.trigger_emergency_safe_state()
                log_event(captured_text, intent, confidence, f"EMERGENCY_{action}")
                voice_reply(
                    "Emergency safe state active. Returning home."
                    if action == "RTL_ACTIVATED"
                    else "Emergency safe state active. Landing immediately."
                )
                continue

            # ── Ground commands (allowed before takeoff) ──────────────
            elif intent == "arm":
                if drone.arm_vehicle():
                    log_event(captured_text, intent, confidence, "MOTORS_ARMED")
                    voice_reply("Arming aircraft propulsion motors.")
                else:
                    voice_reply("The aircraft is already armed.")

            elif intent == "disarm":
                if drone.disarm_vehicle():
                    log_event(captured_text, intent, confidence, "MOTORS_DISARMED")
                    voice_reply("Disarming propulsion systems safely.")
                else:
                    voice_reply("Disarm command failed. Verify vehicle safety locks.")

            elif intent == "takeoff":
                value      = brain.extract_number(captured_text)
                target_alt = float(value) if value else 5.0
                voice_reply(
                    f"Taking off to {int(target_alt)} meters. "
                    f"Commands locked until altitude is reached."
                )
                if drone.execute_takeoff(target_alt):
                    log_event(captured_text, intent, confidence,
                              f"TAKEOFF_COMPLETE_ALT_{target_alt}")
                else:
                    voice_reply("Takeoff sequence aborted.")
                    log_event(captured_text, intent, confidence, "TAKEOFF_FAILED")

            # ── In-flight commands (gated by altitude safety state) ────
            elif intent == "land":
                if drone.land():
                    log_event(captured_text, intent, confidence, "LANDED")
                    voice_reply("Landed successfully.")
                else:
                    voice_reply("Landing timed out. Check vehicle status.")

            elif intent == "rtl":
                if drone.return_to_launch():
                    log_event(captured_text, intent, confidence, "MODE_RTL")
                    voice_reply("Returning to home base.")
                else:
                    voice_reply("Return to launch failed.")

            elif intent == "loiter":
                if drone.set_loiter():
                    log_event(captured_text, intent, confidence, "MODE_LOITER")
                    voice_reply("Position hold engaged. Loitering.")
                else:
                    voice_reply("Loiter command blocked. Altitude gate still active.")

            elif intent == "position_hold":
                if drone.set_position_hold():
                    log_event(captured_text, intent, confidence, "MODE_POSHOLD")
                    voice_reply("Position hold mode active.")
                else:
                    voice_reply("Position hold command blocked.")

            elif intent == "hover":
                drone.hover()
                log_event(captured_text, intent, confidence, "HOVER")
                voice_reply("Hovering. All movement stopped.")

            elif intent == "move_forward":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.move_forward(speed=value):
                    log_event(captured_text, intent, confidence, f"MOVE_FWD_{value}")
                    voice_reply(f"Moving forward at {value:.0f} metres per second.")
                else:
                    voice_reply("Move forward blocked. Altitude gate still active.")

            elif intent == "move_backward":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.move_backward(speed=value):
                    log_event(captured_text, intent, confidence, f"MOVE_BWD_{value}")
                    voice_reply(f"Moving backward at {value:.0f} metres per second.")
                else:
                    voice_reply("Move backward blocked. Altitude gate still active.")

            elif intent == "move_left":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.move_left(speed=value):
                    log_event(captured_text, intent, confidence, f"MOVE_LEFT_{value}")
                    voice_reply(f"Moving left at {value:.0f} metres per second.")
                else:
                    voice_reply("Move left blocked. Altitude gate still active.")

            elif intent == "move_right":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.move_right(speed=value):
                    log_event(captured_text, intent, confidence, f"MOVE_RIGHT_{value}")
                    voice_reply(f"Moving right at {value:.0f} metres per second.")
                else:
                    voice_reply("Move right blocked. Altitude gate still active.")

            elif intent == "ascend":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.ascend(metres=value):
                    log_event(captured_text, intent, confidence, f"ASCEND_{value}m")
                    voice_reply(f"Ascending {value:.0f} metres.")
                else:
                    voice_reply("Ascend command blocked or failed.")

            elif intent == "descend":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.descend(metres=value):
                    log_event(captured_text, intent, confidence, f"DESCEND_{value}m")
                    voice_reply(f"Descending {value:.0f} metres.")
                else:
                    voice_reply("Descend command blocked or failed.")

            elif intent == "rotate_left":
                value = float(brain.extract_number(captured_text) or 90.0)
                if drone.rotate_left(degrees=value):
                    log_event(captured_text, intent, confidence, f"ROTATE_LEFT_{value}deg")
                    voice_reply(f"Rotating left {value:.0f} degrees.")
                else:
                    voice_reply("Rotate left blocked. Altitude gate still active.")

            elif intent == "rotate_right":
                value = float(brain.extract_number(captured_text) or 90.0)
                if drone.rotate_right(degrees=value):
                    log_event(captured_text, intent, confidence, f"ROTATE_RIGHT_{value}deg")
                    voice_reply(f"Rotating right {value:.0f} degrees.")
                else:
                    voice_reply("Rotate right blocked. Altitude gate still active.")

            elif intent == "set_heading":
                value = brain.extract_number(captured_text)
                if value is not None:
                    if drone.set_heading(float(value)):
                        log_event(captured_text, intent, confidence, f"HEADING_{value}deg")
                        voice_reply(f"Heading set to {int(value)} degrees.")
                    else:
                        voice_reply("Heading command blocked. Altitude gate still active.")
                else:
                    voice_reply("Please specify a heading in degrees.")

            elif intent == "set_speed":
                value = float(brain.extract_number(captured_text) or 5.0)
                if drone.set_groundspeed(value):
                    log_event(captured_text, intent, confidence, f"SPEED_{value}ms")
                    voice_reply(f"Ground speed set to {value:.0f} metres per second.")
                else:
                    voice_reply("Speed change failed.")

            # ── Status queries ────────────────────────────────────────
            elif intent == "get_altitude":
                alt = drone.get_altitude()
                if alt is not None:
                    log_event(captured_text, intent, confidence, f"ALT_QUERY_{alt:.1f}m")
                    voice_reply(f"Current altitude is {alt:.1f} metres.")
                else:
                    voice_reply("Altitude data unavailable.")

            elif intent == "get_heading":
                hdg = drone.get_heading()
                if hdg is not None:
                    log_event(captured_text, intent, confidence, f"HDG_QUERY_{hdg:.0f}deg")
                    voice_reply(f"Current heading is {hdg:.0f} degrees.")
                else:
                    voice_reply("Heading data unavailable.")

            elif intent == "get_battery":
                batt = drone.get_battery()
                if batt and batt.get("remaining_pct", -1) != -1:
                    pct = batt["remaining_pct"]
                    log_event(captured_text, intent, confidence, f"BATT_QUERY_{pct}pct")
                    voice_reply(f"Battery is at {pct} percent.")
                else:
                    voice_reply("Battery data unavailable.")

            elif intent == "get_location":
                loc = drone.get_location()
                if loc:
                    log_event(captured_text, intent, confidence, "LOC_QUERY")
                    voice_reply(
                        f"Current position: latitude {loc['lat']:.4f}, "
                        f"longitude {loc['lon']:.4f}, "
                        f"altitude {loc['alt_relative']:.1f} metres."
                    )
                else:
                    voice_reply("Location data unavailable.")

            elif intent == "get_mode":
                mode = drone.get_current_mode()
                if mode:
                    log_event(captured_text, intent, confidence, f"MODE_QUERY_{mode}")
                    voice_reply(f"Current flight mode is {mode}.")
                else:
                    voice_reply("Flight mode data unavailable.")

            elif intent == "get_gps":
                gps = drone.get_gps_status()
                if gps:
                    log_event(captured_text, intent, confidence,
                              f"GPS_QUERY_{gps['satellites_visible']}sats")
                    voice_reply(
                        f"GPS fix type {gps['fix_type']}, "
                        f"{gps['satellites_visible']} satellites visible."
                    )
                else:
                    voice_reply("GPS status unavailable.")

            else:
                log_event(captured_text, intent, confidence, "UNHANDLED_INTENT")
                voice_reply(f"Command understood as {intent} but no action is mapped.")

        except KeyboardInterrupt:
            print("\n[SYSTEM] Keyboard interrupt — shutting down cleanly...")
            break
        except Exception as e:
            print(f"[ERROR] Pipeline runtime error: {e}")

    # ── Cleanup ───────────────────────────────────────────────────────
    if not TEST_MODE:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

    drone.close_connection()
    print("[SYSTEM] Pipeline terminated cleanly.")


if __name__ == "__main__":
    run_central_pipeline()