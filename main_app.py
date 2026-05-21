import os
import time
from datetime import datetime
import speech_recognition as sr
import pyttsx3

# Custom imports from your newly separated modules!
from nlp_brain import DroneNLPBrain
from drone_driver import MAVLinkDroneDriver

# Initialize Feedback voice engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)

def voice_reply(text):
    print(f"[System Speak] -> \"{text}\"")
    tts_engine.say(text)
    tts_engine.runAndWait()

def log_event(raw_text, intent, confidence, outcome):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("voice_flight_commands.log", "a") as f:
        f.write(f"[{timestamp}] TEXT: \"{raw_text}\" | INTENT: {intent.upper()} | CONF: {confidence:.2f} | STATUS: {outcome}\n")

def run_central_pipeline():
    # Instantiate modules
    brain = DroneNLPBrain()
    drone = MAVLinkDroneDriver(connection_string='127.0.0.1:14552')

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        # Force the safety threshold to match or be lower than your pause_threshold
        recognizer.non_speaking_duration = 0.3  
        
        print("\nCalibrating audio environment hardware...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        
        # Now apply your custom speech timings safely
        recognizer.pause_threshold = 0.4
        recognizer.dynamic_energy_threshold = True
        
        print("\n=== SYSTEM IS REUSABLE & MODULARLY ACTIVE ===")
        voice_reply("Voice flight pipeline active. Awaiting your command.")
        
        while True:
            try:
                print("\n[ Listening... ]")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=3.5)
                captured_text = recognizer.recognize_google(audio).lower().strip()
                print(f"Captured: \"{captured_text}\"")

                # Parse Text via independent brain module
                intent, confidence = brain.analyze_phrase(captured_text)

                if intent == "invalid_command":
                    log_event(captured_text, "REJECTED", confidence, "USER_ALERTED_INVALID")
                    voice_reply("Your message is not correct. Please try again.")
                    continue

                # --- 1. Process Standard State Operations & Takeoff ---
                if intent in ["arm", "takeoff", "rtl", "loiter", "land"]:
                    if intent == "arm":
                        if drone.arm_vehicle():
                            log_event(captured_text, intent, confidence, "MOTORS_ARMED")
                            voice_reply("Arming aircraft propulsion motors.")
                        else:
                            voice_reply("The aircraft is already armed.")
                    
                    elif intent == "takeoff":
                        value = brain.extract_number(captured_text)
                        target_alt = value if value else 5.0  # Safe fallback to 5m
                        voice_reply(f"Taking off to target altitude of {int(target_alt)} meters.")
                        
                        if drone.execute_takeoff(target_alt):
                            log_event(captured_text, intent, confidence, f"TAKEOFF_ACTIVE_ALT_{target_alt}")
                        else:
                            voice_reply("Takeoff aborted. Please confirm the aircraft is armed.")
                            
                    elif intent == "rtl":
                        drone.change_mode("RTL")
                        log_event(captured_text, intent, confidence, "MODE_RTL")
                        voice_reply("Command verified. Returning back to home base.")
                    elif intent == "loiter":
                        drone.change_mode("LOITER")
                        log_event(captured_text, intent, confidence, "MODE_LOITER")
                        voice_reply("Holding position.")
                    elif intent == "land":
                        drone.change_mode("LAND")
                        log_event(captured_text, intent, confidence, "MODE_LAND")
                        voice_reply("Landing immediately.")

                # --- 2. Process Spatial & Altitude Movements ---
                else:
                    # Enforce critical in-flight safety boundary
                    if not drone.is_flying():
                        log_event(captured_text, intent, confidence, "REJECTED_NOT_IN_AIR")
                        voice_reply("Safety reject. The drone must be in the air first.")
                        continue

                    value = brain.extract_number(captured_text)
                    if not value:
                        value = 5.0 # Fallback default increment for movements if no number is spoken
                    
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
                    
                    # Altitude shifts
                    elif intent == "change_alt_absolute":
                        drone.set_absolute_altitude(value)
                        voice_reply(f"Changing altitude target to {int(value)} meters.")
                    elif intent == "change_alt_relative_up":
                        target = drone.get_current_altitude() + value
                        drone.set_absolute_altitude(target)
                        voice_reply(f"Climbing up by {int(value)} meters.")
                    elif intent == "change_alt_relative_down":
                        target = max(1.0, drone.get_current_altitude() - value)
                        drone.set_absolute_altitude(target)
                        voice_reply(f"Descending down by {int(value)} meters.")

                    log_event(captured_text, intent, confidence, f"MOVE_EXEC_VAL_{value}")

            except sr.UnknownValueError:
                continue
            except KeyboardInterrupt:
                print("\nShutting down central flight pipeline application loop safely.")
                break
            except Exception as e:
                print(f"Runtime Engine Error: {e}")
                
        drone.close()

if __name__ == "__main__":
    run_central_pipeline()