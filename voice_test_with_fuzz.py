import os
import sys
import queue
import numpy as np
import pyaudio
import warnings
from datetime import datetime
from faster_whisper import WhisperModel

# Import RapidFuzz modules for smart string matching
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024        
MODEL_SIZE = "small" 
DEVICE = "cpu"            
COMPUTE_TYPE = "int8"     

# --- PHRASE SPLITTING CONFIGURATION ---
SILENCE_THRESHOLD = 600      
SILENCE_WINDOW = 1.2         

# --- FUZZY COMMAND CONFIGURATION ---
MATCH_THRESHOLD = 75.0  # Scores above 75% will trigger the action

# Synonyms dictionary mapping commands to intent phrases
COMMAND_DICTIONARY = {
    "ARM": [
        "arm the drone",
        "arm engines",
        "turn on motors",
        "system unlock",
        "engage motors",
        "start up the drone",
        "Unlock flight controller",
        "System arm",
        "Arm",
        "Turn on the drone",
        "Let's ready the drone for flight",
        "Get the props spinning"
    ],
    "DISARM": [
        "disarm the drone",
        "disarm engines",
        "cut power",
        "turn off motors",
        "kill the motors",
        "stop the propellers",
        "lock system",
        "Cut the engines",
        "Shut down the engines",
        "Power down the drone",
        "Okay, you can turn off the power now"
    ],
    "TAKEOFF": [
        "take off", 
        "launch drone", 
        "start flying", 
        "go up", 
        "lift off", 
        "begin flight",
        "fly up",
        "Fly up into the air",
        "Lift off the ground",
        "Go ahead"
    ],
    "RTL": [
        "return to launch", 
        "come back home", 
        "go home", 
        "return home", 
        "rtl", 
        "land at base",
        "fly back",
        "Retrun to base",
        "Go to home position",
        "Fly back to where you started",
        "Bring the drone back to me",
        "Time to come home",
        "Bring it back",
        "Retrun and Land",
        "Go back and Land"
    ]
}

def execute_drone_command(command_type, matched_phrase, score):
    """Placeholder function where your actual drone control API code goes"""
    print("\n" + "="*50)
    print(f" [ACTION TRIGGERED] -> Command: {command_type}")
    print(f" (Matched with '{matched_phrase}' | Confidence: {score:.1f}%)")
    print("="*50 + "\n")
    
    if command_type == "ARM":
        # vehicle.armed = True goes here
        pass
    elif command_type == "DISARM":
        # vehicle.armed = False goes here
        pass
    elif command_type == "TAKEOFF":
        # drone.arm_and_takeoff() goes here
        pass
    elif command_type == "RTL":
        # drone.set_mode("RTL") goes here
        pass

def process_voice_command(text):
    """Uses WRatio to check if the spoken sentence matches any known commands"""
    best_command = None
    best_score = 0.0
    best_match_phrase = ""
    
    # Iterate over our defined intents
    for command_action, target_phrases in COMMAND_DICTIONARY.items():
        # extractOne uses WRatio to find the single best phrase match in the list
        result = process.extractOne(text, target_phrases, scorer=fuzz.WRatio)
        
        if result:
            matched_phrase, score, _ = result
            if score > best_score:
                best_score = score
                best_command = command_action
                best_match_phrase = matched_phrase
                
    # If the score beats our threshold, fire the execution function
    if best_score >= MATCH_THRESHOLD:
        execute_drone_command(best_command, best_match_phrase, best_score)
    else:
        print(f" -> [No Match] (Best attempt: {best_command} at {best_score:.1f}%)")


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Loading AI Speech Engine (Faster-Whisper)...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Whisper engine: {e}")
        sys.exit(1)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"whisper_transcript_{timestamp}.txt"
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== FUZZY VOICE COMMAND SYSTEM STARTED ===")
    print(f" Try saying variations like: 'Arm the drone', 'Cut power', or 'Launch drone'")
    print(" (Speak a sentence, then pause briefly to send it)\n")
    print("="*60)

    audio_queue = queue.Queue()
    audio_buffer = np.zeros(0, dtype=np.float32)
    
    continuous_silence_time = 0.0
    has_spoken_in_phrase = False

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
        
        with open(output_filename, "a", encoding="utf-8") as f:
            while stream.is_active():
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                    audio_data_int16 = np.frombuffer(raw_data, dtype=np.int16)
                    
                    if len(audio_data_int16) == 0:
                        continue
                    
                    mean_square = np.mean(audio_data_int16.astype(np.float64)**2)
                    volume_level = np.sqrt(mean_square) if mean_square > 0 else 0
                    chunk_duration = len(audio_data_int16) / SAMPLE_RATE
                    
                    if volume_level < SILENCE_THRESHOLD:
                        continuous_silence_time += chunk_duration
                    else:
                        continuous_silence_time = 0.0
                        has_spoken_in_phrase = True
                    
                    audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0
                    audio_buffer = np.concatenate((audio_buffer, audio_data_float32))
                    
                    # Live partial tracking printout
                    if has_spoken_in_phrase and continuous_silence_time < SILENCE_WINDOW:
                        if len(audio_buffer) % (SAMPLE_RATE // 2) == 0: 
                            segments, _ = model.transcribe(
                                audio_buffer, language="en", beam_size=2, vad_filter=True,
                                condition_on_previous_text=False, temperature=0.0
                            )
                            text = " ".join([seg.text for seg in segments]).strip()
                            if text:
                                print(f"\rListening: {text}", end="", flush=True)
                    
                    # Finalized Sentence - Send to Fuzzy Matcher
                    elif has_spoken_in_phrase and continuous_silence_time >= SILENCE_WINDOW:
                        segments, _ = model.transcribe(
                            audio_buffer, language="en", beam_size=3, vad_filter=True,
                            condition_on_previous_text=False, temperature=0.0
                        )
                        final_text = " ".join([seg.text for seg in segments]).strip()
                        
                        if final_text:
                            sys.stdout.write("\r" + " " * 120 + "\r")
                            print(f"Spoken: {final_text}")
                            
                            # RUN THE FUZZY ENGINE
                            process_voice_command(final_text)
                            
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {final_text}\n")
                            f.flush()
                        
                        # COLD RESET STATE
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        has_spoken_in_phrase = False
                        continuous_silence_time = 0.0
                        
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        
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