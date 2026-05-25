import os
import sys
import queue
import json
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024        
MODEL_PATH = r"C:\Users\16G7IML\Downloads\vosk-model-small-en-us-0.15" # Path to your downloaded extracted Vosk model folder

def main():
    print("="*60)
    print(" INITIALIZING DRONE VOICE CONTROL INTERFACE (VOSK) ")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Vosk model not found at '{MODEL_PATH}'.")
        print("Please download 'vosk-model-small-en-us-0.15' from alphacephei.com/vosk/models")
        print("and extract it into this directory named as 'model'.")
        sys.exit(1)

    print("Loading Ultra-Low Latency Vosk Engine...")
    model = Model(MODEL_PATH)
    
    # Drone Keyword Grammar optimization (Optional but helps accuracy):
    # You can restrict the AI to only listen for specific drone words to prevent false triggers
    # recognizer = KaldiRecognizer(model, SAMPLE_RATE, '["take off", "land", "hover", "fly forward", "stop"]')
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    audio_queue = queue.Queue()
    
    # PyAudio Input Stream Callback
    def stream_callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    # Initialize Hardware Microphone Device
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=stream_callback
    )
    
    print("\n" + "[READY] Drone System listening for commands instantly...")
    print("="*60)
    print("🗣️ USER COMMAND DISPLAY:")
    print("-" * 60)
    
    try:
        stream.start_stream()
        sys.stdout.write("🎙️ [Listening...] ")
        sys.stdout.flush()
        
        while stream.is_active():
            try:
                raw_data = audio_queue.get(timeout=0.1)
                
                # Vosk accepts raw int16 bytes directly and checks for phrase endings internally
                if recognizer.AcceptWaveform(raw_data):
                    result_json = json.loads(recognizer.Result())
                    transcript = result_json.get("text", "").strip()
                    
                    if transcript:
                        # Clear line and print final command
                        sys.stdout.write(f"\r💬 User Said: \"{transcript}\"\n")
                        sys.stdout.write("🎙️ [Ready for next command...] ")
                        sys.stdout.flush()
                else:
                    # Optional: Grab partial results if you want to see text appear mid-sentence
                    # partial_json = json.loads(recognizer.PartialResult())
                    # partial_text = partial_json.get("partial", "")
                    pass
                    
            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n\nShutting down Drone Voice Control Pipeline...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("System safely disconnected.")

if __name__ == "__main__":
    main()