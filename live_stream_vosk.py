import os
import sys
import json
import queue
import numpy as np
import pyaudio
from datetime import datetime
from vosk import Model, KaldiRecognizer, SetLogLevel

# Mute Vosk/Kaldi internal C++ logging chatter to keep console clean
SetLogLevel(-1)

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024         

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*60)
    print(" INITIALIZING DRONE VOICE CONTROL INTERFACE (VOSK) ")
    print("="*60)
    print("Loading Stream-Optimized Offline Vosk Engine...")
    
    model = Model(lang="en-us")
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    # Generate a unique filename based on the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"drone_transcript_{timestamp}.txt"
    
    audio_queue = queue.Queue()
    
    def stream_callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=stream_callback
    )
    
    print(f"\n[SUCCESS] Saving text history directly to: {output_filename}")
    print("[READY] Drone System listening for commands...")
    print("="*60)
    
    try:
        stream.start_stream()
        
        # Open the text file in append mode ('a') with UTF-8 encoding
        with open(output_filename, "a", encoding="utf-8") as f:
            while stream.is_active():
                try:
                    raw_data = audio_queue.get(timeout=0.1)
                    
                    if recognizer.AcceptWaveform(raw_data):
                        result_json = json.loads(recognizer.Result())
                        transcript = result_json.get("text", "").strip()
                        
                        if transcript:
                            # Clear the terminal line and print final text
                            sys.stdout.write("\r".ljust(100) + "\r")
                            sys.stdout.write(f"Text Transcriptions is: {transcript}\n")
                            sys.stdout.flush()
                            
                            # Save the final transcription to the text file
                            f.write(f"Text Transcriptions is: {transcript}\n")
                            f.flush()  # Force write data to disk immediately
                    else:
                        partial_json = json.loads(recognizer.PartialResult())
                        partial_text = partial_json.get("partial", "").strip()
                        
                        if partial_text:
                            # Display live tracking text on screen
                            sys.stdout.write(f"\rText Transcriptions is: {partial_text}...".ljust(100))
                            sys.stdout.flush()
                            
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