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
MODEL_PATH = r"C:\Users\16G7IML\Downloads\vosk-model-small-en-us-0.15"

# --- NOISE THRESHOLD TUNING ---
# 300 to 500 = Very quiet room / sensitive mic
# 800 to 1200 = Standard room with fan noise/typing
# 1500+ = Loud environment
SILENCE_THRESHOLD = 800  

def main():
    print("="*60)
    print("        VOSK LIVE STREAM TEXT DISPLAY (FIXED HARDWARE)     ")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Vosk model not found at '{MODEL_PATH}'.")
        sys.exit(1)

    print("Loading Vosk Engine...")
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    
    audio_queue = queue.Queue()
    
    def stream_callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    # Initialize Hardware Microphone Device
    p = pyaudio.PyAudio()
    
    try:
        # Fixed to use Index 1 (Intel Mic Array) with 4 Channels to avoid Vol: 0
        stream = p.open(
            format=pyaudio.paInt16,
            channels=4,                # Matches your Index 1 hardware requirement
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=1,      # Targets your active Intel Smart Sound Array
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=stream_callback
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to open Index 1. Trying fallback Index 17 (2 Channels)...")
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=17,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=stream_callback
            )
        except Exception as fallback_error:
            print(f"[CRITICAL ERROR] Could not open microphone channels: {fallback_error}")
            p.terminate()
            sys.exit(1)
    
    print(f"\n[READY] Noise gate active (Threshold: {SILENCE_THRESHOLD}).")
    print("Speak clearly into the microphone. Press Ctrl+C to stop.")
    print("="*60)
    
    try:
        stream.start_stream()
        
        while stream.is_active():
            try:
                raw_data = audio_queue.get(timeout=0.1)
                
                # Calculate live sound volume level
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                volume_level = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
                
                # Noise Gate: If the room is silent, drop data before Vosk processes it
                if volume_level < SILENCE_THRESHOLD:
                    sys.stdout.write(f"\r💤 [Silent Noise Filtered | Vol: {int(volume_level)}]")
                    sys.stdout.flush()
                    continue 
                
                # Send valid audio to the speech engine
                if recognizer.AcceptWaveform(raw_data):
                    # Final stabilized sentence output when you take a small pause
                    result_json = json.loads(recognizer.Result())
                    final_text = result_json.get("text", "").strip()
                    
                    if final_text:
                        sys.stdout.write(f"\r💬 FINAL TEXT: {final_text}\n")
                        sys.stdout.flush()
                else:
                    # Real-time rapid feedback as you speak words
                    partial_json = json.loads(recognizer.PartialResult())
                    partial_text = partial_json.get("partial", "").strip()
                    
                    if partial_text:
                        sys.stdout.write(f"\r🎙️ Live Processing [Vol: {int(volume_level)}]: {partial_text}")
                        sys.stdout.flush()
                        
            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n\nTesting stopped.")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("System safely disconnected.")

if __name__ == "__main__":
    main()