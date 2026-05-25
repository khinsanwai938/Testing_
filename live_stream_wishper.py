import os
import sys

# 1. GUARDRAIL: Suppress Hugging Face warnings and underlying logging clutter
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CTRANS_LOG_LEVEL"] = "3"

import queue
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024        
SILENCE_LIMIT = 0.8       # Snappy cutoff (under 1 sec) so drone reacts fast when you stop speaking
ENERGY_THRESHOLD = 800    # Raised from 500 to ignore background room noise, fans, and heavy breathing
LANGUAGE_CODE = "en"      # Locked to English to completely eliminate accidental Korean translations

def main():
    print("="*60)
    print(" INITIALIZING DRONE VOICE CONTROL INTERFACE ")
    print("="*60)
    print("Loading Optimized Local Whisper Base Engine on CPU...")
    
    # 2. SPEED FIX: Using 'base.en' (74M params) instead of 'large' (809M params) drops math load by 90%.
    # 'cpu_threads=4' forces parallel math execution across your CPU cores.
    model = WhisperModel("medium.en", device="cpu", compute_type="int8", cpu_threads=4)
    
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
    
    print("\n" + "[READY] Drone System listening for commands...")
    print("="*60)
    print("🗣️ USER COMMAND DISPLAY:")
    print("-" * 60)

    audio_buffer = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_LIMIT * (SAMPLE_RATE / CHUNK_SIZE))
    recording_started = False

    try:
        stream.start_stream()
        
        while stream.is_active():
            try:
                # Grab a chunk of sound from the microphone queue
                raw_data = audio_queue.get(timeout=0.1)
                raw_signal = np.frombuffer(raw_data, dtype=np.int16)
                
                # Upcast array to float64 to completely stop the sqrt invalid value overflow crash
                signal_chunk = raw_signal.astype(np.float64)
                energy = np.sqrt(np.mean(signal_chunk**2))
                
                # Check if volume crosses our room-noise threshold
                if energy > ENERGY_THRESHOLD:
                    if not recording_started:
                        sys.stdout.write("\r🎙️ [Listening...]                      ")
                        sys.stdout.flush()
                    audio_buffer.extend(raw_signal)  # Build the audio command phrase
                    silent_chunks = 0
                    recording_started = True
                else:
                    if recording_started:
                        audio_buffer.extend(raw_signal)
                        silent_chunks += 1
                
                # If silence window is hit, pass the collected audio straight to the AI
                if recording_started and silent_chunks > max_silent_chunks:
                    sys.stdout.write("\r🧠 [Processing Command...]             ")
                    sys.stdout.flush()
                    
                    # Normalize amplitude array to Whisper specification standard [-1.0, 1.0]
                    audio_data = np.array(audio_buffer, dtype=np.float32) / 32768.0
                    
                    # Avoid analyzing tiny clicks or pops under 0.4 seconds
                    if len(audio_data) > SAMPLE_RATE * 0.4:
                        
                        # 3. ANTI-HALLUCINATION FIX: Added strict thresholds to kill loop repetitions
                        segments, _ = model.transcribe(
                            audio_data, 
                            beam_size=1, 
                            language=LANGUAGE_CODE,
                            compression_ratio_threshold=2.4,  # Automatically drops repeating text structures
                            no_speech_threshold=0.6,          # Rejects the chunk if it thinks it's just static background noise
                            condition_on_previous_text=False  # Stops the model from copying its own previous outputs
                        )
                        
                        transcript = " ".join([seg.text for seg in segments]).strip()
                        
                        if transcript:
                            sys.stdout.write(f"\r💬 User Said: \"{transcript}\"\n")
                            sys.stdout.flush()
                        else:
                            sys.stdout.write("\r⏱️ [Static/Noise Filtered Out]\n")
                    
                    # Flush the system variables so it's fresh for your next voice command
                    audio_buffer = []
                    silent_chunks = 0
                    recording_started = False
                    sys.stdout.write("\r🎙️ [Ready for next command...] ")
                    sys.stdout.flush()
                    
            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n\nShutting down Drone Voice Control Pipeline...")
    finally:
        # Gracefully sever active audio hardware hooks
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("System safely disconnected.")

if __name__ == "__main__":
    main()