import os
import sys
import queue
import numpy as np
import pyaudio
import warnings
from datetime import datetime
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024        
MODEL_SIZE = "small" 
DEVICE = "cpu"            
COMPUTE_TYPE = "int8"     

# --- PHRASE SPLITTING CONFIGURATION ---
SILENCE_THRESHOLD = 600      # Volume cutoff for noise. Increase if your room has a loud PC fan.
SILENCE_WINDOW = 1.2         # Pause length (seconds) required to finalize a sentence.

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
    print("=== COLD RESET PHRASE TRANSCRIPTION ===")
    print(f" Saving text history to: {output_filename}")
    print(" (Speak a sentence, pause briefly to print, then speak the next.)\n")
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
            channels=1, # Mono                
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
                    
                    # Calculate real-time amplitude/volume
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
                    
                    # Scenario A: User is talking -> Live preview updates on the same line
                    if has_spoken_in_phrase and continuous_silence_time < SILENCE_WINDOW:
                        if len(audio_buffer) % (SAMPLE_RATE // 2) == 0: 
                            segments, _ = model.transcribe(
                                audio_buffer, 
                                language="en", 
                                beam_size=2, 
                                vad_filter=True,
                                condition_on_previous_text=False, # Anti-loop fix
                                temperature=0.0
                            )
                            text = " ".join([seg.text for seg in segments]).strip()
                            if text:
                                print(f"\rListening: {text}", end="", flush=True)
                    
                    # Scenario B: User paused -> Freeze final text, drop a line, and nuke everything
                    elif has_spoken_in_phrase and continuous_silence_time >= SILENCE_WINDOW:
                        segments, _ = model.transcribe(
                            audio_buffer, 
                            language="en", 
                            beam_size=3, 
                            vad_filter=True,
                            condition_on_previous_text=False, # Anti-loop fix
                            temperature=0.0
                        )
                        final_text = " ".join([seg.text for seg in segments]).strip()
                        
                        if final_text:
                            # Clear line and cleanly print finalized sentence
                            sys.stdout.write("\r" + " " * 120 + "\r")
                            print(f"Sentence: {final_text}")
                            
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {final_text}\n")
                            f.flush()
                        
                        # COLD RESET STATE
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        has_spoken_in_phrase = False
                        continuous_silence_time = 0.0
                        
                        # Completely clear background queue so old background whispers don't spill over
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