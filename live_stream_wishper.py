import os
import sys
import queue
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024        
MODEL_SIZE = "small" 
DEVICE = "cpu"            # Can be changed to "cuda" if utilizing an external GPU 
COMPUTE_TYPE = "int8"     

# --- NOISE THRESHOLD TUNING ---
SILENCE_THRESHOLD = 800  

def main():
    print("="*60)
    print("        WHISPER LIVE STREAM AUTOMATIC TEXT DISPLAY        ")
    print("="*60)
    
    print(f"Loading Whisper Engine ({MODEL_SIZE})...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load Whisper engine: {e}")
        sys.exit(1)
        
    audio_queue = queue.Queue()
    audio_buffer = np.zeros(0, dtype=np.float32)
    
    def stream_callback(in_data, frame_count, time_info, status):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    # Initialize Hardware Microphone Device
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=4,                
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=1,      # Targets your active Intel Smart Sound Array
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=stream_callback
        )
        print("[SUCCESS] Microphone connected.")
    except Exception as e:
        print(f"\n[ERROR] Failed to open Index 1. Trying fallback Index 17...")
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
            print("[SUCCESS] Fallback microphone connected.")
        except Exception as fallback_error:
            print(f"[CRITICAL ERROR] Could not open microphone channels: {fallback_error}")
            p.terminate()
            sys.exit(1)
    
    print("\n[READY] Speak clearly into the microphone. Press Ctrl+C to stop.")
    print("="*60)
    
    # Keep track of what we printed last to handle terminal line clearing smoothly
    last_printed_len = 0
    
    try:
        stream.start_stream()
        
        while stream.is_active():
            try:
                raw_data = audio_queue.get(timeout=0.1)
                
                # Convert raw byte chunks to standard int16 for Noise Gate logic
                audio_data_int16 = np.frombuffer(raw_data, dtype=np.int16)
                volume_level = np.sqrt(np.mean(audio_data_int16**2)) if len(audio_data_int16) > 0 else 0
                
                # Noise Gate: If room is silent, clear the buffer and skip processing
                if volume_level < SILENCE_THRESHOLD:
                    if len(audio_buffer) > 0:
                        audio_buffer = np.zeros(0, dtype=np.float32)
                    continue 
                
                # Convert audio data to Whisper's required float32 format
                audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0
                audio_buffer = np.concatenate((audio_buffer, audio_data_float32))
                
                # Transcribe speech context buffer
                segments, info = model.transcribe(
                    audio_buffer, 
                    language="en", 
                    beam_size=5,
                    vad_filter=True,
                    initial_prompt="Live text stream captions."
                )
                
                text_outputs = [segment.text for segment in segments]
                full_text = " ".join(text_outputs).strip()
                
                if full_text:
                    # Construct a clean text string without any volume numbers or codes
                    output_line = f"\r{full_text}"
                    
                    # Pad out old text on screen with spaces if current text is shorter
                    if len(output_line) < last_printed_len:
                        output_line += " " * (last_printed_len - len(output_line))
                        
                    sys.stdout.write(output_line)
                    sys.stdout.flush()
                    last_printed_len = len(output_line.strip())
                
                # Prevent buffer from accumulating over 20 seconds of old audio context
                max_buffer_len = SAMPLE_RATE * 20
                if len(audio_buffer) > max_buffer_len:
                    audio_buffer = audio_buffer[-max_buffer_len:]
                        
            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n\nStream stopped.")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Disconnected safely.")

if __name__ == "__main__":
    main()