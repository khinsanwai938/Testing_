import numpy as np
import pyaudio
import torch
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

def main():
    # 1. Choose a streaming-optimized cache-aware model
    # This checkpoint provides a stable 160ms chunk size with 80ms look-ahead.
    model_name = "stt_en_fastconformer_hybrid_large_streaming_80ms"
    print(f"Loading cache-aware streaming model: {model_name}...")
    
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model.eval()
    
    # Enforce fast greedy decoding over heavy beam-search for drone reactivity
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
    asr_model.change_decoding_strategy(decoding_cfg)

    # 2. Extract structural streaming properties from the model
    sample_rate = asr_model.preprocessor._sample_rate       # Usually 16000Hz
    frame_len = asr_model.encoder.chunk_size                # Typically 2 (representing two 80ms frames)
    step_len = asr_model.encoder.step_size                  # Chunk hop step size
    
    # Calculate bytes per processing window
    # 16000 samples/sec * 0.160 sec = 2560 samples per chunk
    chunk_size_ms = frame_len * 80                          # 160ms
    chunk_samples = int(sample_rate * (chunk_size_ms / 1000))
    
    # 3. Instantiate the stateful streaming cache buffer
    # This keeps track of encoder self-attention/convolution activations across steps
    cache_buffer = CacheAwareStreamingAudioBuffer(
        frame_len=frame_len,
        drop_extra_bits=False
    )
    cache_buffer.init_buffer(batch_size=1, device=str(asr_model.device))
    
    # 4. Configure PyAudio audio input stream
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_samples
    )
    
    print("\n🚀 Drone Audio Control Active. Speak into microphone... (Ctrl+C to stop)")
    
    try:
        while True:
            # Read fresh chunk raw byte data from mic
            raw_data = audio_stream.read(chunk_samples, exception_on_overflow=False)
            
            # Normalize Int16 byte streams to Float32 arrays for PyTorch ingestion
            signal = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(asr_model.device)
            signal_len = torch.tensor([signal_tensor.shape[1]], dtype=torch.long).to(asr_model.device)
            
            # Process chunk through the stateful cache hierarchy
            cache_buffer.append_audio(signal_tensor, signal_len)
            
            # Extract accumulated frames and pass them to the model if ready
            while cache_buffer.is_ready():
                audio_chunk, chunk_length, cache_states = cache_buffer.get_next_chunk()
                
                with torch.no_grad():
                    # Run streaming matrix transformations
                    log_probs, encoded_len, output_states = asr_model.encoder.forward_streaming(
                        audio_signal=audio_chunk,
                        length=chunk_length,
                        cache_states=cache_states
                    )
                    
                    # Update streaming state boundaries with calculated hidden layers
                    cache_buffer.update_cache_states(output_states)
                    
                    # Decode hidden layers into plain-text string tokens
                    hypotheses = asr_model.decoding.ctc_decoder_predictions_tensor(
                        log_probs, encoded_len
                    )
                    
                    # Clean text output string representation
                    text_output = hypotheses[0] if isinstance(hypotheses, list) else hypotheses
                    if text_output.strip():
                        print(f"[Command Detected]: {text_output.upper()}")
                        
                        # --- INTERFACE CORRESPONDING FLIGHT ACTION COMMANDS HERE ---
                        # if "land" in text_output.lower():
                        #     drone.issue_command("LAND")
                        # -----------------------------------------------------------

    except KeyboardInterrupt:
        print("\nStopping audio ingestion engine...")
    finally:
        # Graceful cleanup operations
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()
        print("Audio streams torn down cleanly.")

if __name__ == "__main__":
    main()