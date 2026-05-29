import numpy as np
import pyaudio
import torch
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

def main():
    # 1. Choose a streaming-optimized cache-aware model
    model_name = "stt_en_fastconformer_hybrid_large_streaming_80ms"
    print(f"Loading cache-aware streaming model: {model_name}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference engine on device: {device.upper()}")
    
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model = asr_model.to(device)
    asr_model.eval()
    
    # Enforce fast greedy decoding over heavy beam-search for drone reactivity
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
    asr_model.change_decoding_strategy(decoding_cfg)

    # 2. Extract structural streaming properties safely
    sample_rate = asr_model.preprocessor._sample_rate       # Usually 16000Hz
    
    if hasattr(asr_model.encoder, 'streaming_cfg') and asr_model.encoder.streaming_cfg is not None:
        chunk_size_ms = getattr(asr_model.encoder.streaming_cfg, 'chunk_size_ms', 160)
    else:
        chunk_size_ms = 160
    
    # Calculate samples per processing window (16000 * 0.160 = 2560 samples)
    chunk_samples = int(sample_rate * (chunk_size_ms / 1000))
    print(f"Audio buffer initialized: {chunk_size_ms}ms window ({chunk_samples} samples per chunk)")
    
    # 3. Instantiate the stateful streaming cache buffer
    cache_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=True
    )
    
    # FIX: Native unpacking of the initialized cache states into explicit variables
    init_caches = asr_model.encoder.get_initial_cache_state(batch_size=1, device=asr_model.device)
    cache_last_channel = init_caches[0]
    cache_last_time = init_caches[1]
    cache_last_channel_len = init_caches[2]
    
    previous_hypotheses = None
    pred_out_stream = None
    
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
            
            # Normalize Int16 byte streams to standard Float32 NumPy arrays
            signal = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # NeMo processes the NumPy array and pushes tensors automatically
            processed_signal, processed_signal_len, stream_id = cache_buffer.append_audio(signal)
            
            # Safety check: Ensure the buffer has parsed enough frames to compute
            if processed_signal is None or processed_signal.nelement() == 0:
                continue

            # Ensure processed_signal_len is a 1D tensor with shape (1,) instead of a 0D scalar ()
            if processed_signal_len.ndim == 0:
                processed_signal_len = processed_signal_len.unsqueeze(0)
            elif len(processed_signal_len.shape) > 1:
                processed_signal_len = processed_signal_len.reshape(-1)

            with torch.no_grad():
                # FIX: Pass explicit variables using the correct keyword arguments
                outputs = asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_len,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    return_transcription=True,  
                )
                
                # FIX: Unpack matching the exact output order returned by conformer_stream_step
                pred_out_stream = outputs[0]
                transcribed_texts = outputs[1]
                cache_last_channel = outputs[2]
                cache_last_time = outputs[3]
                cache_last_channel_len = outputs[4]
                previous_hypotheses = outputs[5]
                
                # Extract text output strings from the streaming step results
                if transcribed_texts and len(transcribed_texts) > 0:
                    text_output = transcribed_texts[0]
                    
                    # If it's a structural Hypothesis object, extract its string property
                    if hasattr(text_output, 'text'):
                        text_output = text_output.text
                    elif isinstance(text_output, tuple):
                        text_output = text_output[0]
                    
                    if isinstance(text_output, str) and text_output.strip():
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