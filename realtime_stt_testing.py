if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Start the ultra-low-latency voice control STT test.')
    parser.add_argument('-m', '--model', type=str, help='Path to the STT model or model size. Optimized default is tiny.en.')
    parser.add_argument('-l', '--lang', '--language', type=str, help='Language code for the STT model.')
    parser.add_argument('-d', '--root', type=str, help='Root directory where the Whisper models are downloaded to.')

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    console = Console()
    console.print("[bold green]System initializing for low-latency control, please wait...[/bold green]")

    import os
    import sys
    from RealtimeSTT import AudioToTextRecorder
    import colorama
    import pyautogui

    # Windows-specific initialization fix for PyTorch audio paths
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    colorama.init()

    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    
    # Fast response window tailored for immediate robotic command processing (0.3 seconds)
    command_detection_pause = 0.30 

    def preprocess_text(text):
        text = text.lstrip().rstrip()
        if text.startswith("..."):
            text = text[3:]
        text = text.lstrip()
        if text:
            text = text[0].upper() + text[1:]
        return text

    def process_text(text):
        global full_sentences
        
        text = preprocess_text(text)
        if not text:
            return

        full_sentences.append(text)
        
        # Display the parsed command clearly in the console panel
        rich_text = Text()
        rich_text.append("Command Executed: ", style="bold green")
        rich_text.append(f"{text}", style="bold yellow")
        
        panel = Panel(rich_text, title="[bold cyan]Voice Control Loop[/bold cyan]", border_style="bold cyan")
        live.update(panel)

        # Emulate key entry into the active window frame
        pyautogui.write(f"{text} ", interval=0.002)

    # Hardened recorder configuration built for raw speed and zero buffer queues
    recorder_config = {
        'spinner': False,
        'model': 'tiny.en',                          # Always use tiny.en for real-time CPU control
        'compute_type': 'int8',                      # Highly optimized integer footprint for CPU runtime
        'download_root': None, 
        'language': 'en',
        
        # Voice Activity Detection (VAD) Tuning
        'silero_sensitivity': 0.05,                  # Suppresses background environmental noise
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': command_detection_pause, # Short pause reaction window
        'min_length_of_recording': 0.5,              # Allows short snappy commands like "Arm" or "Stop"
        'min_gap_between_recordings': 0,                
        
        # Performance Overhead Reduction
        'enable_realtime_transcription': False,      # Disabled to eliminate live streaming compute waste
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        
        # Greedy decoding configurations
        'beam_size': 1,                              # 1 means immediate single pass (Fastest decoding)
        'no_log_file': True,
        'silero_use_onnx': True,                     # Uses optimized ONNX runtime execution
        'faster_whisper_vad_filter': False,
    }

    args = parser.parse_args()
    if args.model is not None:
        recorder_config['model'] = args.model
    if args.lang is not None:
        recorder_config['language'] = args.lang
    if args.root is not None:
        recorder_config['download_root'] = args.root

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(Text("Ready for voice commands...", style="green bold"), title="[bold cyan]System Active[/bold cyan]", border_style="bold green")
    live.update(initial_text)

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Control Loop Stopped. Exiting...[/bold red]")
        sys.exit(0)