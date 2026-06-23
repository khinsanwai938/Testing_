"""
stt_engine.py — Speech-to-text capture and the live-mode listening loop.

Wraps RealtimeSTT's AudioToTextRecorder, applies the TTS echo-mute
window, and normalizes captured text before handing it to dispatch.
"""

from rich.console import Console
from RealtimeSTT import AudioToTextRecorder

from voice_io import is_muted, register_recorder

# ── Vocabulary hint for the STT decoder ──────────────────────────────────
PROMPT_CONTEXT = (
    "arm disarm takeoff land loiter RTL return to launch guided poshold "
    "move forward move backward move left move right hover hold position "
    "ascend descend climb go up go down rotate left rotate right yaw "
    "set heading north south east west compass bearing degrees "
    "go to waypoint navigate coordinates latitude longitude "
    "set airspeed set groundspeed speed metres per second "
    "altitude check heading check battery status GPS status "
    "current mode current location where am I how high "
    "emergency abort safe state trigger emergency stop "
    "save waypoint mark position export mission flight plan "
    "change altitude increase altitude decrease altitude "
    "loiter mode position hold circle spin turn face "
    "meters seconds faster slower manual "
    "yes no confirm cancel"
)

RECORDER_CONFIG = {
    'spinner':                        False,
    'model':                          'small',
    'compute_type':                   'int8',
    'download_root':                  None,
    'language':                       'en',
    'initial_prompt':                 PROMPT_CONTEXT,
    'silero_sensitivity':             0.05,
    'webrtc_sensitivity':             3,
    'post_speech_silence_duration':   0.30,
    'min_length_of_recording':        0.5,
    'min_gap_between_recordings':     0,
    'enable_realtime_transcription':  False,
    'silero_deactivity_detection':    True,
    'early_transcription_on_silence': 0,
    'beam_size':                      1,
    'no_log_file':                    True,
    'silero_use_onnx':                True,
    'faster_whisper_vad_filter':      False,
}


def normalize_text(raw_text: str) -> str:
    """Strip, drop a leading stray period, capitalize first letter."""
    text = raw_text.strip().lstrip(".")
    if not text:
        return ""
    return text[0].upper() + text[1:]


def build_recorder() -> AudioToTextRecorder:
    """Construct the RealtimeSTT recorder with project-tuned settings."""
    return AudioToTextRecorder(**RECORDER_CONFIG)


def run_voice_loop(console: Console, vtype: str, on_command, gate=None):
    """
    Block forever, transcribing speech and calling on_command(text) for
    every utterance that isn't dropped as a TTS echo.

    on_command: Callable[[str], None] — receives the normalized text.
    gate: optional ConfirmationGate — if provided, its check_timeout()
          is polled on every silence cycle so a pending "are you sure?"
          question can time out even if the user never replies at all.
    """
    print(f"[STT ENGINE] Building recorder for vehicle type: {vtype}")
    recorder = build_recorder()
    register_recorder(recorder)
    print("[STT ENGINE] Recorder ready. Listening...")

    def on_transcription(text: str):
        if is_muted():
            print(f"[MUTED] Dropped echo: \"{text.strip()}\"")
            return

        text = normalize_text(text)
        if not text:
            print("[STT] Heard silence/empty transcription — ignoring.")
            return

        console.print(f"[bold yellow]Command ({vtype.upper()}):[/bold yellow] {text}")
        print(f'[STT] Transcribed: "{text}"')
        on_command(text)

    while True:
        recorder.text(on_transcription)
        if gate is not None:
            gate.check_timeout()