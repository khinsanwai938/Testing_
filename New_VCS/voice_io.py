"""
voice_io.py — Text-to-speech output and STT mute-window management.

Handles:
  - Speaking replies via pyttsx3 in a background thread
  - Muting the STT input while TTS is speaking (echo suppression)

IMPORTANT: filtering transcription results AFTER the fact is not
enough — the microphone is still actively recording the TTS audio
playing out the speakers, which both creates phantom transcriptions
AND competes with real microphone access during real commands. The
fix used here is to physically disable the recorder's microphone via
RealtimeSTT's set_microphone(False) for the full duration audio could
plausibly still be playing, then re-enable it afterward. is_muted()
is kept as a secondary belt-and-suspenders filter for any transcription
that slips through right at the boundary of the mute window.
"""

import time
import threading
import pyttsx3

# ── TTS mute state ────────────────────────────────────────────────────────
_tts_mute_until = 0.0
_tts_lock       = threading.Lock()
_tts_busy       = False   # True for the entire duration audio could be playing
_recorder       = None    # set via register_recorder(), used to hard-mute the mic

WORDS_PER_SECOND = 170 / 60.0   # matches engine rate=170 wpm, used for estimate
TTS_POST_DELAY   = 1.5          # seconds to mute AFTER speech ends (room echo/reverb)
MIN_MUTE_WINDOW  = 1.0          # floor, in case of very short replies


def register_recorder(recorder):
    """
    Register the live AudioToTextRecorder instance so voice_reply() can
    physically disable its microphone during TTS playback, instead of
    only filtering results after the mic already heard the echo.
    """
    global _recorder
    _recorder = recorder


def _set_mic_enabled(enabled: bool):
    """Best-effort toggle of the recorder's microphone, if one is registered."""
    if _recorder is None:
        return
    try:
        _recorder.set_microphone(enabled)
    except Exception as e:
        print(f"[VOICE IO] Could not toggle microphone ({enabled=}): {e}")


def _mute_for(seconds: float):
    """Extend the STT mute window by `seconds` from now."""
    global _tts_mute_until
    with _tts_lock:
        _tts_mute_until = max(_tts_mute_until, time.time() + seconds)


def is_muted() -> bool:
    """Return True if STT input should currently be ignored (TTS echo guard)."""
    with _tts_lock:
        if _tts_busy:
            return True
        return time.time() < _tts_mute_until


def voice_reply(text: str):
    """Speak `text` aloud via TTS in a background thread. Physically disables
    the registered recorder's microphone for the duration of playback plus
    an echo buffer, and re-enables it afterward."""
    global _tts_busy
    print(f"[SYSTEM SPEAK] → \"{text}\"")

    # Mute immediately, BEFORE the engine even starts, so there's no gap
    # between "decided to speak" and "actually started making sound".
    estimated_duration = max(len(text.split()) / WORDS_PER_SECOND, MIN_MUTE_WINDOW)
    with _tts_lock:
        _tts_busy = True
    _mute_for(estimated_duration + TTS_POST_DELAY)  # generous upfront estimate
    _set_mic_enabled(False)

    def _speak():
        global _tts_busy
        start = time.time()
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            del engine
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            with _tts_lock:
                _tts_busy = False
            # Re-extend mute from the ACTUAL elapsed speaking time, in case
            # the real sentence took longer than our word-count estimate.
            actual_duration = time.time() - start
            _mute_for(max(actual_duration, 0) + TTS_POST_DELAY)

            # Hold the mic disabled a little past the mute window to absorb
            # room echo/reverb, then re-enable for the next real command.
            def _reenable():
                _set_mic_enabled(True)
                print("[VOICE IO] Microphone re-enabled.")

            threading.Timer(TTS_POST_DELAY, _reenable).start()

    threading.Thread(target=_speak, daemon=True).start()