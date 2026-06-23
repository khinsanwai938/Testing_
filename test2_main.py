import os
import sys
import time
import threading
from datetime import datetime

import pyttsx3
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from RealtimeSTT import AudioToTextRecorder

from llm_con import DroneNLPEngine
from driver import MAVLinkDroneDriver

# ── Windows DLL path fix for PyTorch audio ────────────────────────────────────
if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    from torchaudio._extension.utils import _init_dll_path
    _init_dll_path()

TEST_MODE = False

# ── TTS mute state ────────────────────────────────────────────────────────────
# _tts_mute_until holds a future timestamp (from time.time()) until which
# all STT transcriptions are silently discarded.  This is more reliable than
# a simple boolean flag because it survives the race window where RealtimeSTT
# has already captured audio *before* the flag was raised.
_tts_mute_until = 0.0
_tts_lock       = threading.Lock()

TTS_PRE_DELAY   = 0.15   # seconds to mute BEFORE speech starts  (capture lag)
TTS_POST_DELAY  = 1.20   # seconds to mute AFTER  speech ends    (room echo)

console = Console()


# ======================================================================
# TTS mute helpers
# ======================================================================

def _mute_for(seconds: float):
    """Extend the mute window by `seconds` from now."""
    global _tts_mute_until
    with _tts_lock:
        _tts_mute_until = max(_tts_mute_until, time.time() + seconds)


def _is_muted() -> bool:
    with _tts_lock:
        return time.time() < _tts_mute_until


# ======================================================================
# Text-to-speech
# ======================================================================

def voice_reply(text: str):
    print(f"[SYSTEM SPEAK] -> \"{text}\"")

    def _speak():
        try:
            # Mute mic BEFORE audio leaves the speaker so the very first
            # chunk of TTS is never captured by RealtimeSTT.
            _mute_for(TTS_PRE_DELAY)
            time.sleep(TTS_PRE_DELAY)

            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            del engine

            # Keep muted for echo / reverb tail after speech ends.
            _mute_for(TTS_POST_DELAY)

        except Exception as e:
            print(f"[TTS ERROR] {e}")

    threading.Thread(target=_speak, daemon=True).start()


# ======================================================================
# Logging
# ======================================================================

def log_event(raw_text: str, intent: str, confidence: float, outcome: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("voice_flight_commands.log", "a") as f:
        f.write(
            f"[{timestamp}] TEXT: \"{raw_text}\" | "
            f"INTENT: {intent.upper()} | "
            f"CONF: {confidence:.2f} | "
            f"STATUS: {outcome}\n"
        )


# ======================================================================
# Intent dispatch  (shared between LIVE and TEST modes)
# ======================================================================

def dispatch_command(captured_text: str, brain: DroneNLPEngine,
                     drone: MAVLinkDroneDriver):
    """Parse captured_text and execute the corresponding drone command."""

    intent, confidence = brain.match_intent(captured_text)

    if not intent or intent == "invalid_command":
        log_event(captured_text, "REJECTED", confidence, "USER_ALERTED_INVALID")
        voice_reply("Command not recognized. Please try again.")
        return

    # ── Emergency (highest priority — always allowed) ──────────────────
    if intent == "emergency_safe":
        action = drone.trigger_emergency_safe_state()
        log_event(captured_text, intent, confidence, f"EMERGENCY_{action}")
        voice_reply(
            "Emergency safe state active. Returning home."
            if action == "RTL_ACTIVATED"
            else "Emergency safe state active. Landing immediately."
        )

    # ── Ground commands ────────────────────────────────────────────────
    elif intent == "takeoff" or intent == "ascend":  # Synchronized intent names
        value = brain.extract_number(captured_text)
        target_alt = float(value) if value else 5.0
                
        # 1. FIXED: Using the public is_flying check (replaces private _is_armed check logic safely)
        current_altitude = drone.get_altitude()
        if current_altitude is None:
            current_altitude = 0.0  # Safe fallback if initial packet hasn't arrived yet

        # CASE A: Drone is on the ground -> Run standard takeoff
        if current_altitude < 0.5:
            voice_reply(
                f"Taking off to {int(target_alt)} meters. "
                f"Commands locked until altitude is reached."
            )
            if drone.execute_takeoff(target_alt):
                log_event(captured_text, intent, confidence, f"TAKEOFF_COMPLETE_ALT_{target_alt}")
            else:
                voice_reply("Takeoff sequence aborted.")
                log_event(captured_text, intent, confidence, "TAKEOFF_FAILED")

        # CASE B: Drone is already flying -> Direct relative climb via velocity tracking
        else:
            print(f"[DRIVER] Airborne state verified at ({current_altitude:.2f}m).")
            print(f"[DRIVER] Dispatching velocity climb step: +{target_alt}m")
                    
            voice_reply(f"Ascending an additional {int(target_alt)} meters.")
                    
            # FIXED: Removed the deleted absolute method call. 
            # We now pass the raw distance value straight to your driver's optimized ascend method.
            if drone.ascend(distance=target_alt):
               log_event(captured_text, intent, confidence, f"ALTITUDE_INCREASE_BY_{target_alt}M")
            else:
                voice_reply("Ascent sequence halted.")
                log_event(captured_text, intent, confidence, "ALTITUDE_INCREASE_FAILED")         

    # ── In-flight commands ─────────────────────────────────────────────
    elif intent == "land":
        if drone.land():
            log_event(captured_text, intent, confidence, "LANDED")
            voice_reply("Landed.")
        else:
            voice_reply("Landing timed out.")

    elif intent == "rtl":
        if drone.return_to_launch():
            log_event(captured_text, intent, confidence, "MODE_RTL")
            voice_reply("Returning home.")
        else:
            voice_reply("Return to launch failed.")

    elif intent == "loiter":
        if drone.set_loiter():
            log_event(captured_text, intent, confidence, "MODE_LOITER")
            voice_reply("Loitering.")
        else:
            voice_reply("Loiter blocked.")

    elif intent == "position_hold":
        if drone.set_position_hold():
            log_event(captured_text, intent, confidence, "MODE_POSHOLD")
            voice_reply("Position hold active.")
        else:
            voice_reply("Position hold blocked.")

    elif intent == "hover":
        drone.hover()
        log_event(captured_text, intent, confidence, "HOVER")
        voice_reply("Hovering.")

    elif intent == "move_forward":
        value = float(brain.extract_number(captured_text) or 2.0)
        # FIXED: Speed is held safe at 2.0 m/s; extracted voice values apply directly as duration
        if drone.move_forward(speed=2.0, duration=value):
            log_event(captured_text, intent, confidence, f"FORWARD_{value}S")
            voice_reply(f"Moving forward for {value:.0f} seconds.")
        else:
            voice_reply("Forward movement blocked.")

    elif intent == "move_backward":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_backward(speed=2.0, duration=value):
            log_event(captured_text, intent, confidence, f"BACKWARD_{value}S")
            voice_reply(f"Moving backward for {value:.0f} seconds.")
        else:
            voice_reply("Backward movement blocked.")

    elif intent == "move_left":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_left(speed=2.0, duration=value):
            log_event(captured_text, intent, confidence, f"LEFT_{value}S")
            voice_reply(f"Strafe left for {value:.0f} seconds.")
        else:
            voice_reply("Left movement blocked.")

    elif intent == "move_right":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_right(speed=2.0, duration=value):
            log_event(captured_text, intent, confidence, f"RIGHT_{value}S")
            voice_reply(f"Strafe right for {value:.0f} seconds.")
        else:
            voice_reply("Right movement blocked.")

    # ── 6. CLIMB CONTROL (FIXED ARGS) ─────────────────────────────────
    elif intent == "ascend":
        value = float(brain.extract_number(captured_text) or 3.0)
        if drone.ascend(distance=value):
            log_event(captured_text, intent, confidence, f"ASCEND_{value}M")
            voice_reply(f"Climbing {value:.0f} meters.")
        else:
            voice_reply("Ascend sequence failed.")

    elif intent == "descend":
        value = float(brain.extract_number(captured_text) or 3.0)
        if drone.descend(distance=value):
            log_event(captured_text, intent, confidence, f"DESCEND_{value}M")
            voice_reply(f"Descending {value:.0f} meters.")
        else:
            voice_reply("Descend sequence blocked.")

    # ── 7. STEERING & YAW CONTROLS ────────────────────────────────────
    elif intent == "rotate_left":
        value = float(brain.extract_number(captured_text) or 90.0)
        if drone.rotate_left(degrees=value):
            log_event(captured_text, intent, confidence, f"ROTATE_LEFT_{value}DEG")
            voice_reply(f"Turning left {value:.0f} degrees.")

    elif intent == "rotate_right":
        value = float(brain.extract_number(captured_text) or 90.0)
        if drone.rotate_right(degrees=value):
            log_event(captured_text, intent, confidence, f"ROTATE_RIGHT_{value}DEG")
            voice_reply(f"Turning right {value:.0f} degrees.")

    elif intent == "set_heading":
        value = float(brain.extract_number(captured_text) or 0.0)
        if drone.set_heading(heading_degrees=value):
            log_event(captured_text, intent, confidence, f"SET_HEADING_{value}")
            voice_reply(f"Setting heading layout to {value:.0f} degrees.")

    elif intent == "set_speed":
        value = float(brain.extract_number(captured_text) or 5.0)
        if drone.set_groundspeed(value):
            log_event(captured_text, intent, confidence, f"SPEED_{value}ms")
            voice_reply(f"Speed {value:.0f} metres per second.")
        else:
            voice_reply("Speed change failed.")

    elif intent == "save_waypoint":
        if drone.save_waypoint():
            log_event(captured_text, intent, confidence, "WAYPOINT_SAVED")
            voice_reply("Waypoint saved.")
        else:
            voice_reply("Could not save waypoint.")

    elif intent == "goto_waypoint":
        if drone.goto_waypoint():
            log_event(captured_text, intent, confidence, "GOTO_WAYPOINT")
            voice_reply("Navigating to waypoint.")
        else:
            voice_reply("No waypoint saved or navigation failed.")

    elif intent == "export_mission":
        if drone.export_mission():
            log_event(captured_text, intent, confidence, "MISSION_EXPORTED")
            voice_reply("Mission exported.")
        else:
            voice_reply("Mission export failed.")

    # ── Status queries ─────────────────────────────────────────────────
    elif intent == "get_altitude":
        alt = drone.get_altitude()
        if alt is not None:
            log_event(captured_text, intent, confidence, f"ALT_QUERY_{alt:.1f}m")
            voice_reply(f"Altitude {alt:.1f} metres.")
        else:
            voice_reply("Altitude unavailable.")

    elif intent == "get_heading":
        hdg = drone.get_heading()
        if hdg is not None:
            log_event(captured_text, intent, confidence, f"HDG_QUERY_{hdg:.0f}deg")
            voice_reply(f"Heading {hdg:.0f} degrees.")
        else:
            voice_reply("Heading unavailable.")

    elif intent == "get_battery":
        batt = drone.get_battery()
        if batt and batt.get("remaining_pct", -1) != -1:
            pct = batt["remaining_pct"]
            log_event(captured_text, intent, confidence, f"BATT_QUERY_{pct}pct")
            voice_reply(f"Battery {pct} percent.")
        else:
            voice_reply("Battery unavailable.")

    elif intent == "get_location":
        loc = drone.get_location()
        if loc:
            log_event(captured_text, intent, confidence, "LOC_QUERY")
            voice_reply(
                f"Latitude {loc['lat']:.4f}, "
                f"longitude {loc['lon']:.4f}, "
                f"altitude {loc['alt_relative']:.1f} metres."
            )
        else:
            voice_reply("Location unavailable.")

    elif intent == "get_mode":
        mode = drone.get_current_mode()
        if mode:
            log_event(captured_text, intent, confidence, f"MODE_QUERY_{mode}")
            voice_reply(f"Flight mode {mode}.")
        else:
            voice_reply("Flight mode unavailable.")

    elif intent == "get_gps":
        gps = drone.get_gps_status()
        if gps:
            log_event(captured_text, intent, confidence,
                      f"GPS_QUERY_{gps['satellites_visible']}sats")
            voice_reply(
                f"GPS fix {gps['fix_type']}, "
                f"{gps['satellites_visible']} satellites."
            )
        else:
            voice_reply("GPS unavailable.")

    else:
        log_event(captured_text, intent, confidence, "UNHANDLED_INTENT")
        voice_reply("No action mapped for that command.")


# ======================================================================
# Main pipeline
# ======================================================================

def run_central_pipeline():
    os.system('cls' if os.name == 'nt' else 'clear')

    brain = DroneNLPEngine()
    drone = MAVLinkDroneDriver(connection_string="udp:127.0.0.1:14551")

    console.print("\n[bold green]=== VOX-FLIGHT SYSTEM INTEGRATION PIPELINE ACTIVE ===[/bold green]")

    # ── Live voice mode ────────────────────────────────────────────────
    if not TEST_MODE:

        live = Live(console=console, refresh_per_second=10, screen=False)
        live.start()

        recorder_config = {
            'spinner':                        False,
            'model':                          'tiny.en',
            'compute_type':                   'int8',
            'download_root':                  None,
            'language':                       'en',

            # VAD tuning
            'silero_sensitivity':             0.05,
            'webrtc_sensitivity':             3,
            'post_speech_silence_duration':   0.30,
            'min_length_of_recording':        0.5,
            'min_gap_between_recordings':     0,

            # Performance
            'enable_realtime_transcription':  False,
            'silero_deactivity_detection':    True,
            'early_transcription_on_silence': 0,

            # Decoding
            'beam_size':                      1,
            'no_log_file':                    True,
            'silero_use_onnx':                True,
            'faster_whisper_vad_filter':      False,
        }

        recorder = AudioToTextRecorder(**recorder_config)

        live.update(Panel(
            Text("Ready for voice commands...", style="green bold"),
            title="[bold cyan]VOX-FLIGHT Active[/bold cyan]",
            border_style="bold green",
        ))

        console.print("[bold green]>> RUNNING IN LIVE VOICE MODE <<[/bold green]")
        voice_reply("Ready.")   # Keep startup TTS short to minimise initial mute window

        def on_transcription(text: str):
            """Called by RealtimeSTT each time a complete utterance is ready."""

            # ── TTS echo suppression (timestamp-based, race-condition safe) ──
            if _is_muted():
                print(f"[MUTED] Dropped echo: \"{text.strip()}\"")
                return

            text = text.strip().lstrip(".")
            if not text:
                return
            text = text[0].upper() + text[1:]

            live.update(Panel(
                Text().append("Command: ", style="bold green").append(text, style="bold yellow"),
                title="[bold cyan]Voice Control Loop[/bold cyan]",
                border_style="bold cyan",
            ))

            print(f'[STT] Transcribed: "{text}"')
            dispatch_command(text, brain, drone)

        try:
            while True:
                recorder.text(on_transcription)

        except KeyboardInterrupt:
            live.stop()
            console.print("[bold red]Control Loop Stopped. Exiting...[/bold red]")

    # ── Test / keyboard mode ───────────────────────────────────────────
    else:
        console.print("[bold yellow]>> RUNNING IN KEYBOARD TEST MODE <<[/bold yellow]")
        while True:
            try:
                raw = input("\n[ TEST MODE ] Enter command: ").lower().strip()
                if not raw:
                    continue
                dispatch_command(raw, brain, drone)
            except KeyboardInterrupt:
                console.print("[bold red]Test mode stopped.[/bold red]")
                break

    # ── Cleanup ────────────────────────────────────────────────────────
    drone.close_connection()
    console.print("[bold green][SYSTEM] Pipeline terminated cleanly.[/bold green]")


if __name__ == "__main__":
    run_central_pipeline()