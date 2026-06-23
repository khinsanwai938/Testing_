import re
import warnings
import numpy as np
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

DRONE_PROMPT = (
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
    "meters seconds faster slower"
)


class DroneNLPEngine:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        print("[NLP BRAIN] Initializing Faster-Whisper Engine...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.match_threshold = 75.0
        
        self.command_dictionary = {

            # ── driver: arm_vehicle() ─────────────────────────────────
            "arm": [
                "arm", "arm the drone", "arm the plane", "arm motors",
                "arm engines", "turn on motors", "turn on the motors",
                "start the drone", "start the plane", "start up",
                "system arm", "system unlock", "unlock the drone",
                "power on motors", "enable motors", "spin up",
                "get the props spinning", "initialize motors",
            ],

            # ── driver: disarm_vehicle() ──────────────────────────────
            "disarm": [
                "disarm", "disarm the drone", "disarm the plane",
                "disarm motors", "disarm engines", "cut power",
                "turn off motors", "kill the motors", "motors off",
                "shut down motors", "power down", "stop the motors",
                "lock the system", "disable motors",
            ],

            # ── driver: execute_takeoff() ─────────────────────────────
            "takeoff": [
                "take off", "takeoff", "launch", "launch drone",
                "launch the drone", "lift off", "lift off the ground",
                "start flying", "begin flight", "fly up",
                "fly up into the air", "go airborne", "get airborne",
                "take off to", "take off to five metres",
                "take off to ten metres",
            ],

            # ── driver: land() ────────────────────────────────────────
            "land": [
                "land", "land now", "land immediately", "come down",
                "set down", "touch down", "bring it down",
                "descend and land", "land the drone", "put it down",
                "ground the drone",
            ],

            # ── driver: return_to_launch() ────────────────────────────
            "rtl": [
                "rtl", "return to launch", "return to home",
                "go home", "come home", "fly home", "come back home",
                "return home", "land at base", "head home",
                "bring it back", "bring the drone back",
                "return to base", "go to home position",
            ],

            # ── driver: set_loiter() ──────────────────────────────────
            "loiter": [
                "loiter", "loiter mode", "circle",
                "hover in place", "start loitering",
                "enter loiter", "switch to loiter",
            ],

            # ── driver: set_position_hold() ───────────────────────────
            "position_hold": [
                "position hold", "poshold", "hold position",
                "stay here", "maintain position", "stay put",
                "freeze", "stop here", "hold",
            ],

            # ── driver: hover() ───────────────────────────────────────
            "hover": [
                "hover", "stop moving", "stop", "stay",
                "hold steady", "hold on", "stay in place",
                "stay there", "zero velocity", "stand by",
            ],

            # ── driver: move_forward() ────────────────────────────────
            "move_forward": [
                "move forward", "go forward", "fly forward",
                "forward", "advance", "head forward",
                "move ahead", "go ahead", "push forward",
            ],

            # ── driver: move_backward() ───────────────────────────────
            "move_backward": [
                "move backward", "go backward", "fly backward",
                "backward", "reverse", "move back", "go back",
                "pull back", "fly back",
            ],

            # ── driver: move_left() ───────────────────────────────────
            "move_left": [
                "move left", "go left", "fly left",
                "left", "strafe left", "slide left",
                "shift left", "bank left",
            ],

            # ── driver: move_right() ──────────────────────────────────
            "move_right": [
                "move right", "go right", "fly right",
                "right", "strafe right", "slide right",
                "shift right", "bank right",
            ],

            # ── driver: ascend() ─────────────────────────────────────
            "ascend": [
                "ascend", "go up", "climb", "fly up",
                "gain altitude", "increase altitude", "rise",
                "climb up", "go higher", "move up",
                "ascend five metres", "climb ten metres",
            ],

            # ── driver: descend() ────────────────────────────────────
            "descend": [
                "descend", "go down", "fly down", "drop",
                "lose altitude", "decrease altitude", "sink",
                "lower", "come down slowly", "move down",
                "descend five metres", "go down ten metres",
            ],

            # ── driver: rotate_left() ────────────────────────────────
            "rotate_left": [
                "rotate left", "yaw left", "turn left",
                "spin left", "rotate counterclockwise",
                "turn to the left", "bank left",
                "rotate left ninety degrees",
            ],

            # ── driver: rotate_right() ───────────────────────────────
            "rotate_right": [
                "rotate right", "yaw right", "turn right",
                "spin right", "rotate clockwise",
                "turn to the right", "bank right",
                "rotate right ninety degrees",
            ],

            # ── driver: set_heading() ────────────────────────────────
            "set_heading": [
                "set heading", "face north", "face south",
                "face east", "face west", "heading north",
                "heading south", "heading east", "heading west",
                "turn to north", "turn to south",
                "turn to east", "turn to west",
                "point north", "compass heading",
                "turn to degrees", "face degrees",
            ],

            # ── driver: change_altitude_absolute() ───────────────────
            "change_altitude": [
                "change altitude", "go to altitude",
                "set altitude", "altitude to",
                "fly at altitude", "climb to",
                "set height to", "altitude change",
            ],

            # ── driver: set_airspeed() ────────────────────────────────
            "set_airspeed": [
                "set airspeed", "airspeed", "set air speed",
                "airspeed to", "change airspeed",
                "set airspeed to five", "set airspeed to ten",
            ],

            # ── driver: set_groundspeed() ─────────────────────────────
            "set_groundspeed": [
                "set speed", "groundspeed", "set ground speed",
                "go faster", "slow down", "speed up",
                "fly at speed", "set speed to",
                "increase speed", "decrease speed",
                "set groundspeed to five", "set speed to ten",
            ],

            # ── driver: goto_waypoint() ───────────────────────────────
            "goto_waypoint": [
                "go to waypoint", "fly to waypoint",
                "navigate to waypoint", "head to waypoint",
                "return to waypoint", "go to saved waypoint",
                "fly to saved location", "navigate to saved spot",
            ],

            # ── driver: get_altitude() ────────────────────────────────
            "query_altitude": [
                "what is my altitude", "how high am i",
                "altitude check", "current altitude",
                "what altitude am i at", "tell me my altitude",
                "check altitude", "altitude status",
            ],

            # ── driver: get_heading() ─────────────────────────────────
            "query_heading": [
                "what is my heading", "current heading",
                "compass heading", "which way am i facing",
                "what direction am i going", "heading check",
                "check heading", "what direction",
            ],

            # ── driver: get_groundspeed() ─────────────────────────────
            "query_speed": [
                "how fast am i going", "current speed",
                "what is my speed", "ground speed",
                "check speed", "speed status",
                "what speed am i flying at",
            ],

            # ── driver: get_battery() ────────────────────────────────
            "query_battery": [
                "check battery", "battery level", "battery status",
                "how much battery", "battery check",
                "what is my battery", "battery percentage",
                "how much power do i have",
            ],

            # ── driver: get_gps_status() ─────────────────────────────
            "query_gps": [
                "check gps", "gps status", "satellite count",
                "how is my gps", "gps signal",
                "gps check", "how many satellites",
            ],

            # ── driver: get_current_mode() ───────────────────────────
            "query_mode": [
                "what mode am i in", "current flight mode",
                "which mode", "what mode is the drone in",
                "flight mode check", "current mode",
            ],

            # ── driver: get_location() ───────────────────────────────
            "query_location": [
                "where am i", "current position", "my location",
                "what are my coordinates", "gps coordinates",
                "current coordinates", "position check",
                "what is my position",
            ],

            # ── driver: trigger_emergency_safe_state() ───────────────
            "emergency_safe": [
                "emergency", "emergency stop", "abort",
                "abort mission", "trigger safe state",
                "safe state", "emergency abort", "stop immediately",
                "emergency land", "cut everything",
                "kill switch", "mayday",
            ],

            # ── not a driver method — handled in main.py ──────────────
            "save_waypoint": [
                "save waypoint", "mark position", "mark current location",
                "save this spot", "record spot", "mark this location",
                "save location", "drop a pin", "mark waypoint",
            ],

            "export_mission": [
                "export mission", "save flight plan", "export waypoints",
                "download mission", "save mission", "export flight plan",
                "save waypoints to file",
            ],
        }
    print("[NLP BRAIN] Fuzzy parsing rules and dictionaries loaded successfully.")

    def transcribe_audio(self, audio_buffer: np.ndarray, final: bool = False) -> str:
        segments, info = self.model.transcribe(
            audio_buffer,
            language="en",
            beam_size=5,
            best_of=5 if final else 1,
            initial_prompt=DRONE_PROMPT,
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.55,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=150)
        )

        no_speech_prob = getattr(info, 'no_speech_prob', 0.0)
        if no_speech_prob > 0.55:
            print(f"[NLP BRAIN] Low speech confidence ({no_speech_prob:.2f}) — discarding segment.")
            return ""

        return " ".join(seg.text for seg in segments).strip()

    def analyze_phrase(self, text: str):
        return self.match_intent(text)

    def match_intent(self, text: str):
        text = text.lower().strip()
        best_command = "invalid_command"
        best_score = 0.0

        for command_action, target_phrases in self.command_dictionary.items():
            result = process.extractOne(text, target_phrases, scorer=fuzz.WRatio)
            if result:
                # Direct index accessing prevents version mismatch tuple crashes entirely
                score = result[1]
                if score > best_score:
                    best_score = score
                    best_command = command_action

        if best_score >= self.match_threshold:
            return best_command, best_score / 100.0
        return "invalid_command", best_score / 100.0

    def extract_number(self, text: str):
        # Comprehensive vocal flight operational mapping dictionary
        WORD_TO_NUM = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "hundred": 100,
        }
        for word, val in WORD_TO_NUM.items():
            if word in text.lower():
                return float(val)
                
        numbers = re.findall(r'\d+', text)
        return float(numbers[0]) if numbers else None