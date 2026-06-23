"""
nlp_brain.py — Shared NLP engine for drone and plane voice control.

Provides fuzzy intent matching via rapidfuzz and numeric extraction
from natural speech. Covers all intents used by both vehicle types.
"""

import re
import warnings
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DroneNLPEngine:
    def __init__(self):
        print("[NLP BRAIN] Initializing Voice NLP Engine...")

        self.match_threshold = 75.0

        self.command_dictionary = {

            # ── Shared: arm / disarm ──────────────────────────────────────
            "arm": [
                "arm the drone", "arm engines", "turn on motors",
                "system unlock", "arm", "arm motors",
                "arm the plane", "arm the aircraft", "arm propulsion",
            ],
            "disarm": [
                "disarm the drone", "disarm engines", "cut power","disarm",
                "turn off motors", "kill the motors", "disarm motors",
                "motors off", "disarm the plane", "disarm aircraft",
            ],

            # ── Shared: takeoff ───────────────────────────────────────────
            "takeoff": [
                "take off", "launch drone", "start flying",
                "lift off", "launch", "take off now",
                "launch aircraft", "start the flight", "begin takeoff",
            ],

            # ── Shared: RTL / land ────────────────────────────────────────
            "rtl": [
                "return to launch", "come back home", "go home",
                "return home", "rtl", "land at base", "fly home",
            ],
            "land": [
                "land", "land immediately", "come down",
                "land now", "set down", "descend and land",
                "touch down", "bring it down",
            ],

            # ── Shared: navigation ────────────────────────────────────────
            "loiter": [
                "loiter", "hold position", "hold", "stop here",
                "stay put", "maintain position", "orbit", "circle",
            ],
            "goto_waypoint": [
                "go to waypoint", "fly to waypoint", "return to waypoint",
                "navigate to waypoint", "head to waypoint",
            ],
            "save_waypoint": [
                "save waypoint", "mark current location", "save this spot",
                "mark position", "record spot",
            ],
            "export_mission": [
                "save flight plan", "export waypoints", "download mission file",
                "save mission text", "export mission",
            ],

            # ── Shared: telemetry queries ─────────────────────────────────
            "get_altitude": [
                "what is my altitude", "how high am i", "altitude check",
                "current altitude", "check altitude",
            ],
            "get_heading": [
                "what direction am i facing", "what is my heading",
                "compass heading", "which way am i pointing", "current heading",
            ],
            "get_battery": [
                "check battery", "battery level", "how much battery do i have",
                "battery status", "battery remaining",
            ],
            "get_location": [
                "where am i", "current position", "what are my coordinates",
                "gps coordinates", "current location",
            ],
            "get_mode": [
                "what mode am i in", "current flight mode",
                "which mode", "what mode is the drone in",
                "what mode is the plane in",
            ],
            "get_gps": [
                "check gps", "gps status", "satellite count",
                "how is my gps signal", "gps fix",
            ],

            # ── Shared: speed ─────────────────────────────────────────────
            "set_speed": [
                "set speed", "change speed", "speed up", "slow down",
                "increase speed", "decrease speed", "ground speed",
                "set airspeed", "cruise speed", "airspeed",
            ],

            # ── Shared: heading ───────────────────────────────────────────
            "set_heading": [
                "set heading", "face direction", "point to", "heading",
                "change heading", "set bearing", "face north", "face south",
                "face east", "face west",
            ],

            # ── Shared: emergency ─────────────────────────────────────────
            "emergency_safe": [
                "trigger safe state", "abort mission", "emergency stop",
                "stop immediately", "safe state", "emergency abort",
            ],

            # ── DRONE-ONLY: hover / position hold ────────────────────────
            "hover": [
                "hover", "stop moving", "freeze", "stop all movement",
            ],
            "position_hold": [
                "position hold", "pos hold", "lock position",
            ],

            # ── DRONE-ONLY: lateral movement ──────────────────────────────
            "move_forward": [
                "move forward", "go forward", "fly forward", "forward", "advance",
            ],
            "move_backward": [
                "move backward", "go backward", "reverse", "backward", "fly back",
            ],
            "move_left": [
                "move left", "fly left", "left", "go left", "strafe left",
            ],
            "move_right": [
                "move right", "fly right", "right", "go right", "strafe right",
            ],

            # ── DRONE-ONLY: altitude by distance ─────────────────────────
            "ascend": [
                "ascend", "go up", "fly higher", "go higher", "climb",
                "increase altitude", "rise up", "gain altitude",
            ],
            "descend": [
                "descend", "fly lower", "go down", "go lower", "sink",
                "decrease altitude", "lose altitude", "drop altitude",
            ],

            # ── DRONE-ONLY: yaw ───────────────────────────────────────────
            "rotate_left": [
                "rotate left", "spin left", "yaw left", "turn counterclockwise",
                "rotate counterclockwise",
            ],
            "rotate_right": [
                "rotate right", "spin right", "yaw right", "turn clockwise",
                "rotate clockwise",
            ],

            "start_mission": [
                "start mission", "begin mission", "execute mission",
                "run mission", "start automatic flight", "start auto flight",
                "execute waypoints", "run waypoints"
            ],

            # ── PLANE-ONLY: mode shortcuts ────────────────────────────────
            "set_cruise": [
                "cruise mode", "set cruise", "fly cruise", "enable cruise",
                "cruise control",
            ],
            "set_fbwa": [
                 "fly by wire", "fbwa", "fly by wire a",
            ],
            "set_auto": [
                "auto mode", "start mission", "follow mission", "autonomous mode",
                "execute mission","run flight paln"
            ],
            "set_guided": [
                "guided mode", "enable guided", "guided flight",
            ],
            "set_manual":[
                "manual", "manual mode", "direct control"
            ],

            # ── PLANE-ONLY: altitude by target ────────────────────────────
            "climb_to": [
                "climb to", "ascend to", "go up to altitude", "climb to altitude",
                "go up to", "reach altitude",
            ],
            "descend_to": [
                "descend to", "go down to", "reduce altitude to",
                "drop to altitude", "lower to",
            ],

            # ── PLANE-ONLY: turning ───────────────────────────────────────
            "turn_left": [
                "turn left", "bank left", "left turn", "veer left",
            ],
            "turn_right": [
                "turn right", "bank right", "right turn", "veer right",
            ],
            
           
        }

        print("[NLP BRAIN] Command dictionary loaded — "
              f"{len(self.command_dictionary)} intents registered.")

    def analyze_phrase(self, text: str):
        return self.match_intent(text)

    def match_intent(self, text: str):
        text = text.lower().strip()
        best_command = "invalid_command"
        best_score   = 0.0

        for command_action, target_phrases in self.command_dictionary.items():
            result = process.extractOne(text, target_phrases, scorer=fuzz.WRatio)
            if result:
                score = result[1]
                if score > best_score:
                    best_score   = score
                    best_command = command_action

        if best_score >= self.match_threshold:
            return best_command, best_score / 100.0
        return "invalid_command", best_score / 100.0

    def extract_number(self, text: str):
        """Extract the first number from spoken text (words or digits)."""
        word_to_num = {
            "one":     1,  "two":     2,  "three": 3,
            "four":    4,  "five":    5,  "six":   6,
            "seven":   7,  "eight":   8,  "nine":  9,
            "ten":    10,  "fifteen": 15, "twenty": 20,
            "thirty": 30,  "forty":   40, "fifty": 50,
            "sixty":  60,  "seventy": 70, "eighty": 80,
            "ninety": 90,  "hundred": 100,
        }
        text_lower = text.lower()
        for word, val in word_to_num.items():
            if word in text_lower:
                return float(val)

        numbers = re.findall(r'\d+', text)
        return float(numbers[0]) if numbers else None
