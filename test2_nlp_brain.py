import re
import warnings
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DroneNLPEngine:
    def __init__(self):
        print("[NLP BRAIN] Initializing Drone NLP Engine...")

        self.match_threshold = 75.0

        self.command_dictionary = {
            "arm": [
                "arm the drone", "arm engines", "turn on motors",
                "system unlock", "arm", "arm motors"
            ],
            "disarm": [
                "disarm the drone", "disarm engines", "cut power",
                "turn off motors", "kill the motors", "disarm motors", "motors off"
            ],
            "takeoff": [
                "take off", "launch drone", "start flying",
                "lift off", "launch"
            ],
            "rtl": [
                "return to launch", "come back home", "go home",
                "return home", "rtl", "land at base", "fly home"
            ],
            "loiter": [
                "loiter", "hold position", "hold", "stop here",
                 "stay put", "maintain position"
            ],
            "land": [
                "land", "land immediately", "come down",
                "land now", "set down", "descend and land"
            ],
            "move_forward": [
                "move forward", "go forward", "fly forward", "forward", "advance"
            ],
            "move_backward": [
                "move backward", "go backward", "reverse", "backward", "fly back"
            ],
            "move_left": [
                "move left", "fly left", "left", "turn left", "go left"
            ],
            "move_right": [
                "move right", "fly right", "right", "turn right", "go right"
            ],
            "ascend": [
                "ascend", "go up","fly higher", "go higher", "climb", "increase altitude",
                "rise up", "gain altitude"
            ],
            "descend": [
                "descend", "fly lower", "go down","go lower", "sink", "decrease altitude",
                "lose altitude", "drop altitude"
            ],
            "rotate_left": [
                "rotate left", "spin left", "yaw left", "turn counterclockwise",
                "rotate counterclockwise"
            ],
            "rotate_right": [
                "rotate right", "spin right", "yaw right", "turn clockwise",
                "rotate clockwise"
            ],
            "set_heading": [
                "set heading", "face direction", "point to", "heading",
                "change heading", "set bearing"
            ],
            "set_speed": [
                "set speed", "change speed", "speed up", "slow down",
                "increase speed", "decrease speed", "ground speed"
            ],
            "hover": [
                "hover",  "stop moving",
                "freeze", "stop all movement"
            ],
            "position_hold": [
                "position hold", "pos hold", "lock position"
            ],
            "save_waypoint": [
                "save waypoint", "mark current location", "save this spot",
                "mark position", "record spot"
            ],
            "goto_waypoint": [
                "go to waypoint", "fly to waypoint", "return to waypoint",
                "navigate to waypoint", "head to waypoint"
            ],
            "export_mission": [
                "save flight plan", "export waypoints", "download mission file",
                "save mission text", "export mission"
            ],
            "get_altitude": [
                "what is my altitude", "how high am i", "altitude check",
                "current altitude", "check altitude"
            ],
            "get_heading": [
                "what direction am i facing", "what is my heading",
                "compass heading", "which way am i pointing", "current heading"
            ],
            "get_battery": [
                "check battery", "battery level", "how much battery do i have",
                "battery status", "battery remaining"
            ],
            "get_location": [
                "where am i", "current position", "what are my coordinates",
                "gps coordinates", "current location"
            ],
            "get_mode": [
                "what mode am i in", "current flight mode",
                "which mode", "what mode is the drone in"
            ],
            "get_gps": [
                "check gps", "gps status", "satellite count",
                "how is my gps signal", "gps fix"
            ],
            "emergency_safe": [
                "trigger safe state", "abort mission", "emergency stop",
                "stop immediately", "safe state", "emergency abort"
            ],
        }

        print("[NLP BRAIN] Fuzzy parsing rules and dictionaries loaded successfully.")

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
        # Vocal number mapping for common flight values
        word_to_num = {
            "one":     1,  "two":     2,  "three": 3,
            "four":    4,  "five":    5,  "six":   6,
            "seven":   7,  "eight":   8,  "nine":  9,
            "ten":    10,  "fifteen": 15, "twenty": 20,
            "thirty": 30,  "forty":   40, "fifty": 50,
            "sixty":  60,  "seventy": 70, "eighty": 80,
            "ninety": 90,  "hundred": 100
        }
        text_lower = text.lower()
        for word, val in word_to_num.items():
            if word in text_lower:
                return float(val)

        numbers = re.findall(r'\d+', text)
        return float(numbers[0]) if numbers else None