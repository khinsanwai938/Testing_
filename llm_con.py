import re
import warnings
import numpy as np
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

DRONE_PROMPT = (
    "arm disarm takeoff land loiter RTL return to launch "
    "move forward move backward move left move right "
    "hold position fly up go home camera zoom record video take photo "
    "stabilize altitude hold degrees meters waypoint mark spot emergency abort safe state"
)

class DroneNLPEngine:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        print("[NLP BRAIN] Initializing Faster-Whisper Engine...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.match_threshold = 75.0

        self.command_dictionary = {
            "arm": ["arm the drone", "arm engines", "turn on motors", "system unlock", "arm", "arm motors"],
            "disarm": ["disarm the drone", "disarm engines", "cut power", "turn off motors", "kill the motors", "disarm motors", "motors off"],
            "takeoff": ["take off", "launch drone", "start flying", "go up", "lift off", "launch", "ascend"],
            "rtl": ["return to launch", "come back home", "go home", "return home", "rtl", "land at base", "fly home"],
            "loiter": ["loiter", "hold position", "hold", "stop here", "hover", "stay put", "maintain position"],
            "land": ["land", "land immediately", "come down", "land now", "set down", "descend and land"],
            "move_forward": ["move forward", "go forward", "fly forward", "forward", "advance"],
            "move_backward": ["move backward", "go backward", "reverse", "backward", "move back"],
            "move_left": ["move left", "fly left", "left", "turn left", "go left"],
            "move_right": ["move right", "fly right", "right", "turn right", "go right"],
            "save_waypoint": ["save waypoint", "mark current location", "save this spot", "mark position", "record spot"],
            "goto_waypoint": ["go to waypoint", "fly to waypoint", "return to waypoint", "navigate to waypoint", "head to waypoint"],
            "export_mission": ["save flight plan", "export waypoints", "download mission file", "save mission text", "export mission"],
            "emergency_safe": ["trigger safe state", "abort mission", "emergency stop", "stop immediately", "safe state", "emergency abort"]
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
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50
        }
        for word, val in word_to_num.items():
            if word in text.lower():
                return float(val)
                
        numbers = re.findall(r'\d+', text)
        return float(numbers[0]) if numbers else None