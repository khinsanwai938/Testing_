import re
import warnings
import numpy as np
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

UNIVERSAL_AIRCRAFT_PROMPT = (
    "arm disarm takeoff land loiter RTL return to launch plane drone flight "
    "move forward move backward move left move right climb descend pitch up pitch down "
    "speed up slow down increase airspeed turn left turn right bank left bank right "
    "hold position fly up go home stabilized fbwa glide meters seconds"
)

class DroneNLPBrain:
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
            "move_right": ["move right", "fly right", "right", "turn right", "go right"]
        }
        print("[NLP BRAIN] Fuzzy parsing rules and dictionaries loaded successfully.")

    def transcribe_audio(self, audio_buffer: np.ndarray, final: bool = False) -> str:
        segments, info = self.model.transcribe(
            audio_buffer,
            language="en",
            beam_size=5,
            best_of=5 if final else 1,
            initial_prompt=UNIVERSAL_AIRCRAFT_PROMPT,
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

        text = " ".join(seg.text for seg in segments).strip()
        return text

    def match_intent(self, text: str):
        """Matches the incoming text to a known drone action keyword."""
        text = text.lower().strip()
        best_command = "invalid_command"
        best_score = 0.0

        for command_action, target_phrases in self.command_dictionary.items():
            result = process.extractOne(text, target_phrases, scorer=fuzz.WRatio)
            if result:
                # RapidFuzz unpack style: (matched_string, score, index)
                matched_str, score, index = result[:3]
                if score > best_score:
                    best_score = score
                    best_command = command_action

        if best_score >= self.match_threshold:
            return best_command, best_score / 100.0
        return "invalid_command", best_score / 100.0

    def extract_number(self, text: str):
        numbers = re.findall(r'\d+', text)
        return float(numbers[0]) if numbers else None