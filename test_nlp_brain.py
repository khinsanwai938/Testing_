import re
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel


UNIVERSAL_AIRCRAFT_PROMPT = (
    "arm disarm takeoff land loiter RTL return to launch plane drone flight "
    "move forward move backward move left move right climb descend pitch up pitch down "
    "speed up slow down increase airspeed turn left turn right bank left bank right "
    "hold position fly up go home stabilized fbwa glide meters seconds"
)

class DroneNLPBrain:
    """
    Fuzzy NLP engine for parsing voice commands into actionable drone intents.
    """

    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        print("Initializing Faster-Whisper Engine...")

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.match_threshold = 80.0

        self.command_dictionary = {
            "arm": [
                "arm",
                "arm the drone",
                "arm the plane",
                "arm engines",
                "turn on motors",
                "system unlock",
                "start the plane",
                "start the drone",
                "start up the drone",
                "unlock flight controller",
                "system arm",
                "turn on the drone",
                "get the props spinning",
            ],
            "disarm": [
                "disarm",
                "disarm the drone",
                "disarm the plane",
                "disarm engines",
                "cut power",
                "turn off motors",
                "kill the motors",
                "stop the drone",
                "stop the plane",
                "stop the propellers",
                "lock system",
                "cut the engines",
                "shut down the engines",
                "power down the drone",
                "turn off the power",
            ],
            "takeoff": [
                "take off",
                "fly",
                "fly the drone",
                "fly the plane",
                "launch drone",
                "start flying",
                "go up",
                "lift off",
                "begin flight",
                "fly up",
                "fly up into the air",
                "lift off the ground",
            ],
            "rtl": [
                "rtl",
                "return to launch",
                "come back home",
                "go home",
                "return home",
                "return station",
                "land at base",
                "fly back",
                "return to base",
                "go to home position",
                "bring the drone back",
                "time to come home",
                "bring it back",
            ],
            "land": [
                "land",
                "land immediately",
                "set down",
                "touch down",
                "bring it down",
                "descend and land",
            ],
            "forward": [
                "fly forward",
                "move forward",
                "go forward",
                "forward",
            ],
            "backward": [
                "fly backward",
                "move backward",
                "go backward",
                "backward",
            ],
            "left": [
                "fly left",
                "move left",
                "go left",
                "left",
            ],
            "right": [
                "fly right",
                "move right",
                "go right",
                "right",
            ],
            "hover": [
                "hover",
                "hold position",
                "stay in place",
                "stay there",
                "stop moving",
                "hold steady",
                "hold on",
                "maintain position",
            ],
            "ascend": [
                "ascend",
                "go up",
                "climb",
                "gain altitude",
                "increase altitude",
                "rise",
                "fly up",
            ],
            "descend": [
                "descend",
                "go down",
                "lose altitude",
                "decrease altitude",
                "sink",
                "fly down",
            ],
            "rotate_left": [
                "yaw left",
                "turn left",
                "rotate left",
                "spin left",
                "rotate counterclockwise",
                "turn to the left",
            ],
            "rotate_right": [
                "yaw right",
                "turn right",
                "rotate right",
                "spin right",
                "rotate clockwise",
                "turn to the right",
            ],
        }

        # Pre-build a flat lookup so analyze_phrase() doesn't
        # re-scan the nested dict on every call.
        # Maps each phrase string → its intent label.
        self._phrase_to_intent: dict[str, str] = {
            phrase: intent
            for intent, phrases in self.command_dictionary.items()
            for phrase in phrases
        }
        self._all_phrases: list[str] = list(self._phrase_to_intent.keys())

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe_audio(self, audio_buffer, final: bool = False) -> str:
        """
        Transcribes an audio buffer (float32 numpy array, 16 kHz mono)
        using the Faster-Whisper model.

        Args:
            audio_buffer: numpy float32 array of audio samples.
            final:        True for the last chunk (uses larger beam size).

        Returns:
            Transcribed text string (may be empty if nothing detected).
        """
        beam_size = 3 if final else 2
        segments, _ = self.model.transcribe(
            audio_buffer,
            language="en",
            beam_size=beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return " ".join([seg.text for seg in segments]).strip()

    # ------------------------------------------------------------------
    # Intent analysis
    # ------------------------------------------------------------------

    def analyze_phrase(self, text: str) -> tuple[str, float]:
        """
        Matches transcribed text to the closest drone intent using
        fuzzy string matching (rapidfuzz WRatio scorer).

        Args:
            text: Transcribed speech string.

        Returns:
            Tuple of (intent: str, confidence: float 0 - 1).
            intent is "invalid_command" when no match exceeds the threshold.
        """
        text = text.lower().strip()

        result = process.extractOne(text, self._all_phrases, scorer=fuzz.WRatio)

        if result:
            best_phrase, best_score, _ = result
            if best_score >= self.match_threshold:
                intent = self._phrase_to_intent[best_phrase]
                return intent, best_score / 100.0

        return "invalid_command", 0.0

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def extract_number(self, text: str) -> float | None:
        """
        Extracts the first integer or decimal number found in text.

        Added word-to-digit mapping so spoken numbers like
        "take off to twenty metres" are handled correctly.
        Whisper often transcribes small numbers as words, not digits,
        which made the original regex return None for common commands.

        Args:
            text: Raw transcribed string.

        Returns:
            Float value if a number is found, otherwise None.
        """
        WORD_TO_NUM = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "hundred": 100,
        }

        # Digit-first: try to find a numeric literal.
        digit_matches = re.findall(r'\d+(?:\.\d+)?', text)
        if digit_matches:
            return float(digit_matches[0])

        # Word fallback: scan for the first recognised number word.
        for word in text.lower().split():
            word = word.strip(".,!?")
            if word in WORD_TO_NUM:
                return float(WORD_TO_NUM[word])
            
    def extract_duration(self, text: str) -> float | None:
        """Extracts the first duration in seconds found in text, 
        handling formats like "5 seconds", "2.5 minutes", or "one minute".
        """
        match = re.search(r'(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?)', text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if 'second' in unit:
                return value
            elif 'minute' in unit:
                return value * 60
            elif 'hour' in unit:
                return value * 3600

        return None