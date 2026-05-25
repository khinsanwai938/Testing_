import re

class DroneNLPBrain:
    def __init__(self):
        print("[NLP BRAIN] Rule-based parsing parameters loaded successfully.")

    def analyze_phrase(self, text):
        """Analyzes text input strings and maps them to concrete drone intents."""
        text = text.lower().strip()
        
        # Keyword mapping rules
        if "arm" in text:
            return "arm", 1.0
        elif "disarm" in text:
            return "disarm", 1.0
        elif "takeoff" in text or "launch" in text or "fly up" in text:
            return "takeoff", 1.0
        elif "forward" in text:
            return "move_forward", 1.0
        elif "backward" in text or "reverse" in text:
            return "move_backward", 1.0
        elif "left" in text:
            return "move_left", 1.0
        elif "right" in text:
            return "move_right", 1.0
        elif "rtl" in text or "return" in text or "home" in text:
            return "rtl", 1.0
        elif "loiter" in text or "hold" in text:
            return "loiter", 1.0
        elif "land" in text:
            return "land", 1.0
        
        return "invalid_command", 0.0

    def extract_number(self, text):
        """Finds any standalone number digits inside the text command string."""
        numbers = re.findall(r'\d+', text)
        if numbers:
            return float(numbers[0])
        return None