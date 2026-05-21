import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DroneNLPBrain:
    def __init__(self):
        # Reusable training dataset configuration
        self.training_data = {
            "arm": ["arm the motors", "start engines", "spin up propellers", "turn on motors", "arm vehicle"],
            "takeoff": ["takeoff", "take off", "launch drone", "fly up", "go airborne", "takeoff to 5 meters", "lift off"],
            "rtl": ["return to launch", "come back home", "fly back to base", "go home", "abort and return"],
            "loiter": ["stay right there", "hover in place", "freeze", "stop moving", "hold position", "hold here"],
            "land": ["land right now", "descend and land", "touchdown", "land the drone", "put it on the ground"],
            "move_forward": ["move forward", "go forward", "fly forward", "forward"],
            "move_backward": ["move backward", "go back", "fly backward", "back up"],
            "move_left": ["move left", "go left", "fly left", "strafe left"],
            "move_right": ["move right", "go right", "fly right", "strafe right"],
            "change_alt_absolute": ["climb to", "go to altitude", "set altitude to", "fly to height", "change altitude to"],
            "change_alt_relative_up": ["go up", "climb higher", "increase altitude", "fly higher by", "climb up"],
            "change_alt_relative_down": ["go down", "fly lower", "decrease altitude", "descend by", "go lower by"],
            "unknown_noise": ["what is the weather", "hello there", "testing microphone", "look at that bird", "nice day"]
            

        }
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(class_weight='balanced')
        self._train_model()

    def _train_model(self):
        phrases = []
        labels = []
        for intent, examples in self.training_data.items():
            for example in examples:
                phrases.append(example.lower())
                labels.append(intent)

        X_train = self.vectorizer.fit_transform(phrases)
        self.classifier.fit(X_train, np.array(labels))
        print("[NLP BRAIN] Model trained successfully with dynamic command metrics.")

    def analyze_phrase(self, text):
        """Classifies text and filters weak probability guesses."""
        input_vector = self.vectorizer.transform([text.lower().strip()])
        probabilities = self.classifier.predict_proba(input_vector)[0]
        max_idx = np.argmax(probabilities)
        confidence = probabilities[max_idx]
        predicted_intent = self.classifier.classes_[max_idx]
        
        if confidence < 0.58 or predicted_intent == "unknown_noise":
            return "invalid_command", confidence
        return predicted_intent, confidence

    @staticmethod
    def extract_number(text, default_value=5):
        """Utility parsing integers out of the spoken text string."""
        numbers = re.findall(r'\d+', text)
        if numbers:
            return float(numbers[0])
        return float(default_value)