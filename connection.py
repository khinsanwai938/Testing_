import re
import warnings
import numpy as np
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

DRONE_PROMPT = (
    "arm disarm takeoff land loiter RTL return to launch guided poshold "
    "move forward move backward move left move right hover hold position "
    "ascend descend climb go up go down rotate yaw left right "
    "set heading north south east west compass bearing degrees "
    "go to waypoint navigate coordinates latitude longitude "
    "set airspeed set groundspeed speed metres per second throttle "
    "altitude check heading check battery status GPS status "
    "current mode current location where am I how high "
    "emergency abort safe state trigger emergency stop "
    "save waypoint mark position export mission flight plan "
    "FBWA FBWB stabilize auto cruise glide bank circle loiter "
    "flaps throttle up throttle down pitch up pitch down "
    "increase altitude decrease altitude change altitude "
    "plane drone multirotor fixed wing takeoff roll meters seconds"
)

HALLUCINATION_BLACKLIST = frozenset({
    "", ".", "...", "you", "thanks", "thank you", "bye", "goodbye",
    "please", "welcome", "subscribe", "okay", "ok", "uh", "um",
})

# Vehicle type constants
VEHICLE_DRONE     = "drone"     # ArduCopter — multirotor
VEHICLE_PLANE     = "plane"     # ArduPlane  — fixed-wing


class DroneNLPEngine:
    """
    Vehicle-aware fuzzy NLP engine.

    Call set_vehicle_type(VEHICLE_DRONE) or set_vehicle_type(VEHICLE_PLANE)
    after connecting so the right command dictionary is active.
    Defaults to VEHICLE_DRONE for backward compatibility.
    """

    def __init__(self, model_size="small", device="cpu", compute_type="int8",
                 vehicle_type: str = VEHICLE_DRONE):
        print("[NLP BRAIN] Initializing Faster-Whisper Engine...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.match_threshold = 75.0

        self._last_transcript: str = ""
        self._repeat_count: int = 0
        self._max_repeats: int = 1
        self._min_rms_energy: float = 0.01

        # ── Build both dictionaries ───────────────────────────────────
        self._drone_dictionary = self._build_drone_dictionary()
        self._plane_dictionary = self._build_plane_dictionary()

        # ── Shared intents — identical on both vehicle types ──────────
        # These are loaded on top of whichever vehicle dict is active.
        self._shared_dictionary = self._build_shared_dictionary()

        # ── Activate the correct vehicle ──────────────────────────────
        self.command_dictionary: dict = {}
        self._phrase_to_intent: dict[str, str] = {}
        self._all_phrases: list[str] = []
        self.set_vehicle_type(vehicle_type)

    # ======================================================================
    # Vehicle switching
    # ======================================================================

    def set_vehicle_type(self, vehicle_type: str):
        """
        Switch the active command dictionary to match the connected vehicle.
        Call this as soon as you know what vehicle is connected.

            brain.set_vehicle_type(VEHICLE_DRONE)
            brain.set_vehicle_type(VEHICLE_PLANE)
        """
        if vehicle_type not in (VEHICLE_DRONE, VEHICLE_PLANE):
            raise ValueError(f"Unknown vehicle type: {vehicle_type!r}. "
                             f"Use VEHICLE_DRONE or VEHICLE_PLANE.")

        self.vehicle_type = vehicle_type
        base = (self._drone_dictionary if vehicle_type == VEHICLE_DRONE
                else self._plane_dictionary)

        # Merge: shared intents fill in, vehicle dict wins on conflicts
        self.command_dictionary = {**self._shared_dictionary, **base}

        # Rebuild flat lookup
        self._phrase_to_intent = {
            phrase: intent
            for intent, phrases in self.command_dictionary.items()
            for phrase in phrases
        }
        self._all_phrases = list(self._phrase_to_intent.keys())

        print(
            f"[NLP BRAIN] Vehicle type set to '{vehicle_type}' — "
            f"{len(self.command_dictionary)} intents, "
            f"{len(self._all_phrases)} phrases active."
        )

    # ======================================================================
    # Dictionary builders
    # ======================================================================

    def _build_shared_dictionary(self) -> dict:
        """
        Intents that exist on BOTH vehicle types with the same driver call.
        These are merged last so vehicle-specific overrides take priority.
        """
        return {

            # ── driver: arm_vehicle() ─────────────────────────────────
            "arm": [
                "arm", "arm the vehicle", "arm motors", "arm engines",
                "turn on motors", "system arm", "system unlock",
                "enable motors", "power on", "start up",
                "arm the system", "unlock flight controller",
            ],

            # ── driver: disarm_vehicle() ──────────────────────────────
            "disarm": [
                "disarm", "disarm the vehicle", "disarm motors",
                "disarm engines", "cut power", "turn off motors",
                "kill the motors", "motors off", "shutdown motors",
                "power down", "lock the system", "disable motors",
            ],

            # ── driver: land() ────────────────────────────────────────
            "land": [
                "land", "land now", "land immediately",
                "set down", "touch down", "bring it down",
                "descend and land", "put it down",
            ],

            # ── driver: return_to_launch() ────────────────────────────
            "rtl": [
                "rtl", "return to launch", "return to home",
                "go home", "come home", "fly home",
                "come back home", "return home",
                "land at base", "head home",
                "bring it back", "return to base",
                "go to home position",
            ],

            # ── driver: set_loiter() ──────────────────────────────────
            "loiter": [
                "loiter", "loiter mode", "start loitering",
                "enter loiter", "switch to loiter", "circle",
            ],

            # ── driver: goto_waypoint() ───────────────────────────────
            "goto_waypoint": [
                "go to waypoint", "fly to waypoint",
                "navigate to waypoint", "head to waypoint",
                "return to waypoint", "go to saved waypoint",
                "fly to saved location", "navigate to saved spot",
            ],

            # ── driver: set_airspeed() ────────────────────────────────
            "set_airspeed": [
                "set airspeed", "airspeed to", "change airspeed",
                "set air speed", "airspeed",
            ],

            # ── driver: set_groundspeed() ─────────────────────────────
            "set_groundspeed": [
                "set speed", "set ground speed", "groundspeed",
                "fly at speed", "set speed to",
            ],

            # ── driver: change_altitude_absolute() ───────────────────
            "change_altitude": [
                "change altitude", "go to altitude", "set altitude",
                "altitude to", "fly at altitude", "set height to",
            ],

            # ── driver: get_altitude() ────────────────────────────────
            "query_altitude": [
                "what is my altitude", "how high am i",
                "altitude check", "current altitude",
                "check altitude", "altitude status",
                "what altitude am i at",
            ],

            # ── driver: get_heading() ─────────────────────────────────
            "query_heading": [
                "what is my heading", "current heading",
                "compass heading", "which way am i facing",
                "heading check", "check heading", "what direction",
            ],

            # ── driver: get_groundspeed() ─────────────────────────────
            "query_speed": [
                "how fast am i going", "current speed",
                "what is my speed", "ground speed",
                "check speed", "speed status",
            ],

            # ── driver: get_battery() ────────────────────────────────
            "query_battery": [
                "check battery", "battery level", "battery status",
                "how much battery", "battery check",
                "what is my battery", "battery percentage",
            ],

            # ── driver: get_gps_status() ─────────────────────────────
            "query_gps": [
                "check gps", "gps status", "satellite count",
                "how is my gps", "gps signal", "how many satellites",
            ],

            # ── driver: get_current_mode() ───────────────────────────
            "query_mode": [
                "what mode am i in", "current flight mode",
                "which mode", "current mode", "flight mode check",
            ],

            # ── driver: get_location() ───────────────────────────────
            "query_location": [
                "where am i", "current position", "my location",
                "what are my coordinates", "gps coordinates",
                "current coordinates", "position check",
            ],

            # ── driver: trigger_emergency_safe_state() ───────────────
            "emergency_safe": [
                "emergency", "emergency stop", "abort",
                "abort mission", "trigger safe state",
                "safe state", "emergency abort", "stop immediately",
                "kill switch", "mayday", "emergency land",
            ],

            # ── handled in main.py (not a driver method) ─────────────
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

    # ------------------------------------------------------------------

    def _build_drone_dictionary(self) -> dict:
        """
        ArduCopter (multirotor) specific intents.

        Key differences from plane:
        - Takeoff is vertical, no runway roll needed.
        - "Turn left / right" = YAW, not bank.
        - Has true hover, ascend, descend by velocity.
        - No throttle ramp, no flaps, no FBWA/FBWB.
        - Position hold (POSHOLD) is a copter-only mode.
        """
        return {

            # ── driver: execute_takeoff() ─────────────────────────────
            "takeoff": [
                "take off", "takeoff", "launch", "launch drone",
                "lift off", "lift off the ground", "start flying",
                "begin flight", "fly up", "fly up into the air",
                "go airborne", "get airborne", "take off vertically",
                "vertical takeoff",
            ],

            # ── driver: set_position_hold() (POSHOLD — copter only) ───
            "position_hold": [
                "position hold", "poshold", "hold position",
                "stay here", "maintain position", "stay put",
                "freeze position", "stop here", "hold",
                "stay in place", "hold steady",
            ],

            # ── driver: hover() ───────────────────────────────────────
            "hover": [
                "hover", "stop moving", "stop", "stay",
                "zero velocity", "stand by", "hover in place",
                "stay there",
            ],

            # ── driver: move_forward() ────────────────────────────────
            "move_forward": [
                "move forward", "go forward", "fly forward",
                "forward", "advance", "head forward",
                "move ahead", "push forward",
            ],

            # ── driver: move_backward() ───────────────────────────────
            "move_backward": [
                "move backward", "go backward", "fly backward",
                "backward", "reverse", "move back", "go back",
                "pull back",
            ],

            # ── driver: move_left() — STRAFE (not yaw) ───────────────
            "move_left": [
                "move left", "go left", "fly left", "left",
                "strafe left", "slide left", "shift left",
                "translate left",
            ],

            # ── driver: move_right() — STRAFE (not yaw) ──────────────
            "move_right": [
                "move right", "go right", "fly right", "right",
                "strafe right", "slide right", "shift right",
                "translate right",
            ],

            # ── driver: ascend() ─────────────────────────────────────
            "ascend": [
                "ascend", "go up", "climb", "fly up",
                "gain altitude", "increase altitude", "rise",
                "climb up", "go higher", "move up",
            ],

            # ── driver: descend() ────────────────────────────────────
            "descend": [
                "descend", "go down", "fly down", "drop",
                "lose altitude", "decrease altitude",
                "lower", "move down", "come down slowly",
            ],

            # ── driver: rotate_left() — YAW left ────────────────────
            "rotate_left": [
                "rotate left", "yaw left", "turn left",
                "spin left", "rotate counterclockwise",
                "turn to the left", "yaw counterclockwise",
                "pivot left",
            ],

            # ── driver: rotate_right() — YAW right ──────────────────
            "rotate_right": [
                "rotate right", "yaw right", "turn right",
                "spin right", "rotate clockwise",
                "turn to the right", "yaw clockwise",
                "pivot right",
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
        }

    # ------------------------------------------------------------------

    def _build_plane_dictionary(self) -> dict:
        """
        ArduPlane (fixed-wing) specific intents.

        Key differences from drone:
        - Takeoff needs runway roll and rotation speed.
        - "Turn left / right" = BANK/ROLL, not yaw strafe.
        - No true hover — loiter is the closest equivalent.
        - Has throttle control, flaps, FBWA, FBWB, cruise, glide modes.
        - Climb/descend via pitch, not direct velocity.
        - No POSHOLD (copter-only mode).
        """
        return {

            # ── driver: execute_takeoff() — rolling takeoff ───────────
            "takeoff": [
                "take off", "takeoff", "launch", "launch the plane",
                "start takeoff roll", "begin takeoff", "rolling takeoff",
                "start the roll", "throttle up and go",
                "full throttle takeoff", "auto takeoff",
            ],

            # ── driver: change_flight_mode("FBWA") ───────────────────
            # Fly-By-Wire A — stabilised manual with auto level
            "mode_fbwa": [
                "fbwa", "fly by wire a", "fly by wire alpha",
                "stabilized mode", "stabilize", "manual stabilized",
                "assisted manual", "fbw a",
            ],

            # ── driver: change_flight_mode("FBWB") ───────────────────
            # Fly-By-Wire B — speed/altitude hold, pilot controls heading
            "mode_fbwb": [
                "fbwb", "fly by wire b", "fly by wire bravo",
                "speed altitude hold", "cruise assist", "fbw b",
            ],

            # ── driver: change_flight_mode("CRUISE") ─────────────────
            "mode_cruise": [
                "cruise", "cruise mode", "enter cruise",
                "switch to cruise", "auto cruise",
                "level cruise", "cruise flight",
            ],

            # ── driver: change_flight_mode("AUTO") ───────────────────
            "mode_auto": [
                "auto", "auto mode", "autonomous",
                "fly the mission", "start mission",
                "follow waypoints", "execute flight plan",
                "begin auto flight",
            ],

            # ── driver: change_flight_mode("GUIDED") ─────────────────
            "mode_guided": [
                "guided", "guided mode", "switch to guided",
                "enter guided", "guided flight",
            ],

            # ── driver: change_flight_mode("STABILIZE") ──────────────
            "mode_stabilize": [
                "stabilize", "stabilize mode", "manual stabilize",
                "stabilized flight",
            ],

            # ── driver: change_flight_mode("ACRO") ───────────────────
            "mode_acro": [
                "acro", "acrobatic", "acro mode",
                "full manual", "manual control",
            ],

            # ── driver: change_flight_mode("CIRCLE") ─────────────────
            "mode_circle": [
                "circle mode", "enter circle", "start circling",
                "fly circles", "orbit", "orbit mode",
            ],

            # ── Plane turn — BANK LEFT (roll, not yaw strafe) ─────────
            # These map to rotate_left() which sends YAW_CMD;
            # on ArduPlane in GUIDED, yaw commands produce coordinated turns.
            "rotate_left": [
                "turn left", "bank left", "roll left",
                "left turn", "fly left", "bearing left",
                "vector left", "head left",
            ],

            # ── Plane turn — BANK RIGHT ───────────────────────────────
            "rotate_right": [
                "turn right", "bank right", "roll right",
                "right turn", "fly right", "bearing right",
                "vector right", "head right",
            ],

            # ── Plane climb — pitch up / throttle up ─────────────────
            # Maps to ascend() — sends negative vz in GUIDED
            "ascend": [
                "climb", "pitch up", "nose up", "gain altitude",
                "increase altitude", "fly higher", "ascend",
                "go up", "climb to altitude", "throttle up and climb",
            ],

            # ── Plane descend — pitch down / reduce throttle ──────────
            "descend": [
                "descend", "pitch down", "nose down", "lose altitude",
                "decrease altitude", "fly lower", "go down",
                "reduce altitude", "glide down", "descent",
            ],

            # ── Plane glide — engine off descent ─────────────────────
            # Maps to change_flight_mode("GLIDE") or descend() with speed 0
            "glide": [
                "glide", "engine off", "power off descent",
                "glide down", "deadstick", "glide to landing",
                "cut throttle and glide",
            ],

            # ── driver: set_airspeed() — critical for planes ──────────
            # Included here too so plane-specific phrasing is captured
            "set_airspeed": [
                "set airspeed", "airspeed to", "target airspeed",
                "fly at airspeed", "airspeed", "set cruise speed",
                "increase airspeed", "decrease airspeed",
                "speed up", "slow down", "throttle to",
            ],

            # ── Throttle up — increase power ─────────────────────────
            # Handled in main.py as set_groundspeed() with higher value
            "throttle_up": [
                "throttle up", "more throttle", "increase throttle",
                "full throttle", "max power", "increase power",
                "more power", "power up",
            ],

            # ── Throttle down — reduce power ─────────────────────────
            "throttle_down": [
                "throttle down", "less throttle", "reduce throttle",
                "cut throttle", "idle throttle", "reduce power",
                "less power", "throttle back", "power down throttle",
            ],

            # ── driver: set_heading() — compass bearing for planes ────
            "set_heading": [
                "set heading", "heading to", "turn to heading",
                "face north", "face south", "face east", "face west",
                "heading north", "heading south", "heading east", "heading west",
                "fly north", "fly south", "fly east", "fly west",
                "point north", "compass heading", "fly heading",
                "bearing north", "bearing south",
            ],

            # ── Plane hover equivalent — loiter ───────────────────────
            # Planes cannot hover; closest is loiter (circling in place)
            "hover": [
                "hold position", "stay here", "loiter here",
                "circle here", "orbit this point",
                "maintain this position",
            ],

            # ── move_forward() — increase forward speed ───────────────
            "move_forward": [
                "go forward", "fly forward", "move forward",
                "forward", "advance", "increase forward speed",
                "push forward",
            ],

            # ── Plane does not strafe — map left/right to turns ───────
            # "move left" on a plane = bank left turn
            "move_left": [
                "move left", "drift left", "steer left",
                "vector left",
            ],

            "move_right": [
                "move right", "drift right", "steer right",
                "vector right",
            ],

            # ── No meaningful backward on a plane ─────────────────────
            # Map to loiter so the plane at least stops progressing
            "move_backward": [
                "move backward", "go backward", "reverse",
                "slow to a stop", "decelerate",
            ],

            # ── Plane: position hold = LOITER ─────────────────────────
            "position_hold": [
                "position hold", "hold position", "maintain position",
                "stay put", "hold here", "loiter in place",
            ],
        }

    # ======================================================================
    # Audio energy gate
    # ======================================================================

    def _is_silent(self, audio_buffer: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio_buffer.astype(np.float32) ** 2)))
        return rms < self._min_rms_energy

    # ======================================================================
    # Transcription
    # ======================================================================

    def transcribe_audio(self, audio_buffer: np.ndarray, final: bool = False) -> str:
        if self._is_silent(audio_buffer):
            return ""

        beam_size = 5 if final else 2
        segments, info = self.model.transcribe(
            audio_buffer,
            language="en",
            beam_size=beam_size,
            best_of=5 if final else 1,
            initial_prompt=DRONE_PROMPT,
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4],
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.55,
            log_prob_threshold=-0.8,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.60,
                min_speech_duration_ms=200,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
            ),
        )

        no_speech_prob = getattr(info, 'no_speech_prob', 0.0)
        if no_speech_prob > 0.55:
            print(f"[NLP BRAIN] Low speech confidence ({no_speech_prob:.2f}) — discarding segment.")
            return ""

        return " ".join(seg.text for seg in segments).strip()

    def reset_duplicate_guard(self):
        self._last_transcript = ""
        self._repeat_count = 0

    # ======================================================================
    # Intent matching
    # ======================================================================

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

    # ======================================================================
    # Parameter extraction
    # ======================================================================

    def extract_number(self, text: str) -> float | None:
        WORD_TO_NUM = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "hundred": 100,
        }
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        if matches:
            return float(matches[0])
        for word in text.lower().split():
            word = word.strip(".,!?")
            if word in WORD_TO_NUM:
                return float(WORD_TO_NUM[word])
        return None

    def extract_duration(self, text: str) -> float | None:
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?)',
            text, re.IGNORECASE
        )
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if "second" in unit:
                return value
            if "minute" in unit:
                return value * 60
            if "hour" in unit:
                return value * 3600
        return None

    def extract_heading(self, text: str) -> float | None:
        COMPASS_TO_DEG = {
            "north": 0.0,     "northeast": 45.0,
            "east":  90.0,    "southeast": 135.0,
            "south": 180.0,   "southwest": 225.0,
            "west":  270.0,   "northwest": 315.0,
        }
        for word, deg in COMPASS_TO_DEG.items():
            if word in text.lower():
                return deg
        return self.extract_number(text)

    def extract_speed(self, text: str) -> float | None:
        speed = self.extract_number(text)
        return min(speed, 30.0) if speed is not None else None