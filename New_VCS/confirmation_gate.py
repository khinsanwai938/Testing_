"""
confirmation_gate.py — "Are you sure?" voice confirmation gate.

Every actionable command (anything except telemetry queries) is routed
through `confirm_and_run()` before it touches the vehicle. The gate:

  1. Speaks a confirmation question naming the action.
  2. Listens for the next utterance via the same STT recorder.
  3. Accepts yes/no (and close synonyms) via fuzzy match.
  4. If the answer is unintelligible, re-asks ONCE, then cancels.
  5. Only on a clear "yes" does it actually execute the pending action.

This module owns no vehicle logic — it only gates execution of a
callback that the dispatcher supplies.
"""

import time
from rapidfuzz import fuzz, process

from voice_io import voice_reply, is_muted

# ── Confirmation vocabulary ───────────────────────────────────────────────
YES_PHRASES = ["yes", "yeah", "yep", "confirm", "affirmative", "go ahead",
               "do it", "proceed", "correct", "sure", "okay", "ok"]
NO_PHRASES  = ["no", "nope", "negative", "cancel", "stop", "abort",
               "don't", "do not", "never mind", "nevermind"]

CONFIRM_MATCH_THRESHOLD = 70.0
ANSWER_TIMEOUT_SECONDS  = 12.0   # how long to wait for a yes/no reply
REASK_TIMEOUT_SECONDS   = 12.0   # how long to wait after the single re-ask

# Intents that never need confirmation — pure information requests.
TELEMETRY_INTENTS = {
    "get_altitude", "get_heading", "get_battery", "get_location",
    "get_mode", "get_gps",
}


def needs_confirmation(intent: str) -> bool:
    """Return True if `intent` must pass through the confirmation gate."""
    return intent not in TELEMETRY_INTENTS and intent != "invalid_command"


def _classify_answer(text: str):
    """
    Return 'yes', 'no', or None (unintelligible) for a spoken reply.
    """
    text = text.lower().strip()

    yes_match = process.extractOne(text, YES_PHRASES, scorer=fuzz.WRatio)
    no_match  = process.extractOne(text, NO_PHRASES, scorer=fuzz.WRatio)

    yes_score = yes_match[1] if yes_match else 0.0
    no_score  = no_match[1] if no_match else 0.0

    if yes_score < CONFIRM_MATCH_THRESHOLD and no_score < CONFIRM_MATCH_THRESHOLD:
        return None
    return "yes" if yes_score >= no_score else "no"


class ConfirmationGate:
    """
    Stateful gate that sits in front of the normal STT transcription
    callback. While a confirmation is pending, the next utterance(s)
    are consumed as yes/no answers instead of being parsed as new
    flight commands.
    """

    def __init__(self):
        self._pending_action = None      # Callable[[], None]
        self._pending_label  = None       # human-readable action name
        self._pending_deadline = None
        self._reasked = False

    @property
    def awaiting_answer(self) -> bool:
        return self._pending_action is not None

    def request_confirmation(self, label: str, action):
        """
        Ask "are you sure you want to <label>?" and arm the gate to
        capture the next utterance as the yes/no answer.

        `action` is a zero-arg callable executed only on "yes".
        """
        self._pending_action   = action
        self._pending_label    = label
        self._pending_deadline = time.time() + ANSWER_TIMEOUT_SECONDS
        self._reasked          = False
        voice_reply(f"Are you sure you want to {label}?")

    def _cancel(self, reason: str = ""):
        if reason:
            print(f"[CONFIRM] Cancelled — {reason}")
        self._pending_action   = None
        self._pending_label    = None
        self._pending_deadline = None
        self._reasked          = False

    def check_timeout(self):
        """
        Call periodically (e.g. once per loop tick) to handle silent
        timeouts when no answer arrives at all.
        """
        if not self.awaiting_answer:
            return
        if self._pending_deadline is None:
            return
        if time.time() < self._pending_deadline:
            return

        if not self._reasked:
            # Re-ask once.
            label = self._pending_label
            self._reasked = True
            self._pending_deadline = time.time() + REASK_TIMEOUT_SECONDS
            voice_reply(f"I didn't catch that. Are you sure you want to {label}?")
        else:
            voice_reply(f"No response. Cancelling {self._pending_label}.")
            self._cancel("timeout after re-ask")

    def handle_utterance(self, text: str) -> bool:
        """
        If a confirmation is pending, consume `text` as the answer and
        return True (caller should NOT treat this text as a new command).
        Returns False if no confirmation was pending (normal dispatch
        should proceed as usual).
        """
        if not self.awaiting_answer:
            return False

        answer = _classify_answer(text)

        if answer == "yes":
            label, action = self._pending_label, self._pending_action
            self._cancel()
            print(f"[CONFIRM] Approved — executing: {label}")
            action()
            return True

        if answer == "no":
            label = self._pending_label
            self._cancel()
            voice_reply(f"Okay, cancelling {label}.")
            return True

        # Unintelligible reply.
        if not self._reasked:
            label = self._pending_label
            self._reasked = True
            self._pending_deadline = time.time() + REASK_TIMEOUT_SECONDS
            voice_reply(f"Sorry, was that a yes or a no? Are you sure you want to {label}?")
        else:
            voice_reply(f"I still didn't understand. Cancelling {self._pending_label}.")
            self._cancel("unintelligible after re-ask")

        return True
