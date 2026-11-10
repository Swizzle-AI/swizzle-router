import re
from typing import Literal

IntentType = Literal[
    "CODE_REASONING",
    "STRUCTURED_JSON",
    "ROBOTICS_TELEOP",
    "CONVERSATION",
    "GENERAL"
]

class IntentClassifier:
    """Sub-millisecond regex & keyword heuristic intent classifier."""
    CODE_PATTERNS = [r"def\s+\w+", r"function\s+\w+", r"class\s+\w+", r"import\s+\w+", r"```\w+"]
    ROBOTICS_PATTERNS = [r"teleop", r"servoing", r"quaternion", r"trajectory", r"end_effector", r"gripper"]

    @classmethod
    def classify(cls, prompt: str) -> IntentType:
        for p in cls.ROBOTICS_PATTERNS:
            if re.search(p, prompt, re.IGNORECASE):
                return "ROBOTICS_TELEOP"

        for p in cls.CODE_PATTERNS:
            if re.search(p, prompt):
                return "CODE_REASONING"

        if "{" in prompt and "}" in prompt and ("json" in prompt.lower() or "schema" in prompt.lower()):
            return "STRUCTURED_JSON"

        if len(prompt.split()) < 20 and any(w in prompt.lower() for w in ["hi", "hello", "hey", "who are you"]):
            return "CONVERSATION"

        return "GENERAL"
