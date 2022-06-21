from .lsl import LSLTrigger
from .mock import MockTrigger
from .parallel import ParallelPortTrigger
from .software import SoftwareTrigger
from .trigger_def import TriggerDef

__all__ = [
    "LSLTrigger",
    "MockTrigger",
    "ParallelPortTrigger",
    "SoftwareTrigger",
    "TriggerDef",
]
