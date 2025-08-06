"""
VOYAGER-Trader: An autonomous, self-improving trading system.

Inspired by the VOYAGER project, this system focuses on discovering and
developing trading knowledge through systematic exploration.
"""

__version__ = "0.1.0"
__author__ = "VOYAGER-Trader Team"

from .core import VoyagerTrader
from .curriculum import AutomaticCurriculum
from .prompting import IterativePrompting
from .skills import SkillLibrary

__all__ = [
    "VoyagerTrader",
    "AutomaticCurriculum",
    "SkillLibrary",
    "IterativePrompting",
]
