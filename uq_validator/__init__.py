"""
UQ Validator: A composable LLM validation framework for questions without ground truth.

This package provides a flexible framework for evaluating LLM responses using
various judgment strategies including cycle consistency, fact&logic check, correctness, and more.
"""

__version__ = "0.1.0"
__author__ = "UQ Project Team"

from .base import BaseJudge, JudgmentStrategy, JudgmentDecorator
from .factory import JudgmentFactory
from .strategies import (
    RelevanceStrategy,
    CycleConsistencyStrategy,
    FactCheckStrategy,
    FinalAnswerStrategy,
    CorrectnessStrategy,
    VanillaStrategy,
)
from .decorators import (
    RepeatedSamplingDecorator,
    MultiTurnDecorator,
    MajorityVotingDecorator,
    UnanimousVotingDecorator,
    SequentialJudgmentDecorator,
)
from .model_adapters import get_judge, AnthropicJudge, OpenAIJudge, GeminiJudge

__all__ = [
    # Core classes
    "BaseJudge",
    "JudgmentStrategy", 
    "JudgmentDecorator",
    
    # Factory
    "JudgmentFactory",
    
    # Strategies
    "RelevanceStrategy",
    "CycleConsistencyStrategy", 
    "FactCheckStrategy",
    "FinalAnswerStrategy",
    "CorrectnessStrategy",
    "VanillaStrategy",
    
    # Decorators
    "RepeatedSamplingDecorator",
    "MultiTurnDecorator",
    "MajorityVotingDecorator", 
    "UnanimousVotingDecorator",
    "SequentialJudgmentDecorator",
    
    # Model adapters
    "get_judge",
    "AnthropicJudge",
    "OpenAIJudge",
    "GeminiJudge", 
]