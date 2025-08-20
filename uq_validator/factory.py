from typing import Dict, Any, List, Optional
from .base import JudgmentStrategy, BaseJudge
from .model_adapters import get_judge
from .strategies import (
    RelevanceStrategy,
    CycleConsistencyStrategy,
    FactualErrorStrategy,
    FinalAnswerStrategy,
    TotalCorrectnessStrategy,
    VanillaStrategy
)
from .decorators import (
    RepeatedSamplingDecorator,
    MultiTurnDecorator,
    MajorityVotingDecorator,
    UnanimousVotingDecorator,
    SequentialJudgmentDecorator
)

class JudgmentFactory:
    """Factory class to create and compose judgment strategies."""

    @staticmethod
    def _apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting):
        """Apply decorators to a strategy based on parameters."""
        # Apply multi-turn first if needed
        if n_turns > 1:
            strategy = MultiTurnDecorator(strategy, n_turns)
            if multi_turn_voting == "majority":
                strategy = MajorityVotingDecorator(strategy)
            elif multi_turn_voting == "unanimous":
                strategy = UnanimousVotingDecorator(strategy)

        # Then apply resampling if needed
        if n_samples > 1:
            strategy = RepeatedSamplingDecorator(strategy, n_samples)
            if resampling_voting == "majority":
                strategy = MajorityVotingDecorator(strategy)
            elif resampling_voting == "unanimous":
                strategy = UnanimousVotingDecorator(strategy)
        
        return strategy

    @staticmethod
    def create_cycle_consistency(
        model_name: str,
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a cycle consistency judgment strategy with optional decorators."""
        judge = get_judge(model_name)
        strategy = CycleConsistencyStrategy()
        strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
        return strategy, judge

    @staticmethod
    def create_fact_check(
        model_name: str,
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a factual error judgment strategy with optional decorators."""
        judge = get_judge(model_name)
        strategy = FactualErrorStrategy()
        strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
        return strategy, judge

    @staticmethod
    def create_final_answer(
        model_name: str,
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a final answer judgment strategy with optional decorators."""
        judge = get_judge(model_name)
        strategy = FinalAnswerStrategy()
        strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
        return strategy, judge

    @staticmethod
    def create_correctness(
        model_name: str,
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a total correctness judgment strategy with optional decorators."""
        judge = get_judge(model_name)
        strategy = TotalCorrectnessStrategy()
        strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
        return strategy, judge

    @staticmethod
    def create_relevance(
        model_name: str,
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a relevance judgment strategy with optional decorators."""
        judge = get_judge(model_name)
        strategy = RelevanceStrategy()
        strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
        return strategy, judge

    @staticmethod
    def create_vanilla(
        model_name: str,
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a vanilla judgment strategy with optional decorators."""
        judge = get_judge(model_name)
        strategy = VanillaStrategy()
        strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
        return strategy, judge

    @staticmethod
    def create_sequential(
        model_name: str,
        strategies: List[str],
        n_samples: int = 1,
        n_turns: int = 1,
        resampling_voting: str = "majority",
        multi_turn_voting: str = "majority"
    ) -> JudgmentStrategy:
        """Create a sequential judgment strategy with the specified sub-strategies."""
        judge = get_judge(model_name)
        strategy_map = {
            "relevance": RelevanceStrategy(),
            "cycle_consistency": CycleConsistencyStrategy(),
            "fact_check": FactualErrorStrategy(),
            "final_answer": FinalAnswerStrategy(),
            "correctness": TotalCorrectnessStrategy(),
            "vanilla": VanillaStrategy()
        }

        strategy_instances = [strategy_map[s] for s in strategies]

        # Apply decorators to each strategy
        decorated_strategies = []
        for strategy in strategy_instances:
            strategy = JudgmentFactory._apply_decorators(strategy, n_samples, n_turns, resampling_voting, multi_turn_voting)
            decorated_strategies.append(strategy)

        return SequentialJudgmentDecorator(decorated_strategies), judge