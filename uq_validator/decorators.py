from typing import Dict, Any, List, Optional, Callable
import datetime
from .base import JudgmentStrategy, BaseJudge, JudgmentDecorator

class RepeatedSamplingDecorator(JudgmentDecorator):
    """Decorator to add repeated sampling to any judgment strategy."""
    
    def __init__(self, strategy: JudgmentStrategy, n_samples: int = 3):
        super().__init__(strategy)
        self.n_samples = n_samples
    
    def judge(self, question: Dict[str, Any], answer: str, judge: BaseJudge, 
              ref_answer: Optional[str] = None) -> Dict[str, Any]:
        results = []
        
        for _ in range(self.n_samples):
            result = self.strategy.judge(question, answer, judge, ref_answer)
            
            # For multi-turn results, extract the final decision
            if "decisions" in result:
                sample_result = {
                    "strategy": result["strategy"],
                    "judge_model": result["judge_model"],
                    "is_accepted": result["is_accepted"],  # Use the voted result from multi-turn
                    "evaluation": result["evaluations"][-1],  # Use the final evaluation
                    "conversations": result.get("conversations", []),
                    "timestamp": result["timestamp"]
                }
                # Copy any additional fields
                for key in result:
                    if key not in sample_result and key not in ["decisions", "evaluations"]:
                        sample_result[key] = result[key]
                results.append(sample_result)
            else:
                results.append(result)
        
        return {
            "strategy": f"repeated_sampling({self.strategy.__class__.__name__})",
            "judge_model": judge.model_name,
            "n_samples": self.n_samples,
            "sample_results": results,
            "evaluations": [r["evaluation"] for r in results],
            "decisions": [r["is_accepted"] for r in results],
            "timestamp": datetime.datetime.now().isoformat()
        }

class MultiTurnDecorator(JudgmentDecorator):
    """Decorator to add multi-turn prompting to any judgment strategy."""
    
    def __init__(self, strategy: JudgmentStrategy, n_turns: int = 3):
        super().__init__(strategy)
        self.n_turns = n_turns
    
    def create_confirmation_prompt(self) -> str:
        """Create a prompt to confirm previous evaluation."""
        return """Think twice about your judgment. Are you still confident in your assessment?
After careful reconsideration, provide your final decision using the same format: "[[Y]]" if you maintain your acceptance or "[[N]]" if you change to rejection."""
    
    def judge(self, question: Dict[str, Any], answer: str, judge: BaseJudge, 
              ref_answer: Optional[str] = None) -> Dict[str, Any]:
        # Initial judgment
        initial_result = self.strategy.judge(question, answer, judge, ref_answer)
        evaluations = [initial_result["evaluation"]]
        decisions = [initial_result["is_accepted"]]
        
        # Prepare for multi-turn conversation
        messages = [
            {"role": "user", "content": initial_result["prompt"]},
            {"role": "assistant", "content": initial_result["evaluation"]}
        ]
        
        # Subsequent turns
        for _ in range(self.n_turns - 1):
            messages.append({"role": "user", "content": self.create_confirmation_prompt()})
            response = judge.evaluate(messages)
            messages.append({"role": "assistant", "content": response})
            
            evaluations.append(response)
            decisions.append(self._extract_decision(response))
        
        result = {
            "strategy": f"multi_turn({self.strategy.__class__.__name__})",
            "judge_model": judge.model_name,
            "n_turns": self.n_turns,
            "evaluations": evaluations,
            "decisions": decisions,
            "conversations": messages,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Copy any additional fields from initial result
        for key in initial_result:
            if key not in result and key not in ["evaluation", "is_accepted"]:
                result[key] = initial_result[key]
                
        return result

class MajorityVotingDecorator(JudgmentDecorator):
    """Decorator to apply majority voting to sample results."""
    
    def judge(self, question: Dict[str, Any], answer: str, judge: BaseJudge, 
              ref_answer: Optional[str] = None) -> Dict[str, Any]:
        result = self.strategy.judge(question, answer, judge, ref_answer)
        
        # Handle both repeated sampling and multi-turn results
        if "decisions" in result:
            decisions = result["decisions"]
            is_accepted = sum(decisions) > len(decisions) // 2
        elif "sample_results" in result:
            decisions = [r["is_accepted"] for r in result["sample_results"]]
            is_accepted = sum(decisions) > len(decisions) // 2
        else:
            # If neither format is found, just pass through the original result
            return result
        
        result["is_accepted"] = is_accepted
        result["voting_method"] = "majority"
        
        return result

class UnanimousVotingDecorator(JudgmentDecorator):
    """Decorator to apply unanimous voting to sample results."""
    
    def judge(self, question: Dict[str, Any], answer: str, judge: BaseJudge, 
              ref_answer: Optional[str] = None) -> Dict[str, Any]:
        result = self.strategy.judge(question, answer, judge, ref_answer)
        
        # Handle both repeated sampling and multi-turn results
        if "decisions" in result:
            decisions = result["decisions"]
            is_accepted = all(decisions)
        elif "sample_results" in result:
            decisions = [r["is_accepted"] for r in result["sample_results"]]
            is_accepted = all(decisions)
        else:
            # If neither format is found, just pass through the original result
            return result
        
        result["is_accepted"] = is_accepted
        result["voting_method"] = "unanimous"
        
        return result

class SequentialJudgmentDecorator(JudgmentDecorator):
    """Decorator to run multiple judgment strategies in sequence, only proceeding if previous steps pass."""
    
    def __init__(self, strategies: List[JudgmentStrategy]):
        self.strategies = strategies
        # Initialize with the first strategy as the base
        super().__init__(strategies[0])
    
    def create_prompt(self, question: Dict[str, Any], answer: str, ref_answer: Optional[str] = None) -> str:
        # Just delegate to the first strategy
        return self.strategies[0].create_prompt(question, answer, ref_answer)
    
    def judge(self, question: Dict[str, Any], answer: str, judge: BaseJudge, 
              ref_answer: Optional[str] = None) -> Dict[str, Any]:
        results = []
        
        for strategy in self.strategies:
            result = strategy.judge(question, answer, judge, ref_answer)
            results.append(result)
            
            # If this strategy rejects, stop the sequence
            if not result["is_accepted"]:
                break
        
        return {
            "strategy": "sequential_judgment",
            "judge_model": judge.model_name,
            "step_results": results,
            "is_accepted": all(r["is_accepted"] for r in results),
            "completed_steps": len(results),
            "total_steps": len(self.strategies),
            "timestamp": datetime.datetime.now().isoformat()
        }