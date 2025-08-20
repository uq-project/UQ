from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import datetime
import re

class BaseJudge(ABC):
    """Base class for all LLM-based judges."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def evaluate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Call the LLM API to evaluate a prompt."""
        pass

class JudgmentStrategy(ABC):
    """Base class for all judgment strategies."""
    
    @abstractmethod
    def create_prompt(self, question: Dict[str, Any], answer: str, ref_answer: Optional[str] = None, **kwargs) -> str:
        """Create the evaluation prompt for this strategy.
        
        Args:
            question: Dictionary containing question details
            answer: The answer to evaluate
            ref_answer: Optional reference answer
            **kwargs: Additional keyword arguments for customizing prompt creation
        """
        pass
    
    @abstractmethod
    def judge(self, question: Dict[str, Any], answer: str, judge: BaseJudge, 
              ref_answer: Optional[str] = None) -> Dict[str, Any]:
        """Execute the judgment strategy and return the result."""
        pass
    
    def _extract_decision(self, evaluation: str) -> bool:
        """Extract Y/N decision from evaluation text."""
        match = re.search(r"\[\[(Y|N)\]\]", evaluation)
        return match.group(1) == "Y" if match else False

class JudgmentDecorator(JudgmentStrategy):
    """Base decorator for adding behavior to judgment strategies."""
    
    def __init__(self, strategy: JudgmentStrategy):
        self.strategy = strategy
    
    def create_prompt(self, question: Dict[str, Any], answer: str, ref_answer: Optional[str] = None) -> str:
        return self.strategy.create_prompt(question, answer, ref_answer)