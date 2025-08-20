"""Dataset utilities for loading and processing UQ datasets."""

from datasets import load_dataset
from typing import List, Dict, Any
import json


def load_uq_dataset(streaming: bool = False) -> List[Dict[str, Any]]:
    """
    Load the UQ dataset from Hugging Face.
    
    Args:
        streaming: Whether to use streaming mode
        
    Returns:
        List of question dictionaries
    """
    try:
        dataset = load_dataset("uq-project/uq", split="test", streaming=streaming)
        
        if streaming:
            # For streaming, return as is (iterator)
            return dataset
        else:
            # Convert to list of dictionaries
            return list(dataset)
            
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from Hugging Face: {str(e)}")


def filter_by_category(questions: List[Dict[str, Any]], category: str = None) -> List[Dict[str, Any]]:
    """
    Filter questions by category if specified.
    
    Args:
        questions: List of question dictionaries
        category: Category to filter by (optional)
        
    Returns:
        Filtered list of questions
    """
    if not category:
        return questions
    
    return [q for q in questions if q.get("category") == category]


def load_existing_answers(output_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load existing answers from output file to avoid reprocessing.
    
    Args:
        output_file: Path to the output file
        
    Returns:
        Dictionary mapping question_id to result
    """
    existing_answers = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    existing_answers[result['question_id']] = result
    except FileNotFoundError:
        pass  # File doesn't exist yet, return empty dict
    except Exception as e:
        print(f"Warning: Could not load existing answers: {e}")
    
    return existing_answers


def save_result_to_file(result: Dict[str, Any], output_file: str, lock=None) -> None:
    """
    Save a single result to the output file.
    
    Args:
        result: Result dictionary to save
        output_file: Path to output file
        lock: Thread lock for concurrent access (optional)
    """
    if lock:
        with lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")