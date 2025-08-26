# UQ Validator

## Features

- **Multiple Judgment Strategies**: Relevance, cycle consistency, fact checking, correctness, and vanilla evaluation
- **Composable Architecture**: Combine strategies using decorators for complex evaluation pipelines
- **Multi-model Support**: Works with Anthropic, OpenAI, and Google Gemini models
- **Parallel Processing**: Efficient concurrent evaluation with configurable worker threads
- **Voting Mechanisms**: Majority and unanimous voting for multi-sample evaluations
- **Sequential Evaluation**: Chain multiple strategies for comprehensive assessment

## Installation

Install in editable mode for development:

```bash
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Basic evaluation
python validate.py --input_file generated_answer/gpt-4o_uq_dataset.jsonl --strategy relevance --model o3

# Sequential strategy with multiple steps
python validate.py --input_file generated_answer/gpt-4o_uq_dataset.jsonl --strategy sequential --sequential_strategies cycle_consistency fact_check correctness

# Multi-turn evaluation with majority voting
python validate.py --input_file generated_answer/gpt-4o_uq_dataset.jsonl --strategy cycle_consistency --turns 3 --multi_turn_voting majority
```


## Strategies

### Available Strategies

- **Relevance**: Evaluates if the answer is relevant to the question
- **Cycle Consistency**: Checks consistency by generating questions from answers
- **Fact Check**: Validates factual accuracy and logical reasoning
- **Final Answer**: Extracts and evaluates the core answer
- **Correctness**: Comprehensive total correctness assessment
- **Vanilla**: Basic evaluation without specific focus
- **Sequential**: Combines multiple strategies in sequence

### Decorators

- **RepeatedSamplingDecorator**: Repeated Sampling
- **MultiTurnDecorator**: Iterated Reflection
- **MajorityVotingDecorator**: Majority voting decision making
- **UnanimousVotingDecorator**: Requires unanimous agreement
- **SequentialJudgmentDecorator**: Chains multiple strategies

## Configuration

### Environment Variables

Create a `key.env` file with your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
```

### Model Support

- **Anthropic**: Claude models (claude-3-7-sonnet-20250219, claude-opus-4-20250514, etc.)
- **OpenAI**: GPT/o-series models (o3, o4-mini, etc.)
- **Google**: Gemini models (gemini-2.5-pro, etc.)

## Output Format

Results are saved as JSONL files with the following structure:

```json
{
  "question_id": "1", 
  "model_name": "gpt-4o", 
  "evaluation": {
    "strategy": "multi_turn(CycleConsistencyStrategy)", 
    "judge_model": "o3", 
    "n_turns": 3, 
    "evaluations": ["full judgment reasoning"], 
    "decisions": [true, true, true], 
    "timestamp": "2025-08-20T16:35:32.532305", 
    "is_accepted": true, 
    "voting_method": "unanimous"
    }
}
```
