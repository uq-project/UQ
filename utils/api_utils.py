"""API utilities for different LLM providers."""

import openai
import os
from typing import Dict, Any
from anthropic import Anthropic
import google.generativeai as genai
from together import Together
from utils.utils import OPENAI_MODEL_LIST, ANTHROPIC_MODEL_LIST, GEMINI_MODEL_LIST, TOGETHER_MODEL_LIST


def initialize_client(model_name: str):
    """Initialize the appropriate API client based on model name."""
    try:
        if model_name in OPENAI_MODEL_LIST:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in key.env file")
            return openai.OpenAI(api_key=api_key)
            
        elif model_name in ANTHROPIC_MODEL_LIST:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return Anthropic(api_key=api_key)
            
        elif model_name in GEMINI_MODEL_LIST:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(model_name)
            
        elif model_name in TOGETHER_MODEL_LIST:
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
            return Together(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize client for {model_name}: {str(e)}")


def format_prompt(question: Dict[str, Any]) -> str:
    """Format the prompt template with question details."""
    return f"""### Question Details
Title: {question.get("title", "")}
Keywords: {", ".join(question.get("tags", []))}
Site: {question.get("site", "")}
Link: {question.get("link", "")}
Category: {question.get("category", "")}

### Question Content
{question.get("body", question.get("question", ""))}
"""


def generate_openai_response(client, model_name: str, prompt: str) -> str:
    """Generate response using OpenAI models."""
    if model_name == "o3-mini" or model_name == "o4-mini":
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "developer", "content": "Please answer the following question comprehensively and accurately."},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="high",
            temperature=0.3
        )
    elif model_name == "o3-pro" or model_name == "o3":
        response = client.responses.create(
            model=model_name,
            input="Please answer the following question comprehensively and accurately.\n"+prompt
        )
        return response.output_text
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "developer", "content": "Please answer the following question comprehensively and accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
    return response.choices[0].message.content.strip()


def generate_anthropic_response(client, model_name: str, prompt: str) -> str:
    """Generate response using Anthropic models."""
    try:
        # Try with thinking mode for Claude 3.7 models
        if "claude-3-7" in model_name:
            try:
                response = client.messages.create(
                    model=model_name,
                    system="Please answer the following question comprehensively and accurately.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=21000,
                    temperature=0.3,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 16000
                    }
                )
                return response.content[-1].text.strip()
            except Exception as thinking_error:
                # Fallback to regular mode if thinking mode fails
                print(f"Warning: Thinking mode failed for {model_name}, falling back to regular mode: {thinking_error}")
                response = client.messages.create(
                    model=model_name,
                    system="Please answer the following question comprehensively and accurately.",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=21000,
                    temperature=0.3
                )
                return response.content[0].text.strip()
        else:
            response = client.messages.create(
                model=model_name,
                system="Please answer the following question comprehensively and accurately.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.3
            )
            return response.content[0].text.strip()
    except Exception as e:
        # Additional fallback for any Anthropic API issues
        print(f"Anthropic API error for {model_name}: {e}")
        raise


def generate_gemini_response(client, prompt: str) -> str:
    """Generate response using Gemini models."""
    # Add instruction directly to prompt
    full_prompt = "Please answer the following question comprehensively and accurately.\n\n" + prompt
    
    response = client.generate_content(
        full_prompt,
        generation_config={"temperature": 0.3}
    )
    
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        raise ValueError(f"Content blocked: {response.prompt_feedback.block_reason}")
    return response.text.strip()


def generate_together_response(client, model_name: str, prompt: str) -> str:
    """Generate response using Together models."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Please answer the following question comprehensively and accurately.\n"+prompt}
        ],
        max_tokens=21000,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()