from typing import Dict, Any, List, Union
import os
from anthropic import Anthropic
import openai
import google.generativeai as genai
from .base import BaseJudge
from .utils import OPENAI_MODEL_LIST, ANTHROPIC_MODEL_LIST, GEMINI_MODEL_LIST, TOGETHER_MODEL_LIST
from dotenv import load_dotenv

load_dotenv("key.env")

class ConfigurationError(Exception):
    """Raised when there are configuration issues with API keys."""
    pass

class AnthropicJudge(BaseJudge):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "<your_api_key>":
            raise ConfigurationError(
                f"❌ Anthropic API key not found.\n"
                f"Please set ANTHROPIC_API_KEY in your key.env file.\n"
                f"Get your API key from: https://console.anthropic.com/"
            )
        self.client = Anthropic(api_key=api_key)

    def evaluate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 5000
        }

        response = self.client.messages.create(**kwargs)
        return response.content[-1].text.strip() if "claude-3-7" in self.model_name else response.content[0].text.strip()

class OpenAIJudge(BaseJudge):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "<your_api_key>":
            raise ConfigurationError(
                f"❌ OpenAI API key not found.\n"
                f"Please set OPENAI_API_KEY in your key.env file.\n"
                f"Get your API key from: https://platform.openai.com/api-keys"
            )
        self.client = openai.OpenAI(api_key=api_key)

    def evaluate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        if self.model_name == "o3-mini" or self.model_name == "o4-mini":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                reasoning_effort="high"
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        return response.choices[0].message.content.strip()

class GeminiJudge(BaseJudge):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "<your_api_key>":
            raise ConfigurationError(
                f"❌ Google API key not found.\n"
                f"Please set GOOGLE_API_KEY in your key.env file.\n"
                f"Get your API key from: https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)

    def evaluate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        # Convert to string if single message list
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]['content']
            
        if isinstance(prompt, str):
            # Single prompt
            response = self.client.generate_content(prompt)
        else:
            # Multi-turn conversation - convert to Gemini format
            gemini_conversation = []
            for msg in prompt:
                role = "model" if msg.get("role") == "assistant" else "user"
                content = msg.get("content", "")
                gemini_conversation.append({"role": role, "parts": [content]})
            
            # Use chat for multi-turn
            chat = self.client.start_chat(history=gemini_conversation[:-1])
            response = chat.send_message(gemini_conversation[-1]["parts"][0])
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise ValueError(f"Content blocked: {response.prompt_feedback.block_reason}")
        return response.text.strip()

def get_judge(model_name: str) -> BaseJudge:
    """Factory function to get the appropriate judge for a model."""
    if model_name in ANTHROPIC_MODEL_LIST:
        return AnthropicJudge(model_name)
    elif model_name in OPENAI_MODEL_LIST:
        return OpenAIJudge(model_name)
    elif model_name in GEMINI_MODEL_LIST:
        return GeminiJudge(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")