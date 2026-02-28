import sys
import os
import json
import time
from typing import Generator, List
from ..config import Config
from ..ui.interface import UI 
from .api import Client


class HacxBrain:
    """Handles the connection to the LLM"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = Config.get_model()
        self.system_prompt = Config.load_system_prompt()
        self.session_dir = os.path.join(os.path.expanduser("~"), ".hacxgpt_sessions")
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            
        self._init_client()
        
        # HacxGPT models don't need the system prompt history
        if Config.is_hacxgpt_model(self.model):
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system_prompt}]

    def _init_client(self):
        config = Config.get_provider_config()
        base_url = config.get("base_url")
        
        # Initialize the local API client with browser-like headers for compatibility
        self.client = Client(
            api_key=self.api_key,
            base_url=base_url,
            timeout=60,
            default_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "HTTP-Referer": "https://github.com/HacxGPT-Official/HacxGPT-CLI",
                "X-Title": "HacxGPT-CLI"
            }
        )

    def set_model(self, model_name: str):
        """Update the model being used"""
        self.model = model_name
        Config.ACTIVE_MODEL = model_name
        # If switching between HacxGPT and normal models, a reset might be best, 
        # but we'll try to just remove/add the system prompt if the history is fresh.
        if len(self.history) <= 1:
            self.reset()

    def set_provider(self, provider_name: str, api_key: str):
        """Update provider and re-init client"""
        Config.ACTIVE_PROVIDER = provider_name
        Config.ACTIVE_MODEL = None # Reset model to provider default
        self.api_key = api_key
        self.model = Config.get_model()
        self._init_client()
        self.reset() # Reset history on provider switch for safety

    def reset(self):
        if Config.is_hacxgpt_model(self.model):
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system_prompt}]
            
    def save_session(self, name: str) -> str:
        """Save current history to a file"""
        session_data = {
            "model": self.model,
            "provider": Config.ACTIVE_PROVIDER,
            "history": self.history,
            "timestamp": time.time()
        }
        filename = f"{name}.json"
        filepath = os.path.join(self.session_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=4)
        return filepath

    def load_session(self, name: str) -> bool:
        """Load history from a file"""
        filename = f"{name}.json"
        filepath = os.path.join(self.session_dir, filename)
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
            
        self.history = session_data.get("history", [])
        self.model = session_data.get("model", self.model)
        Config.ACTIVE_PROVIDER = session_data.get("provider", Config.ACTIVE_PROVIDER)
        Config.ACTIVE_MODEL = self.model
        self._init_client()
        return True

    def list_sessions(self) -> List[str]:
        """List all saved session names"""
        sessions = [f.replace(".json", "") for f in os.listdir(self.session_dir) if f.endswith(".json")]
        return sorted(sessions)
        
    def chat(self, user_input: str) -> Generator[str, None, None]:
        self.history.append({"role": "user", "content": user_input})
        
        # Use config toggle
        use_stream = Config.STREAMING
        
        try:
            # Using the local API client (it mimics OpenAI SDK)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=use_stream,
                temperature=0.75
            )
            
            full_content = ""
            if use_stream:
                for chunk in response:
                    # chunk is ChatCompletionChunk from api.py
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.content
                        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "thought", None)
                        
                        if reasoning:
                            # Wrap reasoning in <think> tags for the UI to handle it
                            think_content = f"<think>{reasoning}</think>"
                            full_content += think_content
                            yield think_content
                        
                        if content:
                            full_content += content
                            yield content
            else:
                # Non-streaming response handling
                msg = response.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "thought", None)
                if reasoning:
                    full_content = f"<think>{reasoning}</think>{content}"
                else:
                    full_content = content
                yield full_content
            
            if not full_content:
                yield "[bold red]Neural Link Error: No response received. (Uplink Timeout or tunnel drop)[/]"
            else:
                self.history.append({"role": "assistant", "content": full_content})
            
        except Exception as e:
            if "401" in str(e):
                yield f"Error: 401 Unauthorized for {Config.ACTIVE_PROVIDER.upper()}. Check your API Key."
            else:
                yield f"Error: Connection Terminated. Reason: {str(e)}"
