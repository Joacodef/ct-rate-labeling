import json
import logging
from typing import Dict, List, Any, Tuple

import pandas as pd
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from openai.types.chat import ChatCompletion
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from omegaconf import DictConfig

from .parsing import validate_response, clean_and_parse_json

# Configure logger for this module
log = logging.getLogger(__name__)

class LLMClient:
    """
    A client for interacting with the LLM API to extract radiology labels.
    Encapsulates retry logic and configuration.
    """
    def __init__(self, cfg: DictConfig):
        self.client = OpenAI(api_key=cfg.api.api_key)
        self.model = cfg.api.model
        self.temperature = cfg.api.temperature
        self.system_prompt = cfg.prompt.system_prompt
        self.target_labels = list(cfg.prompt.labels)
        
        # Configure the retry strategy dynamically based on config
        self.retrier = Retrying(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(cfg.api.max_retries),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True
        )

    def _call_api(self, messages: List[Dict[str, str]]) -> ChatCompletion:
        """
        Performs the actual API call. This method is wrapped by the retrier.
        Returns the full ChatCompletion object to access usage stats.
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

    def get_labels(self, report_text: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Extracts binary labels from a single radiology report.
        
        Args:
            report_text: The raw text of the radiology report.
            
        Returns:
            A tuple containing:
            1. Dictionary mapping label names to 0 or 1.
            2. Dictionary containing metadata (token usage).
        """
        # Default metadata for failures/skips
        empty_meta = {"prompt_tokens": 0, "completion_tokens": 0}
        empty_labels = {label: 0 for label in self.target_labels}

        # Handle empty or NaN reports immediately
        if pd.isna(report_text) or str(report_text).strip() == "":
            return empty_labels, empty_meta

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Report:\n{report_text}"}
        ]

        try:
            # Execute the API call with the configured retry policy
            response = self.retrier(self._call_api, messages)
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            # Extract usage statistics
            usage = response.usage
            meta = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0
            }

            # Parse and validate
            parsed_json = clean_and_parse_json(content)
            labels = validate_response(parsed_json, self.target_labels)
            
            return labels, meta
            
        except json.JSONDecodeError:
            log.error(f"Malformed JSON received from LLM. Returning all zeros.")
            return empty_labels, empty_meta
        except Exception as e:
            log.error(f"Failed to extract labels after retries: {e}")
            return empty_labels, empty_meta