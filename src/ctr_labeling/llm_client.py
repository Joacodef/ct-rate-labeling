import json
import logging
import time
from datetime import datetime
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

    def _base_meta(self, status: str) -> Dict[str, Any]:
        """Create a fresh metadata dict so callers can extend safely."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "status": status,
            "error_message": "",
            "request_id": "",
            "model_version": "",
            "retry_count": 0,
            "started_at_utc": "",
            "ended_at_utc": ""
        }

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

    def get_labels(self, report_text: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Extracts binary labels from a single radiology report.
        
        Args:
            report_text: The raw text of the radiology report.
            
        Returns:
            A tuple containing:
            1. Dictionary mapping label names to 0 or 1.
            2. Metadata dictionary with token usage, latency, status, and error details.
        """
        # Default metadata for failures/skips
        empty_meta = self._base_meta(status="skipped")
        empty_labels = {label: 0 for label in self.target_labels}

        # Handle empty or NaN reports immediately
        if pd.isna(report_text) or str(report_text).strip() == "":
            return empty_labels, empty_meta

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Report:\n{report_text}"}
        ]

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        request_id = ""
        model_version = self.model
        start_time = time.perf_counter()
        started_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        attempt_counter = {"count": 0}

        def tracked_call(call_messages: List[Dict[str, str]]) -> ChatCompletion:
            attempt_counter["count"] += 1
            return self._call_api(call_messages)

        try:
            # Execute the API call with the configured retry policy
            response = self.retrier(tracked_call, messages)
            latency = time.perf_counter() - start_time
            ended_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            # Extract usage statistics
            usage = response.usage
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens)
            else:
                prompt_tokens = completion_tokens = total_tokens = 0
            request_id = getattr(response, "id", "") or ""
            model_version = getattr(response, "model", None) or self.model
            meta = self._base_meta(status="success")
            meta.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency,
                "request_id": request_id,
                "model_version": model_version,
                "retry_count": attempt_counter["count"],
                "started_at_utc": started_at,
                "ended_at_utc": ended_at
            })

            # Parse and validate
            parsed_json = clean_and_parse_json(content)
            labels = validate_response(parsed_json, self.target_labels)
            
            return labels, meta
            
        except json.JSONDecodeError:
            latency = time.perf_counter() - start_time
            ended_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            log.error("Malformed JSON received from LLM. Returning all zeros.")
            meta = self._base_meta(status="error")
            meta.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency,
                "error_message": "Malformed JSON response",
                "request_id": request_id,
                "model_version": model_version,
                "retry_count": attempt_counter["count"],
                "started_at_utc": started_at,
                "ended_at_utc": ended_at
            })
            return empty_labels, meta
        except Exception as e:
            latency = time.perf_counter() - start_time
            ended_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            log.error(f"Failed to extract labels after retries: {e}")
            meta = self._base_meta(status="error")
            meta.update({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency,
                "error_message": str(e),
                "request_id": request_id,
                "model_version": model_version,
                "retry_count": attempt_counter["count"],
                "started_at_utc": started_at,
                "ended_at_utc": ended_at
            })
            return empty_labels, meta