import json
import logging
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

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


class EmptyResponseError(Exception):
    """Raised when the LLM returns an empty response body."""

class LLMClient:
    """High-level wrapper around the OpenAI Chat API for labeling radiology reports."""
    def __init__(self, cfg: DictConfig):
        """Initialize the LLM client with retry policy, prompts, and label config.

        Args:
            cfg: Hydra/OMEGACONF configuration supplying API credentials, model
                metadata, prompt text, labels, and retry behavior.
        """
        api_key = None
        try:
            # Prefer explicit config if provided, otherwise fall back to OPENAI_API_KEY
            api_key = cfg.api.get("api_key")
        except Exception:
            api_key = None

        if isinstance(api_key, str):
            api_key = api_key.strip() or None

        # If api_key is omitted, the OpenAI SDK will read OPENAI_API_KEY from the environment.
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = cfg.api.model
        self.temperature = cfg.api.temperature
        self.reasoning_effort = cfg.api.get("reasoning_effort", "low")
        self.max_completion_tokens = cfg.api.get("max_completion_tokens", 256)
        self.system_prompt = cfg.prompt.system_prompt
        self.target_labels = list(cfg.prompt.labels)
        self.examples_enabled = bool(cfg.prompt.get("examples_enabled", True))
        self.log_requests = bool(cfg.api.get("log_requests", False))
        self.log_requests_include_text = bool(cfg.api.get("log_requests_include_text", False))
        self.log_path = cfg.api.get("log_path", "")
        raw_examples = cfg.prompt.get("examples", [])
        # Load optional few-shot examples; default to empty list if not present
        if self.examples_enabled and raw_examples:
            self.examples = list(raw_examples)
        else:
            self.examples = []
        
        # Configure the retry strategy dynamically based on config
        self.retrier = Retrying(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError, EmptyResponseError)),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            stop=stop_after_attempt(cfg.api.max_retries),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True
        )

    def _append_debug_log(self, payload: Dict[str, Any]) -> None:
        if not self.log_requests:
            return
        try:
            if self.log_path:
                log_file = Path(self.log_path)
                if log_file.parent:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                log.info("LLM call debug: %s", payload)
        except Exception:
            log.exception("Failed to write debug log payload.")

    def _build_request_log(self, report_text: str, active_labels: List[str], messages: List[Dict[str, str]]) -> Dict[str, Any]:
        report_hash = hashlib.sha256(str(report_text).encode("utf-8")).hexdigest()
        payload = {
            "event": "request",
            "timestamp_utc": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "model": self.model,
            "labels": active_labels,
            "report_hash": report_hash,
            "messages_count": len(messages),
        }
        if self.log_requests_include_text:
            payload["messages"] = messages
        return payload

    def _build_response_log(
        self,
        request_id: str,
        model_version: str,
        content: str,
        meta: Dict[str, Any],
        response_meta: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        response_meta = response_meta or {}
        payload = {
            "event": "response",
            "timestamp_utc": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "request_id": request_id,
            "model_version": model_version,
            "content_length": len(content) if content is not None else 0,
            "finish_reason": response_meta.get("finish_reason", ""),
            "tool_calls_present": response_meta.get("tool_calls_present", False),
            "status": meta.get("status", ""),
            "error_message": meta.get("error_message", ""),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "total_tokens": meta.get("total_tokens", 0),
        }
        if self.log_requests_include_text:
            payload["content"] = content
        return payload

    def _base_meta(self, status: str) -> Dict[str, Any]:
        """Create a fresh metadata dict so callers can extend safely.

        Args:
            status: Initial status string (e.g., "success", "error", "skipped").

        Returns:
            Baseline metadata dictionary with zeroed token counts and timestamps.
        """
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

    def _call_responses_api(self, messages: List[Dict[str, str]]) -> Any:
        """Execute request using the new v1/responses endpoint (for gpt-5-pro)."""
        # Map 'messages' to 'input' and handle parameter renaming cleanly
        request_kwargs = {
            "model": self.model,
            "input": messages,
            "text": {"format": {"type": "json_object"}}
        }
        
        # Map specific parameters to their new names
        if self.reasoning_effort:
            # gpt-5-pro only supports 'high' reasoning effort
            effort = self.reasoning_effort
            if effort == "low":
                effort = "high"
            request_kwargs["reasoning"] = {"effort": effort}
        
        if self.max_completion_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_completion_tokens

        # Use the SDK's native accessor for the new endpoint
        return self.client.responses.create(**request_kwargs)

    def _call_api(self, messages: List[Dict[str, str]]) -> Any:
        """Dispatch the API call to the correct endpoint based on the model."""
        if self.model == "gpt-5-pro":
            return self._call_responses_api(messages)

        # Standard Chat Completion Logic
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"}
        }
        if self.reasoning_effort:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            request_kwargs["temperature"] = self.temperature

        if self.max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = self.max_completion_tokens

        return self.client.chat.completions.create(**request_kwargs)

    def _extract_response_data(self, response: Any) -> Tuple[str, Any, str, str]:
        """Normalize response data from either ChatCompletion or Response objects.
        
        Returns:
            Tuple of (content_string, usage_object, request_id, model_version)
        """
        # Prefer standard ChatCompletion structure when choices are present
        choices = getattr(response, "choices", None)
        if choices is not None:
            content = choices[0].message.content
            return (
                content,
                response.usage,
                getattr(response, "id", "") or "",
                getattr(response, "model", None) or self.model
            )

        # Fallback to the new Response object structure (gpt-5-pro)
        if hasattr(response, "output_text"):
            return (
                response.output_text,
                response.usage,
                getattr(response, "id", "") or "",
                getattr(response, "model", None) or self.model
            )

        raise ValueError("Unsupported response object structure.")

    def _extract_response_meta(self, response: Any) -> Dict[str, Any]:
        """Extract finish reason or tool-call indicators for debugging."""
        meta: Dict[str, Any] = {}
        choices = getattr(response, "choices", None)
        if choices is not None:
            first_choice = choices[0]
            meta["finish_reason"] = getattr(first_choice, "finish_reason", "") or ""
            tool_calls = getattr(getattr(first_choice, "message", None), "tool_calls", None)
            meta["tool_calls_present"] = bool(tool_calls)
            return meta

        meta["finish_reason"] = getattr(response, "finish_reason", "") or ""
        meta["tool_calls_present"] = False
        return meta

    def _format_user_message(self, report_text: str, labels: List[str]) -> str:
        """Create the user-visible text that lists the report and requested labels.

        Args:
            report_text: Radiology report snippet to embed in the prompt.
            labels: Sequence of label names to enumerate for the LLM.

        Returns:
            A formatted string describing the report and desired findings list.
        """
        label_list = labels or self.target_labels
        label_lines = "\n".join(f"- {label}" for label in label_list)
        return (
            f"Report:\n\"\"\"\n{report_text}\n\"\"\"\n\n"
            f"Target findings:\n{label_lines}\n"
            "Return a JSON object that only contains these findings as keys with 0/1 values."
        )

    def _resolve_active_labels(self, labels_override: Sequence[str] | None) -> List[str]:
        """Return the sanitized label list for the current request.

        Args:
            labels_override: Optional single label or iterable overriding defaults.

        Returns:
            A non-empty list of labels drawn from either the override or defaults.
        """
        if labels_override:
            if isinstance(labels_override, str):
                candidate_source = [labels_override]
            else:
                candidate_source = labels_override
            cleaned = [str(label).strip() for label in candidate_source if str(label).strip()]
            if cleaned:
                return cleaned
        return list(self.target_labels)

    def get_labels(self, report_text: str, labels_override: Sequence[str] | None = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Extract binary labels and metadata for a single radiology report.

        Args:
            report_text: Raw report text that should be labeled.
            labels_override: Optional subset of labels; defaults to all configured labels.

        Returns:
            Tuple where the first element is ``{label: 0|1}`` predictions and the
            second element is a metadata dictionary containing token usage, latency,
            request identifiers, and status/error information.
        """
        active_labels = self._resolve_active_labels(labels_override)
        # Default metadata for failures/skips
        empty_meta = self._base_meta(status="skipped")
        empty_labels = {label: 0 for label in active_labels}

        # Handle empty or NaN reports immediately
        if pd.isna(report_text) or str(report_text).strip() == "":
            return empty_labels, empty_meta

        # Construct the conversation history
        messages = [{"role": "system", "content": self.system_prompt}]

        # Inject few-shot examples if configured
        for example in self.examples:
            ex_report = example.get("report", "")
            
            # Parse and filter example output to match the requested labels
            raw_output = example.get("output", {})
            if isinstance(raw_output, str):
                try:
                    raw_output = json.loads(raw_output)
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON

            if isinstance(raw_output, dict):
                # Filter keys to only those in active_labels
                filtered_output = {k: v for k, v in raw_output.items() if k in active_labels}
                ex_output = json.dumps(filtered_output)
            else:
                # Fallback for non-dict outputs
                ex_output = str(raw_output)

            # The example prompt should request exactly the same labels we are currently asking for
            messages.append({"role": "user", "content": self._format_user_message(ex_report, active_labels)})
            messages.append({"role": "assistant", "content": ex_output})

        # Append the actual target report
        messages.append({"role": "user", "content": self._format_user_message(report_text, active_labels)})
        self._append_debug_log(self._build_request_log(report_text, active_labels, messages))

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        request_id = ""
        model_version = self.model
        start_time = time.perf_counter()
        started_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        attempt_counter = {"count": 0}
        last_response_meta: Dict[str, Any] = {}

        def tracked_call(call_messages: List[Dict[str, str]]) -> Tuple[str, Any, str, str, Dict[str, Any]]:
            attempt_counter["count"] += 1
            response = self._call_api(call_messages)
            response_meta = self._extract_response_meta(response)
            last_response_meta.update(response_meta)
            content, usage, req_id, model_ver = self._extract_response_data(response)
            if not content or not str(content).strip():
                raise EmptyResponseError("Empty response from LLM")
            return content, usage, req_id, model_ver, response_meta

        try:
            # Execute the API call with the configured retry policy
            content, usage, request_id, model_version, response_meta = self.retrier(tracked_call, messages)
            latency = time.perf_counter() - start_time
            ended_at = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            
            # Extract usage statistics
            if usage:
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens)
            else:
                prompt_tokens = completion_tokens = total_tokens = 0
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
            labels = validate_response(parsed_json, active_labels)
            
            self._append_debug_log(self._build_response_log(request_id, model_version, content, meta, response_meta))
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
            self._append_debug_log(self._build_response_log(request_id, model_version, "", meta, last_response_meta))
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
            self._append_debug_log(self._build_response_log(request_id, model_version, "", meta, last_response_meta))
            return empty_labels, meta