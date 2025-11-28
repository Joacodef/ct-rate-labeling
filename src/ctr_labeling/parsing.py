import json
import re
import logging
from typing import Dict, List, Any

# Configure logger for this module
log = logging.getLogger(__name__)

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Strip code fences from an LLM response and parse it as JSON.

    Args:
        text: Raw assistant response which might include markdown fences.

    Returns:
        Parsed Python object constructed from the JSON payload.

    Raises:
        json.JSONDecodeError: If JSON parsing fails after cleanup.
    """
    # Remove markdown code blocks
    # Pattern: ```json ... ``` or just ``` ... ```
    # We use DOTALL to match newlines inside the block
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        text = match.group(1)
        
    return json.loads(text)

def normalize_binary_value(value: Any) -> int:
    """Convert a truthy/falsy representation to a canonical 0 or 1.

    Args:
        value: Mixed-type input (int or string) describing presence/absence.

    Returns:
        ``1`` when the value is interpreted as positive/present, ``0`` otherwise.
    """
    if isinstance(value, int):
        # Treat any non-zero integer as 1, though we expect 0 or 1
        return 1 if value != 0 else 0
    
    if isinstance(value, str):
        v_lower = value.strip().lower()
        if v_lower in ["1", "yes", "true", "present", "positive"]:
            return 1
        if v_lower in ["0", "no", "false", "absent", "negative"]:
            return 0
            
    # Default to 0 for unknown types or values
    return 0

def validate_response(response_json: Dict[str, Any], target_labels: List[str]) -> Dict[str, int]:
    """Validate and normalize the LLM response for the requested labels.

    Args:
        response_json: Parsed JSON content returned by the LLM.
        target_labels: Ordered list of label names expected in the output.

    Returns:
        Dict with each target label mapped to a normalized 0/1 integer.
    """
    validated = {}
    
    # Create a case-insensitive map of the response keys for robust lookup
    response_keys_map = {k.lower(): v for k, v in response_json.items()}
    
    for label in target_labels:
        # Try exact match first, then case-insensitive match
        val = response_json.get(label)
        if val is None:
            val = response_keys_map.get(label.lower())
            
        # If the key is completely missing, default to 0 (absent)
        if val is None:
            val = 0
            
        validated[label] = normalize_binary_value(val)
        
    return validated