import json
import re
import logging
from typing import Dict, List, Any

# Configure logger for this module
log = logging.getLogger(__name__)

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Cleans markdown formatting (e.g., ```json ... ```) and parses JSON.
    
    Args:
        text: The raw string response from the LLM.
        
    Returns:
        Parsed dictionary.
        
    Raises:
        json.JSONDecodeError: If parsing fails.
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
    """
    Converts various representations of truth/falsehood to 0 or 1.
    
    Supported:
    - Integers: 0, 1
    - Strings: "0", "1", "yes", "no", "true", "false", "present", "absent"
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
    """
    Validates and normalizes the LLM response.
    
    Args:
        response_json: The parsed JSON object from the LLM.
        target_labels: The list of expected abnormality labels.
        
    Returns:
        A dictionary with guaranteed 0 or 1 values for all target labels.
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