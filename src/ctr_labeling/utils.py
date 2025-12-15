import logging
from typing import Any, Set

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Keep track of models we've warned about to avoid spamming logs
UNPRICED_MODELS_WARNED: Set[str] = set()

def estimate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_table: dict,
    precision: int = 8
) -> float:
    """Estimate the USD cost of a request using the provided pricing table.

    Args:
        model_name: Name/version reported by the API (used for table lookup).
        prompt_tokens: Number of input tokens sent to the model.
        completion_tokens: Number of output tokens returned by the model.
        pricing_table: Mapping of model identifiers to per-million token costs.
        precision: Decimal places to round the resulting cost to.

    Returns:
        Rounded cost estimate in USD; defaults to 0 if pricing data is missing.
    """
    if not pricing_table:
        if model_name not in UNPRICED_MODELS_WARNED:
            log.warning(
                "No pricing table configured. Estimated cost for model '%s' will be 0.",
                model_name
            )
            UNPRICED_MODELS_WARNED.add(model_name)
        return 0.0

    model_base = model_name.lower()
    # Sort keys by length desc to match specific models first (e.g. gpt-4-turbo before gpt-4)
    sorted_models = sorted(
        ((key.lower(), value) for key, value in pricing_table.items()),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for match_key, price_info in sorted_models:
        if match_key in model_base:
            input_price = price_info.get("input_per_million")
            output_price = price_info.get("output_per_million")
            
            if input_price is None or output_price is None:
                if model_name not in UNPRICED_MODELS_WARNED:
                    log.warning(
                        "Pricing entry '%s' lacks input/output rates; reporting 0 cost.",
                        match_key
                    )
                    UNPRICED_MODELS_WARNED.add(model_name)
                return 0.0

            cost = (prompt_tokens / 1_000_000 * input_price) + \
                   (completion_tokens / 1_000_000 * output_price)
            return round(cost, precision)

    if model_base not in UNPRICED_MODELS_WARNED:
        log.warning(
            "No pricing data matched model '%s'. Estimated cost will be reported as 0.",
            model_name
        )
        UNPRICED_MODELS_WARNED.add(model_base)
    return 0.0


def safe_cfg_to_yaml(cfg: DictConfig) -> str:
    """Serialize a config to YAML while redacting common secret fields.

    This is intended for logging/debugging only.
    """

    def redact_inplace(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                key_lower = str(key).lower()
                if key_lower in {"api_key", "apikey", "token", "secret", "authorization"}:
                    obj[key] = "***REDACTED***"
                else:
                    redact_inplace(value)
        elif isinstance(obj, list):
            for item in obj:
                redact_inplace(item)

    try:
        data = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        # As a last resort, fall back to stringification.
        return str(cfg)

    redact_inplace(data)
    return OmegaConf.to_yaml(OmegaConf.create(data))