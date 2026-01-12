from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import json
from json import JSONDecodeError
import logging
import re

from openai import OpenAIError  # if using OpenAI SDK

logger = logging.getLogger(__name__)


def extract_risk_features(chain: Any, analysis_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Invoke a chain to extract risk features from analysis_text.
    Returns (parsed_json, error_str). If successful, error_str is None.
    """
    try:
        response = chain.invoke({"input_text": analysis_text})

        # Normalize response text
        if hasattr(response, "content"):
            response_text = response.content
        elif isinstance(response, dict):
            response_text = response.get("text") or response.get("output") or str(response)
        else:
            response_text = str(response)

        try:
            parsed = json.loads(response_text)
            return parsed, None
        except JSONDecodeError as e:
            logger.warning("JSON parsing failed for LLM response (truncated): %s", response_text[:500])
            return None, f"JSON_DECODE_ERROR: {str(e)}"

    except OpenAIError as e:
        logger.error("OpenAI API error during extract_risk_features", exc_info=True)
        return None, f"OPENAI_API_ERROR: {str(e)}"
    except ValueError as e:
        logger.error("Chain invocation / formatting error during extract_risk_features", exc_info=True)
        return None, f"CHAIN_ERROR: {str(e)}"
    except Exception as e:
        logger.exception("Unexpected error in extract_risk_features")
        return None, f"UNEXPECTED_ERROR: {type(e).__name__}: {str(e)}"


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove starting fence and optional "json"
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t


def discover_new_rules(rule_discovery_chain: Any, example_batch: List[Dict[str, Any]], existing_rules: List[str]) -> List[Dict[str, Any]]:
    """
    Ask the rule discovery chain to produce new rules given a batch of examples and existing rules.
    Returns a list of rule objects (dicts). Raises ValueError on invalid / unexpected LLM output.
    """
    try:
        response = rule_discovery_chain.invoke({
            "examples": example_batch,
            "existing_rules": existing_rules,
        })
    except Exception as e:
        logger.exception("Error invoking rule_discovery_chain")
        raise ValueError(f"Chain invocation failed: {type(e).__name__}: {e}")

    # Normalize response to string
    if hasattr(response, "content"):
        response_text = response.content
    elif isinstance(response, dict):
        response_text = response.get("text") or response.get("output") or str(response)
    else:
        response_text = str(response)

    logger.debug("Raw rule discovery response: %s", (response_text[:1000] + "...") if len(response_text) > 1000 else response_text)

    response_text = _strip_code_fence(response_text)

    try:
        parsed = json.loads(response_text)
    except JSONDecodeError as e:
        logger.warning("LLM returned invalid JSON for rule discovery", exc_info=False)
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw output:\n{response_text}")

    new_rules = parsed.get("new_rules")
    if new_rules is None:
        raise ValueError("JSON parsed but missing 'new_rules' field")
    if not isinstance(new_rules, list):
        raise ValueError("'new_rules' must be a list")

    return new_rules


def iterative_rule_discovery(rule_discovery_chain,
    all_examples: List[Dict[str, Any]],
    seed_rules: Iterable[str],
    batch_size: int = 50,
    max_rounds: int = 1,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Iteratively run rule discovery over batches of examples.
    Returns (new_rule_objects, discovered_rule_names).
    """
    discovered_rules: Set[str] = set(seed_rules)
    new_rule_objects: List[Dict[str, Any]] = []

    for round_idx in range(max_rounds):
        logger.info("=== Discovery Round %d ===", round_idx + 1)
        round_new: List[Dict[str, Any]] = []

        for i in range(0, len(all_examples), batch_size):
            print(i)
            batch = all_examples[i : i + batch_size]
            try:
                new_rules = discover_new_rules(rule_discovery_chain, example_batch=batch, existing_rules=list(discovered_rules))
            except Exception as e:
                # If one batch fails, log and continue with other batches/rounds
                logger.error("discover_new_rules failed for batch starting at %d: %s", i, str(e))
                continue

            for rule in new_rules:
                # Defensive access to rule_name
                name = rule.get("rule_name") if isinstance(rule, dict) else None
                if not name:
                    logger.warning("Skipping invalid rule object (missing rule_name): %s", rule)
                    continue
                if name not in discovered_rules:
                    discovered_rules.add(name)
                    round_new.append(rule)

        if not round_new:
            logger.info("No new rules found in round %d. Stopping.", round_idx + 1)
            break

        new_rule_objects.extend(round_new)
        logger.info("Discovered %d new rules in round %d", len(round_new), round_idx + 1)

    return new_rule_objects, list(discovered_rules)

