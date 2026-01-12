from openai import OpenAIError   # if using OpenAI SDK
from json import JSONDecodeError
import logging
import json

logger = logging.getLogger(__name__)

def extract_risk_features(chain, analysis_text: str):
    try:
        response = chain.invoke({"input_text": analysis_text})
        
        if not hasattr(response, "content"):
            return None, "LLM_RESPONSE_MISSING_CONTENT"

        try:
            parsed = json.loads(response.content)
            return parsed, None

        except JSONDecodeError as e:
            logger.warning("JSON parsing failed: %s", response.content[:500])
            return None, f"JSON_DECODE_ERROR: {str(e)}"

    except OpenAIError as e:
        # Covers rate limit, auth, timeout, server errors
        logger.error("OpenAI API error", exc_info=True)
        return None, f"OPENAI_API_ERROR: {str(e)}"

    except ValueError as e:
        # LangChain prompt / formatting errors
        logger.error("Chain invocation error", exc_info=True)
        return None, f"CHAIN_ERROR: {str(e)}"

    except Exception as e:
        # Truly unexpected errors
        logger.exception("Unexpected error in extract_risk_features")
        return None, f"UNEXPECTED_ERROR: {type(e).__name__}: {str(e)}"
