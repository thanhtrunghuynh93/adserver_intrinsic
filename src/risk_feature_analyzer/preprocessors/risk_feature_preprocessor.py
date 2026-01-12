from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re
import pandas as pd
import ast

def extract_amazon_fields(raw_value):
    """
    Extract amazon_present, amazon_score_impact, amazon_evidence
    from a stringified Python dict.
    """
    if not isinstance(raw_value, str):
        return pd.Series([None, None, None])

    try:
        parsed = ast.literal_eval(raw_value)
        amazon = parsed.get("amazon_presence", {})
        return pd.Series([
            amazon.get("present"),
            amazon.get("score_impact"),
            amazon.get("evidence")
        ])
    except (ValueError, SyntaxError):
        return pd.Series([None, None, None])

class RiskTextPreprocessor:
    def __init__(self, add_category_tokens: bool = True):
        self.add_category_tokens = add_category_tokens

        # Canonical phrase mappings (regex → token)
        self.canonical_patterns = {
            r"\bno prohibited\b.*": "no_prohibited_content",
            r"\bno restricted\b.*": "no_prohibited_content",
            r"\bno political\b.*": "no_prohibited_content",
            r"\bno porn\b.*": "no_prohibited_content",

            r"\bless than 30 reviews\b": "amazon_below_30_reviews",
            r"\bamazon\b.*\b(brand store|storefront)\b": "amazon_brand_store",

            r"\blinkedin\b.*\b\d+\s*k\b.*\bfollowers\b": "linkedin_large_following",
            r"\binstagram\b.*\b\d+\s*k\b.*\bfollowers\b": "instagram_large_following",

            r"\bdomain\b.*\bcreated\b.*\b202\d\b": "new_domain",
            r"\bdomain\b.*\bmulti[- ]?year\b": "old_domain",
        }

        # Boilerplate removal patterns
        self.boilerplate_patterns = [
            r"http\S+",
            r"evidence\s*:",
            r"score impact\s*:",
            r"\(https?:.*?\)",
        ]

    @staticmethod
    def _strip_thousands_separators(text: str) -> str:
        """
        Convert '6,425' -> '6425', '1,200' -> '1200'
        """
        return re.sub(r"(?<=\d),(?=\d)", "", text)

    @staticmethod
    def _normalize_trustpilot_ratings(text: str) -> str:
        """
        Replace patterns like '4.2/5', '4.2 / 5', '4.2 out of 5' with tokens.
        Avoid leaving trailing '5'.
        """
        # Handle "X.X/5" or "X.X / 5"
        def repl(match):
            rating = float(match.group(1))
            if rating >= 3.5:
                return "trustpilot_positive_rating"
            if rating <= 2.9:
                return "trustpilot_negative_rating"
            return "trustpilot_mixed_rating"

        text = re.sub(r"\b([0-5]\.\d)\s*/\s*5\b", repl, text)
        text = re.sub(r"\b([0-5]\.\d)\s+out\s+of\s+5\b", repl, text)
        return text

    @staticmethod
    def _bucket_review_counts(text: str) -> str:
        """
        Bucket review counts: '6425 reviews' -> 'reviews_100_plus', etc.
        Assumes thousands separators are already stripped.
        """
        # Example: "6425 reviews"
        def repl_reviews(match):
            n = int(match.group(1))
            if n >= 100:
                return "reviews_100_plus"
            if n >= 30:
                return "reviews_30_99"
            if n >= 10:
                return "reviews_10_29"
            return "reviews_lt_10"

        # Match "<number> reviews" (supports large integers)
        return re.sub(r"\b(\d{1,7})\s+reviews\b", repl_reviews, text)

    def _add_category_tokens(self, text: str) -> str:
        tokens = []
        if "amazon" in text:
            tokens.append("[AMAZON]")
        if "trustpilot" in text or "reviews" in text or "reviews_" in text:
            tokens.append("[REVIEWS]")
        if "linkedin" in text or "instagram" in text or "tiktok" in text:
            tokens.append("[SOCIAL]")
        if "domain" in text:
            tokens.append("[DOMAIN]")
        if "policy" in text or "prohibited" in text:
            tokens.append("[POLICY]")
        return (" ".join(tokens) + " " + text) if tokens else text

    def preprocess(self, text: Optional[str]) -> str:
        if not text:
            return ""

        # 1) Lowercase
        text = text.lower()

        # 2) Strip URLs/boilerplate
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, " ", text)

        # 3) Normalize numeric formatting early
        text = self._strip_thousands_separators(text)

        # 4) Normalize Trustpilot rating expressions (removes trailing '/5')
        text = self._normalize_trustpilot_ratings(text)

        # 5) Canonicalize phrases
        for pattern, token in self.canonical_patterns.items():
            text = re.sub(pattern, token, text)

        # 6) Bucket review counts (after commas removed)
        text = self._bucket_review_counts(text)

        # 7) Remove punctuation (keep brackets for category tokens)
        text = re.sub(r"[^\w\s\[\]]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 8) Add category tokens
        if self.add_category_tokens:
            text = self._add_category_tokens(text)

        return text

