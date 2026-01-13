from langchain.prompts import PromptTemplate

RISK_FEATURE_PROMPT = PromptTemplate(
    input_variables=["analysis_text"],
    template="""
You are extracting risk-scoring factors from an analysis.

CRITICAL OUTPUT REQUIREMENTS (MUST FOLLOW EXACTLY):
- You MUST output ALL 9 criteria listed in the schema.
- You MUST include every key exactly once.
- If a criterion is NOT explicitly mentioned in the text, set:
  - "present": false
  - "score_impact": 0
  - "evidence": "Not stated."
- NEVER omit a criterion.
- NEVER stop early.
- Output ALL fields even if most are false.

DO NOT summarize.
DO NOT explain.
DO NOT make a decision.
DO NOT compute totals or final scores.
IGNORE any stated baseline, aggregate, or final risk score.
IGNORE step-level or section-level summary scores.

The analysis is structured into:
- Step 1: On-site / landing page signals
- Step 2: External / third-party signals
- Step 3: Explicit risk scoring criteria

Extract ONLY explicitly stated criteria and score impacts from Steps 1, 2, and 3.
Do NOT infer, estimate, reconcile, or correct scores.

Return JSON ONLY in the exact schema below.

Schema:
{{
  "policy_violation": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "landing_page_accessible": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "amazon_presence": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "major_retailer_presence": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "social_media_presence": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "reviews_presence": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "domain_or_corporate_signal": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "other_red_flags": {{ "present": true_or_false, "score_impact": number, "evidence": string }},
  "mitigating_signals": {{ "present": true_or_false, "score_impact": number, "evidence": string }}
}}

Rules:
- Use ONLY information explicitly stated in the text.
- If multiple score impacts appear for one criterion, use the strongest stated score.
- Evidence must be factual and fully covered.
- Use true/false only.
- Output MUST contain exactly 9 top-level keys.
- Output MUST be valid JSON.

Text:
<<<
{input_text}
>>>
"""
)

RULE_DISCOVERY_PROMPT = """
You are performing rule discovery from evidence texts.

You are given:
1) A list of EXISTING_RULES
2) A batch of EVIDENCE_EXAMPLES

Your task:
- Identify NEW, interpretable rules that are NOT already covered by EXISTING_RULES
- Each rule must be:
  • Binary (evaluates to true or false)
  • Deterministic from text alone
  • Reusable across multiple cases
  • Domain-specific to Amazon presence / marketplace validation
- DO NOT restate, rename, or slightly rephrase existing rules
- DO NOT invent scores, thresholds, or probabilities
- DO NOT summarize or explain the evidence examples
- DO NOT merge multiple conditions into a single rule

For each NEW rule:
- Provide a concise snake_case rule_name
- Provide a single-sentence rule_definition

If no new rules are found, return an empty list.

Return JSON ONLY using the schema below.

Schema:
{{
  "new_rules": [
    {{
      "rule_name": "string",
      "rule_definition": "string"
    }}
  ]
}}

EXISTING_RULES:
{existing_rules}

EVIDENCE_EXAMPLES:
{examples}
"""

RULE_MATCHING_PROMPT = """
You are performing rule matching against Amazon presence rules.

You are given:
1) A list of RULES, each with a rule_name and rule_definition
2) A single EVIDENCE_TEXT describing a brand or product

Your task:
- Identify which rules in RULES are TRUE based strictly on the EVIDENCE_TEXT
- At least 1 rule matches with the evidence provided.

STRICT CONSTRAINTS:
- Base decisions ONLY on the provided EVIDENCE_TEXT
- DO NOT infer, assume, or use outside knowledge
- DO NOT invent facts, numbers, or thresholds
- DO NOT restate or paraphrase rule definitions
- DO NOT explain your reasoning
- DO NOT include unmatched rules
- DO NOT include commentary, markdown, or text outside the JSON

OUTPUT REQUIREMENTS:
- Return VALID JSON ONLY
- Use EXACTLY the schema below
- Each returned rule_name MUST exist in AMAZON_RULES
- If no rules are matched, return an empty list

Schema (copy exactly):
{{
  "matched_rules": [
    {{
      "rule_name": "string"
    }}
  ]
}}

RULES:
{rules}

EVIDENCE_TEXT:
{evidence_text}
"""

amazon_rules = [

    # ─────────────────────────────────────────────
    # A. No / Absent Amazon Presence
    # ─────────────────────────────────────────────
    {
        "rule_name": "no_amazon_listing_found",
        "rule_definition": "No Amazon listings are found for the brand or its products after direct search attempts."
    },
    {
        "rule_name": "no_amazon_storefront_found",
        "rule_definition": "No official Amazon brand storefront or brand page is found."
    },
    {
        "rule_name": "brand_sold_exclusively_off_amazon",
        "rule_definition": "The brand’s products are sold exclusively on the brand’s own website or non-Amazon channels."
    },
    {
        "rule_name": "service_or_digital_only_brand",
        "rule_definition": "The entity is a service-based or app-only brand with no physical products sold on Amazon."
    },

    # ─────────────────────────────────────────────
    # B. Weak or Limited Amazon Presence
    # ─────────────────────────────────────────────
    {
        "rule_name": "limited_listing_coverage",
        "rule_definition": "Amazon listings exist but are few in number or lack meaningful visibility."
    },
    {
        "rule_name": "low_review_volume",
        "rule_definition": "Amazon listings exist but none have 30 or more reviews."
    },
    {
        "rule_name": "limited_ratings_presence",
        "rule_definition": "Products are present on Amazon but have fewer than 30 ratings overall."
    },
    {
        "rule_name": "no_verified_brand_store",
        "rule_definition": "No verified Amazon brand store is found for the brand."
    },

    # ─────────────────────────────────────────────
    # C. Brand Attribution & Listing Quality Issues
    # ─────────────────────────────────────────────
    {
        "rule_name": "products_listed_under_other_brands",
        "rule_definition": "Products appear on Amazon but are listed under generic, incorrect, or unrelated brand names."
    },
    {
        "rule_name": "incorrect_or_missing_brand_tag",
        "rule_definition": "Listings exist but have incorrect or missing brand tags."
    },
    {
        "rule_name": "listings_fragmented_across_brands",
        "rule_definition": "The same products are fragmented across multiple brand names or seller identities."
    },
    {
        "rule_name": "generic_or_unrelated_search_results",
        "rule_definition": "Search results for the brand return mostly unrelated or generic items."
    },

    # ─────────────────────────────────────────────
    # D. Third-Party–Only Presence
    # ─────────────────────────────────────────────
    {
        "rule_name": "third_party_only_listings",
        "rule_definition": "Products are sold exclusively by third-party sellers without official brand participation."
    },

    # ─────────────────────────────────────────────
    # E. Policy / Structural Reminder Constraints
    # ─────────────────────────────────────────────
    {
        "rule_name": "amazon_category_restrictions",
        "rule_definition": "Amazon restricts or prohibits sales in the product category associated with the brand."
    },
    {
        "rule_name": "brand_discourages_amazon_resale",
        "rule_definition": "The brand’s terms explicitly discourage or prohibit resale on Amazon."
    },

    # ─────────────────────────────────────────────
    # F. Positive / Strong Amazon Signals
    # ─────────────────────────────────────────────
    {
        "rule_name": "strong_amazon_footprint",
        "rule_definition": "The brand has multiple well-referenced listings with consistent branding on Amazon."
    },
    {
        "rule_name": "official_brand_sales_on_amazon",
        "rule_definition": "The brand sells products directly through Amazon."
    },
    {
        "rule_name": "brand_participates_in_amazon_luxury_stores",
        "rule_definition": "The brand participates in Amazon Luxury Stores."
    },
    {
        "rule_name": "external_links_point_to_amazon",
        "rule_definition": "The brand’s official external links (e.g., Linktree) point to Amazon product pages."
    }
]

