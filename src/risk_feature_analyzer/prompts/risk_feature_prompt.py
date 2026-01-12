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
- A rule is TRUE only if the evidence clearly and explicitly satisfies the rule_definition
- If a rule is not clearly supported, DO NOT include it

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
    {
      "rule_name": "no_official_amazon_brand_store",
      "rule_definition": "No official Amazon brand store or brand page is found for the brand."
    },
    {
      "rule_name": "no_product_listings_with_30_plus_reviews",
      "rule_definition": "No product listings with 30 or more reviews are found on Amazon."
    },
    {
      "rule_name": "unrelated_items_appear_in_search",
      "rule_definition": "Unrelated items appear in Amazon search results for the brand."
    },
    {
      "rule_name": "product_listed_under_different_brand",
      "rule_definition": "A product is found on Amazon but is listed under a different brand name than expected."
    },
    {
      "rule_name": "no_verified_brand_store",
      "rule_definition": "No verified Amazon brand store is found for the brand."
    },
    {
      "rule_name": "unrelated_third_party_listings",
      "rule_definition": "Only unrelated third-party listings are found using the brand's name on Amazon."
    },
    {
      "rule_name": "no_brand_tag_found",
      "rule_definition": "No brand tag is found for the product on Amazon."
    },
    {
      "rule_name": "product_sold_by_third_party",
      "rule_definition": "The product is sold by a third-party seller on Amazon."
    },
    {
      "rule_name": "terms_discourage_amazon_resale",
      "rule_definition": "The brand's terms of service explicitly discourage resale on Amazon."
    },
    {
      "rule_name": "brand_sold_exclusively_on_brand_site",
      "rule_definition": "The brand's products are only available on the brand's own website and not on Amazon."
    },
    {
      "rule_name": "service_firm_no_physical_products",
      "rule_definition": "The entity is a service firm and does not offer physical products on Amazon."
    },
    {
      "rule_name": "app_brand_no_physical_products",
      "rule_definition": "The brand is primarily an app and does not have physical products listed on Amazon."
    },
    {
      "rule_name": "no_dedicated_brand_store",
      "rule_definition": "No dedicated brand store or brand-tagged listings are found on Amazon."
    },
    {
      "rule_name": "low_review_count_per_listing",
      "rule_definition": "Amazon listings exist but have low review counts per listing."
    },
    {
      "rule_name": "generic_listings_without_brand_tag",
      "rule_definition": "Only generic listings are found without a specific brand tag on Amazon."
    },
    {
      "rule_name": "brand_listed_under_generic_or_unrelated_brand",
      "rule_definition": "A product is listed on Amazon under a generic or unrelated brand name instead of its own brand."
    },
    {
      "rule_name": "no_official_brand_page_with_30_plus_reviews",
      "rule_definition": "No official Amazon brand page is found with any product having 30 or more reviews."
    },
    {
      "rule_name": "product_listed_with_incorrect_brand_tag",
      "rule_definition": "A product is listed on Amazon with an incorrect brand tag that does not match the product's actual brand."
    },
    {
      "rule_name": "no_amazon_listing_found",
      "rule_definition": "No Amazon listing is found for the brand or product after direct query attempts."
    },
    {
      "rule_name": "no_amazon_storefront_found",
      "rule_definition": "No Amazon storefront is found for the brand after direct query attempts."
    },
    {
      "rule_name": "no_amazon_presence_for_dtc_brands",
      "rule_definition": "Direct-to-consumer (DTC) brands have no presence on Amazon."
    },
    {
      "rule_name": "amazon_restricts_product_sales",
      "rule_definition": "Amazon generally restricts the sales of certain product categories."
    },
    {
      "rule_name": "strong_amazon_footprint",
      "rule_definition": "The entity is widely referenced across multiple listings on Amazon."
    },
    {
      "rule_name": "no_amazon_presence",
      "rule_definition": "The entity is not found on Amazon or major retailers."
    },
    {
      "rule_name": "strong_brand_presence",
      "rule_definition": "The entity is associated with several brands and multiple listings on Amazon."
    },
    {
      "rule_name": "minor_risk_increase",
      "rule_definition": "The entity is not found on Amazon US."
    },
    {
      "rule_name": "limited_ratings_presence",
      "rule_definition": "The product is present on Amazon but has fewer than 30 ratings."
    },
    {
      "rule_name": "direct_brand_sales_on_amazon",
      "rule_definition": "The brand sells products directly on Amazon."
    },
    {
      "rule_name": "third_party_listings_exist",
      "rule_definition": "Listings exist on Amazon but are third-party under different brand names."
    },
    {
      "rule_name": "no_clear_amazon_listings",
      "rule_definition": "No clear Amazon listings exist for a specified product."
    },
    {
      "rule_name": "brand_participates_in_amazon_luxury_stores",
      "rule_definition": "The brand participates in Amazon Luxury Stores."
    },
    {
      "rule_name": "listings_fragmented_under_other_brand_names",
      "rule_definition": "Product listings are fragmented under different brand names on Amazon."
    },
    {
      "rule_name": "prohibited_product_mention",
      "rule_definition": "The text explicitly mentions that Amazon prohibits certain products."
    },
    {
      "rule_name": "linktree_points_to_amazon",
      "rule_definition": "The brand's Linktree points to an Amazon product page."
    }
  ]

