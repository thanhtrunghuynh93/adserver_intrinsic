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
