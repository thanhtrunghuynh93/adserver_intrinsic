import pandas as pd
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import time
import json
from pathlib import Path

from utils import extract_columns_from_json, RISK_FEATURE_PROMPT
from openai import OpenAIError   # if using OpenAI SDK
from json import JSONDecodeError
import logging

from src.risk_feature_analyzer.prompts.risk_feature_prompt import RISK_FEATURE_PROMPT
from src.risk_feature_analyzer.extractors.risk_feature_extractor import extract_risk_features

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize GPT via LangChain
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)

path = Path("checkpoint.parquet")

if not path.exists():
    print("Checkpoint does not exists, processing from scratch...")

    df = pd.read_csv("data/adserver.csv")
    df = df[df["status"] == "FINISHED"]
    df = df[["website", "detect_result", "risk_score", "violation_details"]]
    target_cols = ["policyViolations", "summary", "detailedBreakdown"]

    df = extract_columns_from_json(
        df,
        source_col="violation_details",
        target_cols=target_cols
    )

    df["risk_features"] = None
    df["processing_error"] = None

else:
    print("Loading from checkpoint...")
    df = pd.read_parquet("checkpoint.parquet")

CHECKPOINT_EVERY = 10
SLEEP_SECONDS = 1.2
chain = RISK_FEATURE_PROMPT | llm

for idx, row in df.iterrows():

    print(f"Processing row {idx}...")

    # Skip already processed rows
    if df.at[idx, "risk_features"] is not None:
        continue

    output, error = extract_risk_features(chain, row["detailedBreakdown"])

    df.at[idx, "risk_features"] = output
    df.at[idx, "processing_error"] = error

    #Rate limit
    time.sleep(SLEEP_SECONDS)

    # Periodic checkpoint
    if idx % CHECKPOINT_EVERY == 0:
        df.to_parquet("checkpoint.parquet")

#Final save
df.to_parquet("checkpoint.parquet")
df.to_csv("final_output.csv")


# analysis_text = df.iloc[2]["detailedBreakdown"]
# features = extract_risk_features(analysis_text)
# print(features)
