import pandas as pd
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import time
from pathlib import Path
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate


import logging

from src.risk_feature_analyzer.prompts.risk_feature_prompt import RULE_MATCHING_PROMPT, amazon_rules
from src.risk_feature_analyzer.extractors.risk_feature_extractor import matching_rules
from src.risk_feature_analyzer.preprocessors.risk_feature_preprocessor import extract_amazon_fields 

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

path = Path("data/checkpoint.parquet")

if not path.exists():
    print("Checkpoint does not exists, processing from scratch...")

    df = pd.read_csv("data/final_output.csv")
    df[[
        "amazon_present",
        "amazon_score_impact",
        "amazon_evidence"
    ]] = df["risk_features"].apply(extract_amazon_fields)
    
    df["amazon_presence_features"] = None
    df["processing_error"] = None

else:
    print("Loading from checkpoint...")
    df = pd.read_parquet("checkpoint.parquet")

CHECKPOINT_EVERY = 10
SLEEP_SECONDS = 0

prompt = ChatPromptTemplate.from_template(RULE_MATCHING_PROMPT)

chain = prompt | llm | StrOutputParser()

for idx, row in df.iterrows():

    print(f"Processing row {idx}...")

    # Skip already processed rows
    if df.at[idx, "amazon_presence_features"] is not None:
        continue

    output, error = matching_rules(chain, row["amazon_evidence"], amazon_rules)

    df.at[idx, "amazon_presence_features"] = output
    df.at[idx, "processing_error"] = error

    #Rate limit
    time.sleep(SLEEP_SECONDS)

    # Periodic checkpoint
    if idx % CHECKPOINT_EVERY == 0:        
        df.to_parquet("checkpoint.parquet")

#Final save
df.to_parquet("checkpoint.parquet")
df.to_csv("final_amazon_output.csv")


# analysis_text = df.iloc[2]["detailedBreakdown"]
# features = extract_risk_features(analysis_text)
# print(features)
