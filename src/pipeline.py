"""
pipeline.py
-----------
End-to-end orchestrator for the Fraud Detection Engine.

Run this file to execute the full pipeline:
  python src/pipeline.py

Steps:
  1. Load raw data
  2. Build + save features
  3. Fit + save FeaturePipeline (for serving)
  4. Train all models
  5. Log experiments to MLflow
"""

import time
from pathlib import Path

from features import load_raw_data, build_features, save_features, FeaturePipeline
from model import train_all

FEATURES_PATH  = "data/processed/features.parquet"
PIPELINE_PATH  = "data/processed/feature_pipeline.pkl"


def run_pipeline(force_rebuild: bool = False):
    start = time.time()

    print("=" * 60)
    print("FRAUD DETECTION ENGINE — FULL PIPELINE")
    print("=" * 60)

    # Step 1 & 2: Feature engineering
    if Path(FEATURES_PATH).exists() and not force_rebuild:
        print(f"\nProcessed features found at {FEATURES_PATH}")
        print("Skipping feature engineering. Pass --force-rebuild to rerun.")
    else:
        print("\n[STEP 1/3] Feature Engineering")
        df_raw      = load_raw_data()
        df_features = build_features(df_raw)
        save_features(df_features, FEATURES_PATH)

        print("\n[STEP 2/3] Fitting and saving FeaturePipeline")
        pipeline = FeaturePipeline()
        pipeline.fit(df_raw)
        pipeline.save(PIPELINE_PATH)

    # Step 3: Model training
    print("\n[STEP 3/3] Model Training")
    results = train_all(FEATURES_PATH)

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed/60:.1f} minutes.")
    print("Launch MLflow UI with:  mlflow ui")
    print("Launch API server with: uvicorn src.serve:app --reload --port 8000")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fraud Detection Engine Pipeline")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild features even if cached version exists")
    args = parser.parse_args()
    run_pipeline(force_rebuild=args.force_rebuild)