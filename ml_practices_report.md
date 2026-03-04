# ML Concepts, Best Practices, and Gaps Report

## Summary
This report inventories ML concepts and best practices implemented in this codebase, identifies gaps vs. common ML best practices, and maps each implemented practice to concrete code locations.

## Implemented ML Concepts
1. Two-stage recommender (candidate generation + reranking).
   Evidence: ml_pipeline/train.py, ml_pipeline/prediction_core.py, streamlit_modules/prediction_engine.py
2. Graph embeddings (Node2Vec via Neo4j GDS).
   Evidence: ml_pipeline/data_loader.py
3. Embedding similarity features (cosine similarity, Euclidean distance).
   Evidence: ml_pipeline/feature_engineering.py
4. Peer adoption features (close peers, direct team, role peers, department peers).
   Evidence: ml_pipeline/feature_engineering.py, streamlit_modules/results_display.py
5. Learning-to-rank via probabilistic binary classifiers (XGBoost) and ranking by score.
   Evidence: ml_pipeline/train.py, ml_pipeline/prediction_core.py
6. Monotonic constraints to encode domain knowledge (peer adoption rates).
   Evidence: ml_pipeline/train.py, artifacts/leakage_report.json
7. Model explainability with SHAP for per-recommendation feature attribution.
   Evidence: ml_pipeline/prediction_core.py, streamlit_modules/explainability.py, api/routes/predictions.py

## Implemented ML Best Practices
1. Stratified train/validation split.
   Evidence: ml_pipeline/train.py
2. Hyperparameter optimization (Optuna) for both models.
   Evidence: ml_pipeline/train.py
3. Early stopping during model training.
   Evidence: ml_pipeline/train.py
4. Leakage guardrail via user-disjoint holdout AUC diagnostic.
   Evidence: ml_pipeline/train.py, artifacts/leakage_report.json
5. Proper holdout evaluation for new-access recommendation (Precision@K/Recall@K).
   Evidence: ml_pipeline/evaluate_holdout.py
6. Coverage/diversity evaluation (catalog coverage, Gini coefficient).
   Evidence: ml_pipeline/evaluate_coverage.py
7. Hard fail feature alignment between training and inference.
   Evidence: ml_pipeline/prediction_core.py
8. Feature schema sanity checks (numeric coercion, NaN fill, duplicate/blank checks).
   Evidence: ml_pipeline/feature_engineering.py
9. Data type standardization after loading from Neo4j.
   Evidence: ml_pipeline/data_loader.py
10. Embedding deduplication to prevent duplicated rows.
    Evidence: ml_pipeline/data_loader.py, ml_pipeline/feature_engineering.py
11. Artifact caching (singleton loader) for consistent inference and reduced I/O.
    Evidence: ml_pipeline/prediction_core.py
12. SLO/latency checks for API performance (p95/p99/error rate).
    Evidence: ml_pipeline/check_slo.py

## Gaps vs. Best Practices
1. No dedicated locked test set for final reporting (only random split + user-holdout + holdout sim).
   Evidence: ml_pipeline/train.py, ml_pipeline/evaluate_holdout.py
2. No probability calibration (Platt/Isotonic) despite score use.
   Evidence: integration_debug_script.py (mentions only)
3. No explicit class-imbalance handling (e.g., scale_pos_weight).
   Evidence: ml_pipeline/train.py
4. No systematic feature selection or explicit regularization strategy beyond XGBoost defaults.
   Evidence: ml_pipeline/train.py
5. No data drift monitoring / retraining policy implemented in code.
   Evidence: PRODUCTION_READINESS.md (TODO)
6. No offline/online distribution checks beyond feature alignment.
   Evidence: ml_pipeline/prediction_core.py
7. No automated CI gate that blocks deploy on metric regressions.
   Evidence: PRODUCTION_READINESS.md (TODO)
8. No fairness/bias evaluation.
   Evidence: not present in ml_pipeline/ or streamlit_modules/
9. No experiment tracking (MLflow/W&B) or dataset lineage tracking.
   Evidence: not present in repo
10. Optuna sampler seed not explicitly set for reproducibility.
    Evidence: ml_pipeline/train.py
11. Monitoring focuses on latency/error only; no real-world prediction quality monitoring.
    Evidence: ml_pipeline/check_slo.py

## Mapping: Practice ? Code Locations
1. Two-stage recommender
   Files: ml_pipeline/train.py, ml_pipeline/prediction_core.py, streamlit_modules/prediction_engine.py
2. Graph embeddings (Node2Vec)
   Files: ml_pipeline/data_loader.py
3. Embedding similarity features
   Files: ml_pipeline/feature_engineering.py
4. Peer adoption features
   Files: ml_pipeline/feature_engineering.py, streamlit_modules/results_display.py
5. Optuna hyperparameter tuning
   Files: ml_pipeline/train.py
6. Early stopping
   Files: ml_pipeline/train.py
7. Monotonic constraints
   Files: ml_pipeline/train.py, artifacts/leakage_report.json
8. Leakage guardrail (user-disjoint AUC)
   Files: ml_pipeline/train.py, artifacts/leakage_report.json
9. Holdout evaluation (Precision@K/Recall@K)
   Files: ml_pipeline/evaluate_holdout.py
10. Coverage/diversity evaluation
    Files: ml_pipeline/evaluate_coverage.py
11. Strict feature alignment at inference
    Files: ml_pipeline/prediction_core.py
12. Data type standardization
    Files: ml_pipeline/data_loader.py
13. Embedding deduplication
    Files: ml_pipeline/data_loader.py, ml_pipeline/feature_engineering.py
14. Artifact caching
    Files: ml_pipeline/prediction_core.py
15. SHAP explainability
    Files: ml_pipeline/prediction_core.py, streamlit_modules/explainability.py, api/routes/predictions.py
16. SLO/latency checks
    Files: ml_pipeline/check_slo.py
