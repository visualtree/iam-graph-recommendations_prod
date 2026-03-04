# Production Readiness Checklist

Last updated: 2026-02-24

## Scope
This checklist tracks functional and operational readiness for the IAM recommendation pipeline (ETL -> Train -> API -> Streamlit).

Status legend:
- DONE: implemented and validated
- PARTIAL: implemented but needs stronger controls/evidence
- TODO: not implemented yet

## 1) Functional Readiness

1. End-to-end training pipeline: DONE
- Evidence: `ml_pipeline/train.py`
- Latest run metrics captured in:
  - `artifacts/leakage_report.json`
  - `artifacts/metrics_holdout_summary.csv`
  - `artifacts/coverage_summary.json`

2. Inference pipeline stability (API + core): DONE
- Evidence:
  - `api/routes/predictions.py`
  - `ml_pipeline/prediction_core.py`
- Verified API/core parity (same top-K for identical input).

3. Evaluation metrics (AUC, Precision@K, Recall@K, Coverage): DONE
- Scripts:
  - `ml_pipeline/evaluate_holdout.py`
  - `ml_pipeline/evaluate_coverage.py`
- Outputs written to `artifacts/`.

4. Streamlit demo functional integration: DONE
- API-first integration and fallback behavior:
  - `streamlit_modules/prediction_engine.py`
  - `streamlit_modules/results_display.py`

5. Data quality / leakage checks in training: DONE
- Evidence: leakage diagnostics emitted into `artifacts/leakage_report.json`.

## 2) Performance & Capacity

1. Single-user latency benchmarking: DONE
2. Scoped concurrency benchmarking (c=4, c=8): DONE
- Latest observed (scoped):
  - c=4: p95 < 1s
  - c=8: p95 ~1.7s

3. Unscoped/high-cardinality tail latency handling: PARTIAL
- Need formal SLO/SLA thresholds with pass/fail gates.

## 3) Operational Readiness

1. Structured logging and request tracing: DONE
- Request ID + duration headers/logging present.

2. Startup readiness / artifact preload: DONE
- Lifespan preload and health endpoint implemented.

3. Monitoring dashboards + alerts: TODO
- Need centralized metrics and alerting (error rate, p95/p99, queue wait, compute).

4. Runbooks / incident procedures: TODO
- Need on-call playbook (API degraded, artifact load failure, model rollback).

5. CI/CD quality gates: PARTIAL
- Git workflow in place.
- Need automated pipeline: lint/tests/eval thresholds before deploy.

## 4) Security & Compliance

1. API authentication/authorization: DONE
- Guard implemented in `api/dependencies.py` and enforced across `/predict*` routes.
- Uses `IAM_API_TOKEN` via `X-API-Key` or `Authorization: Bearer <token>`.
2. Secret management hardening (vault/secure injection): PARTIAL
3. TLS and ingress hardening: TODO
4. Audit logging policy: TODO

## 5) Model Governance

1. Versioned artifact manifest tied to commit SHA: PARTIAL
2. Stage -> prod promotion workflow: TODO
3. Drift monitoring policy and rollback criteria: TODO

## Immediate Next Actions (Priority Order)

1. Add deploy-time env validation for required secrets/config: DONE
- Runtime validation added in `api/app.py` with strict mode support:
  - `IAM_STRICT_ENV_VALIDATION`
  - `IAM_REQUIRE_API_TOKEN`
  - validates `IAM_API_WORKERS`, `API_PORT`, `LOG_LEVEL`, token presence policy
2. Add minimal SLO checks script (p95/p99/error-rate) as release gate: DONE
- Script added: `ml_pipeline/check_slo.py`
- Exits non-zero on threshold breach (p95/p99/error-rate).
3. Add release manifest (`model_version`, `git_sha`, `data_timestamp`) in artifacts.
4. Add CI command workflow for train/eval thresholds and API smoke tests.

## Release Gate Proposal

A release is "Production Ready" when all are true:
- Functional: AUC + top-K + coverage files generated and within accepted bounds.
- Performance: p95/p99 under defined SLO at target concurrency.
- Security: API auth enabled and secrets not hardcoded.
- Operations: monitoring + alerting + runbook available.
- Governance: model/artifact version manifest + rollback path documented.
