from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RequestResult:
    ok: bool
    status: int
    latency_ms: float
    error: str | None = None


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _load_users(users_file: Path) -> list[int]:
    raw = json.loads(users_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "all" in raw:
        vals = raw["all"]
    elif isinstance(raw, list):
        vals = raw
    else:
        raise ValueError(f"Unsupported users file shape: {users_file}")
    users = [int(x) for x in vals if int(x) != 1]
    if not users:
        raise ValueError("No valid users found for SLO check.")
    return users


def _call_predict(
    url: str,
    user_id: int,
    top_n: int,
    initial_candidates: int,
    endpoint_id: int | None,
    timeout_s: float,
    api_token: str | None,
) -> RequestResult:
    payload: dict[str, int] = {
        "user_id": user_id,
        "top_n": top_n,
        "initial_candidates": initial_candidates,
    }
    if endpoint_id is not None:
        payload["endpoint_id"] = endpoint_id

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["X-API-Key"] = api_token

    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(ok=200 <= resp.status < 300, status=resp.status, latency_ms=latency_ms)
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(ok=False, status=exc.code, latency_ms=latency_ms, error=f"HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(ok=False, status=0, latency_ms=latency_ms, error=str(exc))


def run_slo_check(
    url: str,
    users: list[int],
    requests: int,
    concurrency: int,
    top_n: int,
    initial_candidates: int,
    endpoint_id: int | None,
    timeout_s: float,
    seed: int,
    api_token: str | None,
    p95_ms_max: float,
    p99_ms_max: float,
    error_rate_max: float,
) -> int:
    rng = random.Random(seed)
    sampled_users = [rng.choice(users) for _ in range(requests)]

    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _call_predict,
                url=url,
                user_id=uid,
                top_n=top_n,
                initial_candidates=initial_candidates,
                endpoint_id=endpoint_id,
                timeout_s=timeout_s,
                api_token=api_token,
            )
            for uid in sampled_users
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    latencies = sorted(r.latency_ms for r in results)
    ok = sum(1 for r in results if r.ok)
    fail = len(results) - ok
    err_rate = (fail / len(results)) if results else 1.0
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    avg = sum(latencies) / len(latencies) if latencies else 0.0

    print("\nSLO CHECK SUMMARY")
    print(f"url={url}")
    print(f"requests={len(results)} concurrency={concurrency}")
    print(f"ok={ok} fail={fail} error_rate={err_rate:.4f}")
    print(f"avg_ms={avg:.2f} p95_ms={p95:.2f} p99_ms={p99:.2f} max_ms={max(latencies) if latencies else 0:.2f}")
    print(f"thresholds: p95<={p95_ms_max:.2f} p99<={p99_ms_max:.2f} error_rate<={error_rate_max:.4f}")

    pass_p95 = p95 <= p95_ms_max
    pass_p99 = p99 <= p99_ms_max
    pass_err = err_rate <= error_rate_max

    if pass_p95 and pass_p99 and pass_err:
        print("SLO RESULT: PASS")
        return 0

    print("SLO RESULT: FAIL")
    if not pass_p95:
        print(f" - p95 exceeded: {p95:.2f} > {p95_ms_max:.2f}")
    if not pass_p99:
        print(f" - p99 exceeded: {p99:.2f} > {p99_ms_max:.2f}")
    if not pass_err:
        print(f" - error_rate exceeded: {err_rate:.4f} > {error_rate_max:.4f}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser("SLO gate for IAM /predict endpoint")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8010/predict")
    parser.add_argument("--users-file", type=str, default="cohort_users.json")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--endpoint-id", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-token", type=str, default=os.environ.get("IAM_API_TOKEN"))
    parser.add_argument("--p95-ms-max", type=float, default=2000.0)
    parser.add_argument("--p99-ms-max", type=float, default=3000.0)
    parser.add_argument("--error-rate-max", type=float, default=0.01)
    args = parser.parse_args()

    users = _load_users(Path(args.users_file))
    return run_slo_check(
        url=args.url,
        users=users,
        requests=args.requests,
        concurrency=args.concurrency,
        top_n=args.top_n,
        initial_candidates=args.initial_candidates,
        endpoint_id=args.endpoint_id,
        timeout_s=args.timeout_s,
        seed=args.seed,
        api_token=args.api_token,
        p95_ms_max=args.p95_ms_max,
        p99_ms_max=args.p99_ms_max,
        error_rate_max=args.error_rate_max,
    )


if __name__ == "__main__":
    raise SystemExit(main())
