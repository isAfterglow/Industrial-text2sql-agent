"""Small repeatable API task load test (stdlib only).

Usage: python tools/load_test_tasks.py --base-url http://127.0.0.1:8000 --token TOKEN --concurrency 5 --count 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, method: str = "GET", token: str = "", body: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=payload, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def run_one(base_url: str, token: str, question: str) -> dict:
    started = time.perf_counter()
    try:
        task = request_json(base_url + "/api/tasks", "POST", token, {"question": question, "profile": "resin"})
        task_id = task["task_id"]
        while True:
            current = request_json(base_url + f"/api/tasks/{task_id}", token=token)
            if current.get("status") in {"completed", "approval_required", "failed", "cancelled"}:
                break
            time.sleep(0.25)
        return {"status": current.get("status"), "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    except (HTTPError, URLError, TimeoutError, KeyError) as exc:
        return {"status": "worker_error", "error": f"{type(exc).__name__}: {exc}", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    questions = ["查询原始密度最高的3个样本。"] * max(1, args.count)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        results = list(executor.map(lambda q: run_one(args.base_url, args.token, q), questions))
    values = [float(item["elapsed_ms"]) for item in results]
    summary = {
        "count": len(results), "concurrency": args.concurrency,
        "status_counts": {status: sum(1 for item in results if item.get("status") == status) for status in sorted({item.get("status") for item in results})},
        "mean_ms": round(statistics.mean(values), 2) if values else 0,
        "p95_ms": round(sorted(values)[max(0, int(len(values) * .95) - 1)], 2) if values else 0,
        "wall_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status_counts"].get("worker_error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

