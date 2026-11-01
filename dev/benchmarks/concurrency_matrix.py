#!/usr/bin/env python3
"""Measure how a running Splash server scales with concurrent chat requests.

The engine's decode batcher is four lanes wide (SPLASH_MAXIMUM_BATCH_WIDTH),
while the server admits up to --queue-size requests.  This script sends bursts
of 1, 2, 4, 8, ... requests at an already running server and records, per
burst, the aggregate output rate, each request's time to first token and decode
rate, and the Mac's fans and hot sensors from /status.

It measures the *current* system only: it never starts a server, so run it
against a server you started yourself.

    ./splash serve --model ... --port 8123
    .venv/bin/python -m dev.benchmarks.concurrency_matrix \
        --base-url http://127.0.0.1:8123 --model Qwen3.8-27B-4bit \
        --requests 1,2,4,8 --samples 2 --max-tokens 64 \
        --output build/release/concurrency-matrix.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

# A short seed prompt; each request appends its own nonce so a burst does not
# share a cached prefix, and the prompt stays small beside the generated text.
SEED = (
    "Write a concise, factual paragraph about why unified memory helps a local "
    "inference engine. Do not repeat the question. "
)


def status(client: httpx.Client) -> dict:
    response = client.get("/status")
    response.raise_for_status()
    return response.json()


def sample_status(client: httpx.Client) -> dict:
    """The fields a burst cares about, flattened for the peak sampler."""
    value = status(client)
    fans = value.get("fans") or {}
    return {
        "speeds_rpm": list(fans.get("speeds_rpm") or []),
        "hottest_celsius": fans.get("hottest_celsius"),
        "thermal_state": value.get("thermal_state"),
        "duty_percent": (value.get("throttle") or {}).get("duty_percent"),
        "paced_waits": (value.get("throttle") or {}).get("paced_waits"),
    }


class PeakSampler:
    """Poll /status while a burst runs and keep the loudest readings."""

    def __init__(self, base_url: str, interval: float = 0.25) -> None:
        self.base_url = base_url
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.max_rpm = 0
        self.max_celsius = 0.0
        self.states: set[str] = set()
        self.duties: set[float] = set()
        self.error: str | None = None

    def _run(self) -> None:
        try:
            with httpx.Client(base_url=self.base_url, timeout=2.0) as client:
                while not self.stop.wait(self.interval):
                    value = sample_status(client)
                    self.max_rpm = max(self.max_rpm, *(value["speeds_rpm"] or [0]))
                    if value["hottest_celsius"] is not None:
                        self.max_celsius = max(
                            self.max_celsius, value["hottest_celsius"]
                        )
                    if value["thermal_state"]:
                        self.states.add(value["thermal_state"])
                    if value["duty_percent"] is not None:
                        self.duties.add(value["duty_percent"])
        except Exception as error:  # a closed server should not fail the burst
            self.error = str(error)

    def __enter__(self) -> "PeakSampler":
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)


def request_once(
    base_url: str, model: str, prompt: str, max_tokens: int, barrier: threading.Barrier
) -> dict:
    """One streaming chat request; returns its timings and token counts."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token: float | None = None
    tokens = 0
    usage = None
    with httpx.Client(base_url=base_url, timeout=None) as client:
        barrier.wait()
        with client.stream("POST", "/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if not (delta.get("content") or delta.get("reasoning_content")):
                        continue
                    if first_token is None:
                        first_token = time.perf_counter()
                    tokens += 1
    finished = time.perf_counter()
    if usage:
        tokens = int(usage.get("completion_tokens", tokens))
    return {
        "started": started,
        "first_token": first_token,
        "finished": finished,
        "tokens": tokens,
        "ttft_ms": None if first_token is None else (first_token - started) * 1000,
        "decode_ms": None if first_token is None else (finished - first_token) * 1000,
    }


def run_burst(
    base_url: str, model: str, width: int, prompt: str, max_tokens: int
) -> dict:
    barrier = threading.Barrier(width)
    with PeakSampler(base_url) as sampler:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=width) as pool:
            results = list(
                pool.map(
                    lambda index: request_once(
                        base_url,
                        model,
                        f"{prompt}\n\nRequest {index}.",
                        max_tokens,
                        barrier,
                    ),
                    range(width),
                )
            )
        finished = time.perf_counter()
    wall = finished - started
    tokens = sum(result["tokens"] for result in results)
    ttfts = [result["ttft_ms"] for result in results if result["ttft_ms"] is not None]
    decode_rates = [
        result["tokens"] * 1000.0 / result["decode_ms"]
        for result in results
        if result["decode_ms"] and result["tokens"]
    ]
    return {
        "width": width,
        "wall_ms": wall * 1000.0,
        "output_tokens": tokens,
        "aggregate_tokens_per_second": tokens / wall if wall else 0.0,
        "ttft_ms": {
            "min": min(ttfts) if ttfts else None,
            "median": statistics.median(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        "decode_tokens_per_second": {
            "min": min(decode_rates) if decode_rates else None,
            "median": statistics.median(decode_rates) if decode_rates else None,
            "max": max(decode_rates) if decode_rates else None,
        },
        "fan_rpm_max": sampler.max_rpm,
        "hottest_celsius_max": sampler.max_celsius or None,
        "thermal_states": sorted(sampler.states),
        "duty_percent": sorted(sampler.duties),
        "status_error": sampler.error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--requests",
        default="1,2,4,8",
        help="comma-separated burst widths (default: 1,2,4,8)",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-chars",
        type=int,
        default=0,
        help="repeat the seed prompt to about this many characters (0: the seed alone)",
    )
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    widths = [int(value) for value in args.requests.split(",") if value]
    if not widths or any(width < 1 for width in widths):
        parser.error("--requests needs positive comma-separated widths")

    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        before = sample_status(client)
        models = client.get("/v1/models")
        models.raise_for_status()
        served = {item["id"] for item in models.json()["data"]}

    if args.model not in served:
        parser.error(f"{args.model!r} is not served; /v1/models lists {sorted(served)}")

    results = []
    for sample in range(args.samples):
        for width in widths:
            seed = args.seed
            if args.prompt_chars > 0:
                seed = seed * (args.prompt_chars // len(seed) + 1)
            prompt = f"{seed} Sample {sample}."
            result = run_burst(
                args.base_url, args.model, width, prompt, args.max_tokens
            )
            result["sample"] = sample
            results.append(result)
            print(
                f"sample {sample} width {width}: "
                f"{result['aggregate_tokens_per_second']:.1f} tok/s aggregate, "
                f"ttft {result['ttft_ms']['median']:.0f} ms, "
                f"fans {result['fan_rpm_max']} rpm, "
                f"{result['hottest_celsius_max']} °C",
                flush=True,
            )

    document = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "prompt_chars": args.prompt_chars,
        "before": before,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        print(f"wrote {args.output}")
    else:
        print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
