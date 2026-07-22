# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()

logger = init_logger(__name__)


def base(request: Request) -> ServingTokenization:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> ServingTokenization:
    return request.app.state.serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/responses
    # - /v1/responses/{response_id}
    # - /v1/responses/{response_id}/cancel
    # - /v1/messages
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/audio/translations
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(content={"server_load": request.app.state.server_load_metrics})


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


def _get_output_processor(request: Request):
    """Best-effort access to the AsyncLLM OutputProcessor.

    Returns None when the engine client has no output_processor attribute
    (e.g. non-v1 engine clients or test stubs), so /get_load degrades to an
    empty load entry instead of 500ing.
    """
    engine = getattr(request.app.state, "engine_client", None)
    return getattr(engine, "output_processor", None)


def _build_load_entry(output_processor) -> dict[str, Any]:
    # Aggregated counts come from the most recent SchedulerStats cached by
    # the OutputProcessor (updated every engine step). May be None before the
    # first step completes.
    stats = getattr(output_processor, "_last_scheduler_stats", None)
    num_running = getattr(stats, "num_running_reqs", 0) or 0
    num_waiting = getattr(stats, "num_waiting_reqs", 0) or 0
    kv_cache_usage = getattr(stats, "kv_cache_usage", 0.0) or 0.0

    request_states = getattr(output_processor, "request_states", {})

    per_request = []
    num_waiting_uncached_tokens = 0
    for req_state in request_states.values():
        prompt_len = getattr(req_state, "prompt_len", 0) or 0
        cached_tokens = getattr(req_state, "num_cached_tokens", 0) or 0
        is_prefilling = getattr(req_state, "is_prefilling", False)
        # Approximation of "waiting, not-yet-cached prompt tokens": for a
        # request still in prefill, the prompt tokens not covered by a prefix
        # cache hit still need to be computed. vLLM has no exact aggregate for
        # this on the frontend; the router only needs a non-None signal here.
        if is_prefilling:
            num_waiting_uncached_tokens += max(prompt_len - cached_tokens, 0)

        req_stats = getattr(req_state, "stats", None)
        generated_tokens = (
            getattr(req_stats, "num_generation_tokens", 0) if req_stats else 0
        )
        per_request.append(
            {
                "request_id": getattr(req_state, "external_req_id", None),
                "prompt_tokens": prompt_len,
                "generated_tokens": generated_tokens,
                # num_cached_tokens is populated only after prefill completes
                # (from prefill_stats); it is 0 while a request is still
                # prefilling.
                "cached_tokens": cached_tokens,
                "is_prefilling": is_prefilling,
                "arrival_time": (
                    getattr(req_stats, "arrival_time", 0.0) if req_stats else 0.0
                ),
                "max_tokens": getattr(req_state, "max_tokens_param", None),
            }
        )

    return {
        # Single entry: vLLM v1 frontend request_states / SchedulerStats are a
        # merged view across DP ranks, not split per-rank.
        "dp_rank": 0,
        # "null" -> router normalizes to Integrated (prefill+decode capable).
        "load_role": "null",
        "num_reqs": num_running + num_waiting,
        "num_running_reqs": num_running,
        "num_waiting_reqs": num_waiting,
        "num_waiting_uncached_tokens": num_waiting_uncached_tokens,
        # vLLM frontend has no aggregate total/used token counter; router
        # ignores both fields.
        "num_tokens": 0,
        "num_pending_tokens": 0,
        "ts_tic": time.perf_counter(),
        # Diagnostic extras (router ignores unknown fields).
        "kv_cache_usage": kv_cache_usage,
        "per_request": per_request,
    }


@router.get("/get_load")
async def get_load(request: Request):
    """Per-worker load snapshot polled by the SGLang router (`load_poller`).

    Returns a JSON array (one entry per DP rank; vLLM reports a single merged
    entry) whose shape mirrors the SGLang worker's /get_load so the router's
    `GetLoadEntry` deserializer (all fields default, unknown fields ignored)
    consumes it. Router total pressure = sum(num_reqs + num_waiting_reqs).
    A worker is prefill-capable only if num_running_reqs and
    num_waiting_uncached_tokens are both present.
    """
    output_processor = _get_output_processor(request)
    entry = _build_load_entry(output_processor) if output_processor else {
        "dp_rank": 0,
        "load_role": "null",
        "num_reqs": 0,
        "num_running_reqs": 0,
        "num_waiting_reqs": 0,
        "num_waiting_uncached_tokens": 0,
        "num_tokens": 0,
        "num_pending_tokens": 0,
        "ts_tic": time.perf_counter(),
        "kv_cache_usage": 0.0,
        "per_request": [],
    }
    return JSONResponse(content=[entry])
