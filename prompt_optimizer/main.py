"""
Prompt Optimization Agent - FastAPI Application.

Serves the GUI and provides the optimization API endpoints.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .techniques import get_all_techniques, get_techniques_by_ids
from .engine import (
    build_optimization_system_prompt,
    build_optimization_user_prompt,
    build_analysis_system_prompt,
    build_analysis_user_prompt,
)
from .api_clients import create_client, get_available_models

app = FastAPI(
    title="Prompt Optimization Agent",
    description="AI-powered prompt engineering tool with PE technique selection",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).parent / "static"


# ── Request / Response Models ─────────────────────────────────────

class OptimizeRequest(BaseModel):
    original_prompt: str
    technique_ids: list[str]
    provider: str  # "openai" or "anthropic"
    model: str
    api_key: str
    language: str = "en"  # "en" or "ja"
    temperature: float = 0.7
    max_tokens: int = 4096
    include_analysis: bool = True


class OptimizeResponse(BaseModel):
    optimized_prompt: str
    analysis: str | None = None
    model_used: str
    techniques_applied: list[str]
    usage: dict


# ── API Endpoints ─────────────────────────────────────────────────

@app.get("/api/techniques")
def list_techniques():
    """Return all available PE techniques."""
    return {"techniques": get_all_techniques()}


@app.get("/api/models")
def list_models():
    """Return available models grouped by provider."""
    return {"models": get_available_models()}


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize_prompt(req: OptimizeRequest):
    """Optimize a prompt using selected PE techniques."""

    # Validate
    if not req.original_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if not req.technique_ids:
        raise HTTPException(status_code=400, detail="Select at least one technique.")
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required.")

    techniques = get_techniques_by_ids(req.technique_ids)
    if not techniques:
        raise HTTPException(status_code=400, detail="No valid techniques selected.")

    # Build prompts
    system_prompt = build_optimization_system_prompt(techniques, language=req.language)
    user_prompt = build_optimization_user_prompt(req.original_prompt)

    # Call LLM
    try:
        client = create_client(req.provider, req.model, req.api_key)
        response = client.chat(
            user_message=user_prompt,
            system_prompt=system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    technique_names = [t.name for t in techniques]
    optimized = response.content
    total_usage = response.usage.copy()

    # Optional analysis
    analysis = None
    if req.include_analysis:
        try:
            analysis_system = build_analysis_system_prompt()
            analysis_user = build_analysis_user_prompt(
                req.original_prompt, optimized, technique_names
            )
            analysis_response = client.chat(
                user_message=analysis_user,
                system_prompt=analysis_system,
                temperature=0.5,
                max_tokens=2048,
            )
            analysis = analysis_response.content
            # Merge usage
            for key in total_usage:
                total_usage[key] += analysis_response.usage.get(key, 0)
        except Exception:
            analysis = None  # Analysis is optional; don't fail the whole request

    return OptimizeResponse(
        optimized_prompt=optimized,
        analysis=analysis,
        model_used=response.model,
        techniques_applied=technique_names,
        usage=total_usage,
    )


# ── Static File Serving ──────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the main GUI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── CLI Entry Point ──────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8501):
    """Run the application server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
