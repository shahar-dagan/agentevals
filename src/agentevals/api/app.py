"""FastAPI application for agentevals REST API."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

# Load environment variables from .env file if it exists
# This makes GOOGLE_API_KEY available for LLM-based evaluators
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

app = FastAPI(
    title="agentevals API",
    version="0.1.0",
    description="REST API for evaluating agent traces using ADK's scoring framework",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
