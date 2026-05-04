#!/usr/bin/env python3
"""CLUED-IN: A Chat Agent That Actually Learns Across Sessions.

FastAPI backend + custom HTML/CSS/JS frontend matching the Accrue design
language (serif typography, forest green palette, geometric aesthetic).

Launch:
    cd /workspace/clue
    python -m clued_in.app
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clued_in.data_formatter import DataFormatter
from clued_in.engine import CLUEEngine
from clued_in.model_server import ModelServer
from utils import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ICONS_DIR = Path(__file__).parent.parent / "icons"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: int
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    learned: bool = False


def _save_session(session: SessionState, sessions_dir: str):
    path = Path(sessions_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"session_{session.session_id:04d}.json"
    filepath.write_text(json.dumps(asdict(session), indent=2))


def _load_sessions(sessions_dir: str) -> list[SessionState]:
    path = Path(sessions_dir)
    sessions = []
    if path.exists():
        for f in sorted(path.glob("session_*.json")):
            data = json.loads(f.read_text())
            sessions.append(SessionState(**data))
    return sessions


def _summarize(messages: list[dict]) -> str:
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    if not user_msgs:
        return "(empty)"
    first = user_msgs[0][:70]
    return f"{first}{'...' if len(user_msgs[0]) > 70 else ''}"


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_messages: list[dict] = []


class CompareRequest(BaseModel):
    prompt: str


class TeachRequest(BaseModel):
    fact: str


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_app(config: dict | None = None) -> FastAPI:
    if config is None:
        config = load_config()

    use_synthetic = config.get("_cli", {}).get("synthetic", False)
    logger.info(f"Config: epochs={config['training']['epochs']}, lr={config['training']['lr']}, synthetic={use_synthetic}")

    logger.info("Loading model...")
    model_server = ModelServer(config)
    logger.info("Initializing CLUE engine...")
    engine = CLUEEngine(model_server, config)
    formatter = DataFormatter(model_server.tokenizer, config["ui"]["system_prompt"])
    # Only wire synthetic generation if --synthetic flag is set
    if use_synthetic:
        formatter.set_generate_fn(lambda msgs, max_new_tokens=1024: model_server.generate(
            msgs, max_new_tokens=max_new_tokens, temperature=0.01
        ))
        logger.info("Synthetic training pair generation ENABLED")
    else:
        logger.info("Synthetic training pair generation DISABLED (use --synthetic to enable)")

    completed_sessions = _load_sessions(config["paths"]["sessions_dir"])
    app_state = {
        "completed_sessions": completed_sessions,
        "next_session_id": len(completed_sessions),
    }

    app = FastAPI(title="CLUED IN")

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/icons", StaticFiles(directory=str(ICONS_DIR)), name="icons")

    # -------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = Path(__file__).parent / "index.html"
        html = html_path.read_text()
        return HTMLResponse(content=html, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
        })

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        if engine.is_training:
            status = engine.get_status()
            return {
                "response": f"*Learning in progress ({status.get('progress', '...')}). Please wait.*",
                "busy": True,
            }
        messages = req.session_messages or []
        messages.append({"role": "user", "content": req.message})
        sys_msg = {"role": "system", "content": config["ui"]["system_prompt"]}
        full_messages = [sys_msg] + messages
        try:
            response = model_server.generate(full_messages)
        except Exception as e:
            response = f"Error: {e}"
        messages.append({"role": "assistant", "content": response})
        return {"response": response, "session_messages": messages, "busy": False}

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest):
        if engine.is_training:
            status = engine.get_status()
            async def busy_stream():
                msg = f"*Learning in progress ({status.get('progress', '...')}). Please wait.*"
                yield f"data: {json.dumps({'token': msg})}\n\n"
                yield f"data: {json.dumps({'done': True, 'session_messages': req.session_messages})}\n\n"
            return StreamingResponse(busy_stream(), media_type="text/event-stream")
        messages = list(req.session_messages or [])
        messages.append({"role": "user", "content": req.message})
        sys_msg = {"role": "system", "content": config["ui"]["system_prompt"]}
        full_messages = [sys_msg] + messages
        def token_stream():
            full_response = []
            try:
                for chunk in model_server.generate_stream(full_messages):
                    full_response.append(chunk)
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'token': f'Error: {e}'})}\n\n"
            response_text = "".join(full_response).strip()
            messages.append({"role": "assistant", "content": response_text})
            yield f"data: {json.dumps({'done': True, 'session_messages': messages})}\n\n"
        return StreamingResponse(token_stream(), media_type="text/event-stream")

    @app.post("/api/end-session")
    async def end_session(req: Request):
        body = await req.json()
        session_messages = body.get("session_messages", [])
        if not session_messages:
            return {"ok": False, "reason": "empty session"}
        session_id = app_state["next_session_id"]
        app_state["next_session_id"] = session_id + 1
        session = SessionState(
            session_id=session_id,
            messages=session_messages,
            summary=_summarize(session_messages),
        )
        app_state["completed_sessions"].append(session)
        _save_session(session, config["paths"]["sessions_dir"])
        def _learn():
            try:
                engine._status = {"state": "training", "session": session_id, "progress": "Generating training data..."}
                examples = formatter.format_session(session_messages)
                logger.info(f"Session {session_id}: {len(examples)} training examples")
                for i, ex in enumerate(examples):
                    logger.info(f"  Example {i}: {ex['label_text'][:80]}...")
                engine.learn_session(session_id, examples)
                session.learned = True
                _save_session(session, config["paths"]["sessions_dir"])
            except Exception as e:
                import traceback
                logger.error(f"Learning failed: {e}\n{traceback.format_exc()}")
        threading.Thread(target=_learn, daemon=True).start()
        return {"ok": True, "session_id": session_id}

    @app.post("/api/compare")
    async def compare(req: CompareRequest):
        messages = [
            {"role": "system", "content": config["ui"]["system_prompt"]},
            {"role": "user", "content": req.prompt},
        ]
        try:
            learned = model_server.generate(messages, use_adapter=True)
        except Exception as e:
            learned = f"Error: {e}"
        try:
            base = model_server.generate(messages, use_adapter=False)
        except Exception as e:
            base = f"Error: {e}"
        return {"learned": learned, "base": base}

    @app.post("/api/teach")
    async def teach(req: TeachRequest):
        examples = formatter.format_quick_teach(req.fact)
        session_id = app_state["next_session_id"]
        app_state["next_session_id"] = session_id + 1
        session = SessionState(
            session_id=session_id,
            messages=[
                {"role": "user", "content": f"Remember: {req.fact}"},
                {"role": "assistant", "content": "I'll remember that."},
            ],
            summary=f"Taught: {req.fact[:60]}",
        )
        app_state["completed_sessions"].append(session)
        _save_session(session, config["paths"]["sessions_dir"])
        def _learn():
            try:
                engine.learn_session(session_id, examples)
                session.learned = True
                _save_session(session, config["paths"]["sessions_dir"])
            except Exception as e:
                logger.error(f"Quick teach failed: {e}")
        threading.Thread(target=_learn, daemon=True).start()
        return {"ok": True, "session_id": session_id}

    @app.get("/api/session/{session_id}")
    async def get_session(session_id: int):
        for s in app_state["completed_sessions"]:
            if s.session_id == session_id:
                return {"messages": s.messages, "summary": s.summary, "learned": s.learned}
        return {"messages": [], "summary": "", "learned": False}

    @app.post("/api/learn")
    async def learn_current(req: Request):
        body = await req.json()
        session_messages = body.get("session_messages", [])
        if not session_messages:
            return {"ok": False, "reason": "empty session"}
        if engine.is_training:
            return {"ok": False, "reason": "already learning"}
        session_id = app_state["next_session_id"]
        app_state["next_session_id"] = session_id + 1
        session = SessionState(
            session_id=session_id,
            messages=session_messages,
            summary=_summarize(session_messages),
        )
        app_state["completed_sessions"].append(session)
        _save_session(session, config["paths"]["sessions_dir"])
        def _learn():
            try:
                engine._status = {"state": "training", "session": session_id, "progress": "Generating training data..."}
                examples = formatter.format_session(session_messages)
                logger.info(f"Learn (mid-session) {session_id}: {len(examples)} examples")
                for i, ex in enumerate(examples):
                    logger.info(f"  Example {i}: {ex['label_text'][:80]}...")
                engine.learn_session(session_id, examples)
                session.learned = True
                _save_session(session, config["paths"]["sessions_dir"])
            except Exception as e:
                import traceback
                logger.error(f"Learning failed: {e}\n{traceback.format_exc()}")
        threading.Thread(target=_learn, daemon=True).start()
        return {"ok": True, "session_id": session_id}

    @app.get("/api/status")
    async def status():
        st = engine.get_status()
        sessions = [
            {"id": s.session_id, "summary": s.summary, "learned": s.learned}
            for s in app_state["completed_sessions"]
        ]
        return {
            "state": st["state"],
            "progress": st.get("progress", ""),
            "session": st.get("session"),
            "sessions_learned": engine.task_idx,
            "sessions": sessions,
        }

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLUED-IN chat agent with CL")
    parser.add_argument("--synthetic", action="store_true",
                        help="Enable self-annotation (generates synthetic training pairs). Slower but may improve learning.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs (default: from config.yaml)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config learning rate")
    parser.add_argument("--port", type=int, default=None,
                        help="Override server port")
    args = parser.parse_args()

    config = load_config()
    config["_cli"] = {"synthetic": args.synthetic}
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.port is not None:
        config["ui"]["server_port"] = args.port

    app = create_app(config)
    uvicorn.run(
        app,
        host=config["ui"]["server_name"],
        port=config["ui"]["server_port"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
