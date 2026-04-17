#!/usr/bin/env python3
"""CLUED-IN: A Chat Agent That Actually Learns Across Sessions.

Gradio application with:
- Chat interface with Send / End Session controls
- Session history sidebar showing completed sessions
- A/B comparison panel (learned model vs base model)
- Learning status indicator
- Quick Teach input for one-shot facts

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

# Ensure clue/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clued_in.data_formatter import DataFormatter
from clued_in.engine import CLUEEngine
from clued_in.model_server import ModelServer
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: int
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    learned: bool = False


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_session(session: SessionState, sessions_dir: str):
    """Save a completed session to disk."""
    path = Path(sessions_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"session_{session.session_id:04d}.json"
    filepath.write_text(json.dumps(asdict(session), indent=2))


def _load_sessions(sessions_dir: str) -> list[SessionState]:
    """Load all saved sessions from disk."""
    path = Path(sessions_dir)
    sessions = []
    if path.exists():
        for f in sorted(path.glob("session_*.json")):
            data = json.loads(f.read_text())
            sessions.append(SessionState(**data))
    return sessions


def _summarize_session(messages: list[dict]) -> str:
    """Generate a one-line summary of a session from its messages."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    if not user_msgs:
        return "(empty session)"
    first = user_msgs[0][:80]
    return f"{first}{'...' if len(user_msgs[0]) > 80 else ''} ({len(user_msgs)} turns)"


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_app():
    """Build and return the Gradio Blocks app."""
    import gradio as gr

    config = load_config()

    # Initialize components
    logger.info("Loading model...")
    model_server = ModelServer(config)
    logger.info("Initializing CLUE engine...")
    engine = CLUEEngine(model_server, config)
    formatter = DataFormatter(model_server.tokenizer, config["ui"]["system_prompt"])

    # Load completed sessions from disk
    completed_sessions = _load_sessions(config["paths"]["sessions_dir"])
    next_session_id = len(completed_sessions)

    # Shared mutable state (protected by GIL for simple ops)
    app_state = {
        "completed_sessions": completed_sessions,
        "next_session_id": next_session_id,
    }

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def send_message(user_msg, chat_history, session_messages):
        """Handle user sending a message."""
        if not user_msg or not user_msg.strip():
            return "", chat_history, session_messages

        if engine.is_training:
            # Model busy — return a status message
            status = engine.get_status()
            progress = status.get("progress", "learning")
            bot_msg = f"*I'm currently learning from our last session ({progress}). Please wait a moment...*"
            chat_history = chat_history + [[user_msg, bot_msg]]
            return "", chat_history, session_messages

        user_msg = user_msg.strip()
        session_messages = session_messages or []
        session_messages.append({"role": "user", "content": user_msg})

        # Build messages with system prompt
        sys_msg = {"role": "system", "content": config["ui"]["system_prompt"]}
        full_messages = [sys_msg] + session_messages

        # Generate response
        try:
            response = model_server.generate(full_messages)
        except Exception as e:
            response = f"*Error generating response: {e}*"

        session_messages.append({"role": "assistant", "content": response})
        chat_history = chat_history + [[user_msg, response]]

        return "", chat_history, session_messages

    def end_session(chat_history, session_messages):
        """End current session, trigger background learning."""
        session_messages = session_messages or []

        if not session_messages:
            return chat_history, session_messages, _format_sidebar(app_state), _format_status(engine)

        session_id = app_state["next_session_id"]
        app_state["next_session_id"] = session_id + 1

        # Create session record
        session = SessionState(
            session_id=session_id,
            messages=session_messages,
            summary=_summarize_session(session_messages),
            learned=False,
        )
        app_state["completed_sessions"].append(session)
        _save_session(session, config["paths"]["sessions_dir"])

        # Format training data
        examples = formatter.format_session(session_messages)
        logger.info(f"Session {session_id}: {len(examples)} training examples from {len(session_messages)} messages")

        # Start background learning
        def _learn():
            try:
                engine.learn_session(session_id, examples)
                session.learned = True
                _save_session(session, config["paths"]["sessions_dir"])
            except Exception as e:
                logger.error(f"Background learning failed: {e}")

        thread = threading.Thread(target=_learn, daemon=True)
        thread.start()

        # Reset chat for new session
        new_history = []
        new_messages = []

        return new_history, new_messages, _format_sidebar(app_state), _format_status(engine)

    def compare_responses(prompt):
        """Generate side-by-side responses with and without adapter."""
        if not prompt or not prompt.strip():
            return "", ""

        messages = [
            {"role": "system", "content": config["ui"]["system_prompt"]},
            {"role": "user", "content": prompt.strip()},
        ]

        try:
            learned = model_server.generate(messages, use_adapter=True)
        except Exception as e:
            learned = f"Error: {e}"

        try:
            base = model_server.generate(messages, use_adapter=False)
        except Exception as e:
            base = f"Error: {e}"

        return learned, base

    def quick_teach(fact, chat_history, session_messages):
        """Teach a single fact via synthetic training pairs."""
        if not fact or not fact.strip():
            return chat_history, session_messages, _format_status(engine)

        fact = fact.strip()
        examples = formatter.format_quick_teach(fact)

        session_id = app_state["next_session_id"]
        app_state["next_session_id"] = session_id + 1

        session = SessionState(
            session_id=session_id,
            messages=[
                {"role": "user", "content": f"Remember: {fact}"},
                {"role": "assistant", "content": f"I'll remember that."},
            ],
            summary=f"Quick teach: {fact[:60]}",
            learned=False,
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

        thread = threading.Thread(target=_learn, daemon=True)
        thread.start()

        # Add confirmation to chat
        chat_history = chat_history + [
            [f"*Quick teach:* {fact}", "Got it, I'm learning that now. Give me a moment..."]
        ]

        return chat_history, session_messages, _format_status(engine)

    def poll_status():
        """Periodic status update for the UI."""
        return _format_status(engine), _format_sidebar(app_state)

    # -----------------------------------------------------------------------
    # UI formatting helpers
    # -----------------------------------------------------------------------

    def _format_sidebar(state: dict) -> str:
        sessions = state["completed_sessions"]
        if not sessions:
            return "*No sessions yet. Start chatting!*"
        lines = []
        for s in sessions:
            icon = "+" if s.learned else "~"
            lines.append(f"[{icon}] **Session {s.session_id}**: {s.summary}")
        return "\n\n".join(lines)

    def _format_status(eng: CLUEEngine) -> str:
        status = eng.get_status()
        state = status["state"]
        if state == "idle":
            progress = status.get("progress", "")
            n_learned = eng.task_idx
            parts = [f"Sessions learned: **{n_learned}**", "Status: Ready"]
            if progress:
                parts.append(f"Last: {progress}")
            return "\n\n".join(parts)
        elif state == "training":
            return f"**Learning in progress...**\n\n{status.get('progress', '')}"
        elif state == "error":
            return f"**Error:** {status.get('progress', 'Unknown error')}"
        return "Unknown state"

    # -----------------------------------------------------------------------
    # Build Gradio layout
    # -----------------------------------------------------------------------

    with gr.Blocks(
        title="CLUED-IN",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# CLUED-IN: A Chat Agent That Actually Learns")
        gr.Markdown(
            "Each conversation session teaches the model. End a session to trigger learning. "
            "Start a new session to see what it remembers."
        )

        session_messages = gr.State([])

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat",
                    show_copy_button=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    end_btn = gr.Button("End Session", variant="stop", scale=1)

            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("## Sessions")
                sidebar = gr.Markdown(
                    _format_sidebar(app_state),
                    label="Session History",
                )

                gr.Markdown("## Status")
                status_display = gr.Markdown(
                    _format_status(engine),
                    label="Learning Status",
                )

                gr.Markdown("## Quick Teach")
                teach_input = gr.Textbox(
                    placeholder="e.g., Always use type hints in Python",
                    show_label=False,
                    lines=2,
                )
                teach_btn = gr.Button("Teach", size="sm")

        # A/B Comparison
        with gr.Accordion("A/B Comparison: Learned vs Base Model", open=False):
            gr.Markdown(
                "Enter a prompt to see how the learned model differs from the base model."
            )
            with gr.Row():
                compare_input = gr.Textbox(
                    placeholder="Enter a prompt to compare...",
                    show_label=False,
                    scale=4,
                )
                compare_btn = gr.Button("Compare", scale=1)
            with gr.Row():
                learned_output = gr.Textbox(
                    label="With Learning (merged LoRA)",
                    lines=10,
                    interactive=False,
                )
                base_output = gr.Textbox(
                    label="Base Model (no LoRA)",
                    lines=10,
                    interactive=False,
                )

        # ---------------------------------------------------------------
        # Wire up events
        # ---------------------------------------------------------------

        # Send message
        send_btn.click(
            send_message,
            inputs=[msg_input, chatbot, session_messages],
            outputs=[msg_input, chatbot, session_messages],
        )
        msg_input.submit(
            send_message,
            inputs=[msg_input, chatbot, session_messages],
            outputs=[msg_input, chatbot, session_messages],
        )

        # End session
        end_btn.click(
            end_session,
            inputs=[chatbot, session_messages],
            outputs=[chatbot, session_messages, sidebar, status_display],
        )

        # A/B comparison
        compare_btn.click(
            compare_responses,
            inputs=[compare_input],
            outputs=[learned_output, base_output],
        )

        # Quick teach
        teach_btn.click(
            quick_teach,
            inputs=[teach_input, chatbot, session_messages],
            outputs=[chatbot, session_messages, status_display],
        )

        # Poll status every 3 seconds
        timer = gr.Timer(3)
        timer.tick(
            poll_status,
            outputs=[status_display, sidebar],
        )

    return app


def main():
    app = create_app()
    config = load_config()
    app.launch(
        server_name=config["ui"]["server_name"],
        server_port=config["ui"]["server_port"],
        share=False,
    )


if __name__ == "__main__":
    main()
