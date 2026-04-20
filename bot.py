"""
WhispryBot — Telegram bot that transcribes voice and audio locally with faster-whisper.

Copy .env.example to .env and set TELEGRAM_BOT_TOKEN (and optional WHISPER_* / ALLOWED_USER_IDS).
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_USER_IDS = {
    int(x.strip()) for x in os.environ.get("ALLOWED_USER_IDS", "").split(",") if x.strip()
}
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
TRANSCRIBE_TIMEOUT_S = int(os.environ.get("WHISPER_TIMEOUT_S", "900"))
WARMUP_ON_START = os.environ.get("WHISPER_WARMUP_ON_START", "1").strip() not in {"0", "false", "False"}

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisprybot")

# Silence libraries that log full request URLs (includes bot token).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

_model: WhisperModel | None = None
_model_lock = threading.Lock()


def _repo_id_for_model_size(size: str) -> str:
    return f"Systran/faster-whisper-{size}"


def warmup_model() -> None:
    """Download + initialize Whisper model (runs in a thread)."""
    _ = get_model()


async def _post_init(app: object) -> None:
    # Runs once after PTB is initialized, before polling starts.
    # We start warmup in background so /start remains responsive.
    if not WARMUP_ON_START:
        logger.info("Model warmup disabled (WHISPER_WARMUP_ON_START=0).")
        return
    logger.info("Starting model warmup in background…")
    asyncio.create_task(asyncio.to_thread(warmup_model))


def get_model() -> WhisperModel:
    """Load Whisper on first use so Telegram polling can start immediately."""
    global _model
    with _model_lock:
        if _model is None:
            repo_id = _repo_id_for_model_size(MODEL_SIZE)
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id, allow_patterns=["*"], local_files_only=False)
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        return _model


def _transcribe_file_sync(in_path: str) -> tuple[str, Any]:
    model = get_model()
    segments, info = model.transcribe(
        in_path,
        beam_size=BEAM_SIZE,
        vad_filter=True,
        language=LANGUAGE,
    )
    parts: list[str] = []
    seg_count = 0
    for seg in segments:
        seg_count += 1
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    text = " ".join(parts).strip()
    return text, info


def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def safe_name(name: str) -> str:
    name = Path(name or "audio").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    user = update.effective_user
    if not msg:
        return
    if not user or not is_allowed(user.id):
        await msg.reply_text("Not allowed.")
        return
    await msg.reply_text(
        "Send me a Telegram voice note or an audio file and I will transcribe it locally."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    await msg.reply_text(
        "/start - intro\n"
        "/help - this help\n"
        "/lang xx - force language, e.g. /lang it or /lang en\n"
        "/lang auto - auto detect again"
    )


async def set_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global LANGUAGE
    msg = update.effective_message
    user = update.effective_user
    if not msg:
        return
    if not user or not is_allowed(user.id):
        await msg.reply_text("Not allowed.")
        return
    if not context.args:
        await msg.reply_text(f"Current language: {LANGUAGE or 'auto'}")
        return
    value = context.args[0].strip().lower()
    LANGUAGE = None if value == "auto" else value
    await msg.reply_text(f"Language set to: {LANGUAGE or 'auto'}")


async def transcribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    msg = update.effective_message
    if not user or not msg or not is_allowed(user.id):
        if msg:
            await msg.reply_text("Not allowed.")
        return

    media = None
    filename = "audio"
    if msg.voice:
        media = msg.voice
        filename = f"voice_{media.file_unique_id}.ogg"
    elif msg.audio:
        media = msg.audio
        filename = safe_name(msg.audio.file_name or f"audio_{msg.audio.file_unique_id}")
    elif msg.document and (msg.document.mime_type or "").startswith("audio/"):
        media = msg.document
        filename = safe_name(msg.document.file_name or f"audio_{msg.document.file_unique_id}")
    else:
        await msg.reply_text("Send a voice note or an audio file.")
        return

    status = await msg.reply_text("Downloading audio...")
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, filename)
        tg_file = await context.bot.get_file(media.file_id)
        await tg_file.download_to_drive(custom_path=in_path)

        await status.edit_text("Transcribing locally...")
        try:
            text, info = await asyncio.wait_for(
                asyncio.to_thread(_transcribe_file_sync, in_path),
                timeout=TRANSCRIBE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.exception("Transcription timed out after %ss", TRANSCRIBE_TIMEOUT_S)
            await status.edit_text(
                f"Transcription timed out after {TRANSCRIBE_TIMEOUT_S}s. "
                "Try a shorter clip or use a smaller model."
            )
            return
        except Exception:
            logger.exception("Transcription failed")
            await status.edit_text("Transcription failed. Check logs or try again.")
            return

        if not text:
            text = "[No speech detected]"

        meta = f"Language: {info.language or 'unknown'}\nDuration: {round(info.duration, 1)}s"
        out_text = f"{meta}\n\n{text}"

        if len(out_text) <= 4000:
            await status.edit_text(out_text)
        else:
            await status.edit_text(meta)
            txt_path = os.path.join(tmpdir, "transcript.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            with open(txt_path, "rb") as f:
                await msg.reply_document(document=f, filename="transcript.txt", caption="Transcript")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Error: something went wrong.")
        except Exception:
            pass


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")

    app = ApplicationBuilder().token(TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("lang", set_lang))
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
            transcribe,
        )
    )
    app.add_error_handler(error_handler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
