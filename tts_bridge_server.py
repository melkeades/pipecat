"""
OpenAI-compatible TTS bridge for Kyutai Unmute TTS.

Exposes:
- POST /v1/audio/speech  -> returns audio/wav by synthesizing text via the Kyutai TTS WS API
- GET  /healthz          -> liveness
- GET  /v1/models        -> minimal model listing for client discovery

Environment:
- KYUTAI_TTS_URL: ws URL to the TTS server (e.g., ws://127.0.0.1:8020)
- KYUTAI_TTS_DEFAULT_VOICE: default voice name/path forwarded to TTS

Run:
- uv run python -m unmute.scripts.tts_bridge_server --host 0.0.0.0 --port 8070
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import wave
from typing import Any, Literal
import asyncio
import contextlib
import json

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from unmute.kyutai_constants import SAMPLE_RATE
from unmute.tts.text_to_speech import (
    TTSAudioMessage,
    TTSClientEosMessage,
    TextToSpeech,
)


# No default or forced voice; the bridge will only forward what the client sends.

logger = logging.getLogger(__name__)


class AudioOptions(BaseModel):
    voice: str | None = None
    format: Literal["wav"] | None = Field(default="wav")


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str | None = None
    text: str | None = None
    voice: str | None = None
    format: Literal["wav"] | None = None
    audio: AudioOptions | None = None


def _pcm_float_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    if pcm.dtype != np.float32:
        pcm = pcm.astype(np.float32)
    pcm_int16 = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


async def _synthesize(text: str, voice: str | None) -> np.ndarray:
    tts = await _connect_tts(voice)

    await tts.send(text)
    await tts.send(TTSClientEosMessage())

    chunks: list[list[float]] = []
    try:
        async for msg in tts:
            if isinstance(msg, TTSAudioMessage):
                chunks.append(msg.pcm)
    finally:
        await tts.shutdown()

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    flat = np.fromiter((s for c in chunks for s in c), dtype=np.float32)
    return flat


async def _connect_tts(voice: str | None) -> TextToSpeech:
    """Connect to the first reachable TTS endpoint and return a started client."""
    candidates: list[str] = []
    if env_url := os.environ.get("KYUTAI_TTS_URL"):
        candidates.append(env_url)
    candidates.extend(
        [
            "ws://127.0.0.1:8020",
            "ws://localhost:8020",
            "ws://127.0.0.1:8089",
            "ws://localhost:8089",
        ]
    )

    last_exc: Exception | None = None
    for url in candidates:
        tts = TextToSpeech(tts_instance=url, voice=voice)
        try:
            logger.info("TTS bridge: attempting connection to %s", url)
            await tts.start_up()
            logger.info("TTS bridge: connected to %s", url)
            return tts
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning("TTS bridge: connection to %s failed: %s", url, repr(e))
            continue

    urls_str = ", ".join(candidates)
    raise RuntimeError(f"Unable to connect to TTS at any of: {urls_str}") from last_exc


def create_app() -> FastAPI:
    app = FastAPI(title="Kyutai Unmute TTS Bridge", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "kyutai-unmute-tts", "object": "model", "owned_by": "kyutai"}
            ],
        }

    @app.post("/v1/audio/speech")
    async def audio_speech(body: SpeechRequest):
        text = body.input or body.text
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'input' (or 'text')")

        voice = body.voice or (body.audio.voice if body.audio else None)
        logger.info("TTS bridge: requested voice=%s", voice)

        fmt = body.format or (body.audio.format if body.audio else "wav")
        if fmt != "wav":
            raise HTTPException(
                status_code=400, detail="Only 'wav' format is supported"
            )

        try:
            pcm = await _synthesize(text, voice)
        except Exception as e:  # noqa: BLE001
            target_env = os.environ.get("KYUTAI_TTS_URL")
            attempted = [
                t
                for t in [
                    target_env,
                    "ws://127.0.0.1:8020",
                    "ws://localhost:8020",
                    "ws://127.0.0.1:8089",
                    "ws://localhost:8089",
                ]
                if t
            ]
            target_info = ", ".join(attempted)
            raise HTTPException(
                status_code=503,
                detail=(
                    "TTS backend unreachable. Attempted: "
                    + target_info
                    + ". Ensure the container is running (8020:8080) or dockerless on 8089. "
                    + f"Error: {e.__class__.__name__}: {e}"
                ),
            ) from e
        wav_bytes = _pcm_float_to_wav_bytes(pcm)
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.websocket("/v1/audio/speech/stream")
    async def audio_speech_stream(ws: WebSocket):
        """
        Low-latency streaming over WebSocket.

        Protocol:
        - Client connects and sends a JSON message: {"input"|"text": string, "voice": optional string, "hold_open": optional bool}
        - Server replies with a JSON header: {"type":"ready","sample_rate":24000,"encoding":"pcm_s16le"}
        - Server streams binary frames with raw PCM S16LE mono at 24kHz as TTSAudioMessage chunks arrive
        - If "hold_open": true, the client can send incremental text while the connection is open:
            * {"type":"delta","text":"..."} or legacy {"text":"..."}
            * {"type":"eos"} when no more text
            * {"type":"cancel"} to abort
          If "hold_open" is false or omitted, the server behaves as single-shot: finalize after initial text.
        - On completion, server sends {"type":"eos"} and closes
        - On error, server sends {"type":"error","detail": str} and closes
        """
        await ws.accept()
        try:
            init_msg = await ws.receive_json()
            text = init_msg.get("input") or init_msg.get("text") or ""
            voice = init_msg.get("voice")
            hold_open = bool(init_msg.get("hold_open", False))
            if not hold_open and not text:
                await ws.send_json({"type": "error", "detail": "Missing 'input' (or 'text')"})
                await ws.close(code=1003)
                return

            # Inform client about stream format as early as possible
            await ws.send_json({"type": "ready", "sample_rate": SAMPLE_RATE, "encoding": "pcm_s16le"})

            # Connect to TTS and start synthesis
            try:
                tts = await _connect_tts(voice)
            except Exception as e:  # noqa: BLE001
                target_env = os.environ.get("KYUTAI_TTS_URL")
                attempted = [
                    t
                    for t in [
                        target_env,
                        "ws://127.0.0.1:8020",
                        "ws://localhost:8020",
                        "ws://127.0.0.1:8089",
                        "ws://localhost:8089",
                    ]
                    if t
                ]
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": (
                            "TTS backend unreachable. Attempted: "
                            + ", ".join(attempted)
                            + f". Error: {e.__class__.__name__}: {e}"
                        ),
                    }
                )
                await ws.close(code=1011)
                return

            logger.info("TTS stream: starting synthesis (voice=%s)", voice)
            # Initial text (may be empty if hold_open)
            if text:
                await tts.send(text)
            if not hold_open:
                await tts.send(TTSClientEosMessage())

            frames = 0
            cancel_event = asyncio.Event()

            async def pump_ws_to_tts() -> None:
                if not hold_open:
                    # In single-shot mode, only listen for disconnect/cancel to stop early
                    try:
                        while True:
                            msg = await ws.receive()
                            if msg.get("type") == "websocket.disconnect":
                                cancel_event.set()
                                break
                            if "text" in msg and msg["text"]:
                                try:
                                    data = json.loads(msg["text"])
                                    if data.get("type") == "cancel":
                                        cancel_event.set()
                                        break
                                except Exception:
                                    # ignore raw text during single-shot
                                    pass
                    except WebSocketDisconnect:
                        cancel_event.set()
                    return

                # hold_open mode: accept deltas/eos/cancel
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            cancel_event.set()
                            break
                        if "text" in msg and msg["text"]:
                            try:
                                data = json.loads(msg["text"])
                            except Exception:
                                # Treat raw text as delta
                                await tts.send(str(msg["text"]))
                                continue

                            typ = data.get("type")
                            if typ == "delta":
                                segment = data.get("text") or ""
                                if segment:
                                    await tts.send(segment)
                            elif typ == "eos":
                                await tts.send(TTSClientEosMessage())
                                # do not break; let TTS drain
                            elif typ == "cancel":
                                cancel_event.set()
                                break
                            else:
                                # legacy {"text": "..."} or {"input": "..."}
                                segment = data.get("text") or data.get("input") or ""
                                if segment:
                                    await tts.send(segment)
                        # ignore client binary
                except WebSocketDisconnect:
                    cancel_event.set()

            async def pump_tts_to_ws() -> None:
                nonlocal frames
                try:
                    async for msg in tts:
                        if cancel_event.is_set():
                            break
                        if isinstance(msg, TTSAudioMessage):
                            pcm = np.asarray(msg.pcm, dtype=np.float32)
                            pcm = np.clip(pcm, -1.0, 1.0)
                            pcm_i16 = (pcm * 32767.0).astype(np.int16)
                            await ws.send_bytes(pcm_i16.tobytes())
                            frames += len(pcm_i16)
                finally:
                    await tts.shutdown()

            ws_task = asyncio.create_task(pump_ws_to_tts())
            tts_task = asyncio.create_task(pump_tts_to_ws())

            # Wait for TTS to finish (or be canceled)
            await tts_task

            if frames == 0 and not cancel_event.is_set():
                await ws.send_json(
                    {
                        "type": "error",
                        "detail": "No audio frames produced by TTS (check voice and server)",
                    }
                )
            else:
                await ws.send_json({"type": "eos"})
            # Close the websocket to unblock the reader task gracefully
            with contextlib.suppress(Exception):
                await ws.close()
            # Ensure the reader finishes (it should see WebSocketDisconnect)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await ws_task
        except WebSocketDisconnect:
            logger.info("TTS bridge: client disconnected from stream")
        except asyncio.CancelledError:
            # Task was canceled; treat as normal shutdown for this connection
            logger.info("TTS bridge: stream task canceled")
        except Exception as e:  # noqa: BLE001
            logger.exception("TTS bridge: stream error: %s", repr(e))
            try:
                await ws.send_json({"type": "error", "detail": f"{e.__class__.__name__}: {e}"})
            except Exception:  # noqa: BLE001
                pass
            try:
                await ws.close(code=1011)
            except Exception:  # noqa: BLE001
                pass

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Kyutai Unmute TTS OpenAI bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
