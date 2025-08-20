"""
Kyutai Unmute streaming TTS service (via OpenAI-compatible bridge).

This service connects to the WebSocket endpoint exposed by the local bridge
defined in `tts_bridge_server.py` and supports low-latency, on-the-fly speech
as LLM tokens arrive. It keeps the stream open while text deltas are sent,
then sends EOS to finalize and lets the bridge close the connection.

Recommended bridge endpoint (default): ws://127.0.0.1:8070/v1/audio/speech/stream

Notes:
- Uses raw PCM S16LE mono at the server-provided sample rate (defaults to 24000Hz).
- Designed to work well with SmartTurn v2: we emit TTSStarted/TTSStopped frames
  and respect interruptions (sending cancel and reconnecting as needed).
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    EndFrame,
    LLMFullResponseEndFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from websockets.exceptions import ConnectionClosedOK

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Kyutai TTS bridge, you need to `pip install websockets`."
    )
    raise Exception(f"Missing module: {e}")


class KyutaiTTSService(InterruptibleTTSService):
    """Streaming TTS using Kyutai Unmute bridge over WebSocket.

    The service opens a WebSocket on the first text chunk, sends deltas for
    subsequent chunks, and on flush (end-of-turn) sends EOS. The bridge will
    then stream remaining audio, send an 'eos' JSON, and close the socket.
    """

    def __init__(
        self,
        *,
        base_ws_url: str = "ws://127.0.0.1:8070/v1/audio/speech/stream",
        voice: Optional[str] = None,
        # Kyutai Unmute commonly uses 24000Hz; we'll adopt this unless the server says otherwise
        sample_rate: Optional[int] = 24000,
        aggregate_sentences: bool = False,
        # Forward end-of-turn frames so downstream processors/UI update correctly
        push_text_frames: bool = True,
        push_stop_frames: bool = True,
        pause_frame_processing: bool = True,
        **kwargs,
    ):
        """Initialize the Kyutai TTS service.

        Args:
            base_ws_url: Bridge WS endpoint (see tts_bridge_server.py).
            voice: Optional voice id/path to forward to the bridge.
            sample_rate: Target output sample rate; updated from bridge 'ready'.
            aggregate_sentences: Disable sentence aggregation to speak on the fly.
            push_text_frames: Don't push TTSTextFrame per chunk; let context aggregators handle.
            push_stop_frames: Auto-send TTSStopped on idle.
            pause_frame_processing: Pause frame processing while audio is playing to avoid overlap.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=push_text_frames,
            push_stop_frames=push_stop_frames,
            pause_frame_processing=pause_frame_processing,
            sample_rate=sample_rate,
            **kwargs,
        )

        if base_ws_url.endswith("/"):
            base_ws_url = base_ws_url[:-1]

        self._url = base_ws_url
        self._voice_id = voice or ""

        # Internal state per utterance
        self._stream_active: bool = False
        self._receive_task = None
        self._got_first_audio = False

    def can_generate_metrics(self) -> bool:
        return True

    async def _connect(self):
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self._websocket = await websocket_connect(self._url)
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                # Best-effort close
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            self._stream_active = False
            self._got_first_audio = False

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(
        self, frame: StartInterruptionFrame, direction: FrameDirection
    ):
        # Attempt to cancel current synthesis to reduce tail latency
        try:
            if self._stream_active and self._websocket:
                await self._get_websocket().send(json.dumps({"type": "cancel"}))
        except Exception:
            pass
        await super()._handle_interruption(frame, direction)

    async def _send_init_if_needed(self, first_text: str):
        if self._stream_active:
            return
        await self._connect()
        payload = {"hold_open": True}
        if first_text:
            payload["input"] = first_text
        if self._voice_id:
            payload["voice"] = self._voice_id
        await self.start_ttfb_metrics()
        self._got_first_audio = False
        await self._get_websocket().send(json.dumps(payload))
        self._stream_active = True

    async def _send_delta(self, text: str):
        if not text:
            return
        try:
            await self._get_websocket().send(
                json.dumps({"type": "delta", "text": text})
            )
        except Exception as e:
            logger.error(f"{self} error sending delta: {e}")
            # Let receive task handle reconnection via error path

    async def flush_audio(self):
        """Finalize current utterance by sending EOS to the bridge."""
        if not self._stream_active or not self._websocket:
            return
        try:
            await self._get_websocket().send(json.dumps({"type": "eos"}))
        except Exception as e:
            logger.error(f"{self} error sending eos: {e}")

    async def _receive_messages(self):
        sent_explicit_eos = False
        async for message in self._get_websocket():
            try:
                if isinstance(message, (bytes, bytearray)):
                    if not self._got_first_audio:
                        await self.stop_ttfb_metrics()
                        self._got_first_audio = True
                    # Raw PCM S16LE mono @ current sample rate
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=bytes(message),
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )
                    )
                else:
                    # JSON control messages
                    data = json.loads(message)
                    typ = data.get("type")
                    if typ == "ready":
                        sr = int(data.get("sample_rate", 0) or 0)
                        if sr and sr != self.sample_rate:
                            # Update output sample rate to match bridge
                            self._sample_rate = sr
                    elif typ == "error":
                        await self.push_error(
                            ErrorFrame(error=f"Kyutai TTS error: {data.get('detail')}")
                        )
                    elif typ == "eos":
                        sent_explicit_eos = True
                        await self.push_frame(TTSStoppedFrame())
                        # Bridge will close shortly after
                        self._stream_active = False
            except Exception as e:
                logger.error(f"{self} receive error: {e}")
                raise
        # Normal close of stream: if we produced audio but didn't get EOS, finalize now
        if self._got_first_audio and not sent_explicit_eos:
            try:
                await self.push_frame(TTSStoppedFrame())
            except Exception:
                pass
        raise ConnectionClosedOK(1000, "Kyutai bridge closed stream")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Send text (or deltas) to the Kyutai bridge and stream audio back.

        We yield TTSStartedFrame on the first chunk of an utterance; audio is
        streamed from the receive task as TTSAudioRawFrame.
        """
        try:
            if not self._stream_active:
                # First chunk for this utterance
                await self._send_init_if_needed(first_text=text)
                yield TTSStartedFrame()
            else:
                await self._send_delta(text)
            # No direct audio yield here; audio arrives via _receive_messages
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=str(e))

    # stop/cancel behavior: use base InterruptibleTTSService cleanup

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Ensure we finalize the stream at end-of-turn for low tail latency."""
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()
