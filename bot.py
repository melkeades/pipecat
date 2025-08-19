#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

The example connects between client and server using a P2P WebRTC connection.

Run the bot using::

    python bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading AI models (30-40 seconds first run, <2 seconds after)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline

from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments

# from pipecat.services.cartesia.tts import CartesiaTTSService
# from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

import os
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.transcriptions.language import Language

from pipecat.services.kokoro import KokoroTTSService

logger.info("✅ Pipeline components loaded")

logger.info("Loading WebRTC transport...")
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    stt = WhisperSTTService(
        model=Model.LARGE_V3_TURBO,
        device="cuda",
        compute_type="float16",  # Reduce memory usage
        no_speech_prob=0.3,  # Lower threshold for speech detection
        language=Language.EN,  # Specify language for better performance
    )

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    # )
    tts = KokoroTTSService(
        model_path="tts-models/kokoro-v1.0.onnx",
        voices_path="tts-models/voices-v1.0.bin",
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    # llm = OLLamaLLMService(
    #     model="qwen2.5:7b",  # Must be pulled first: ollama pull llama3.1
    #     base_url="http://localhost:11434/v1",  # Default Ollama endpoint
    #     params=OLLamaLLMService.InputParams(temperature=0.7, max_tokens=1000),
    # )
    llm = OLLamaLLMService(
        # model="qwen/qwen3-4b-2507",  # Must be pulled first: ollama pull llama3.1
        # base_url="http://127.0.0.1:1234/v1",  # Default Ollama endpoint
        model="qwen2.5:7b",  # Must be pulled first: ollama pull llama3.1
        base_url="http://localhost:11434/v1",  # Default Ollama endpoint
        params=OLLamaLLMService.InputParams(temperature=0.7, max_tokens=1000),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational. NEVER ADD emojis!!! NEVER USE emojis!!! NEVER ADD smiley faces!!! NEVER USE smiley faces!!!",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Say hello and briefly introduce yourself."}
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


# smart_turn_model_path = os.getenv("LOCAL_SMART_TURN_MODEL_PATH")
smart_turn_model_path = "H:/Git/pipecat/smart-turn-v2"


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # vad_analyzer=SileroVADAnalyzer(),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path=smart_turn_model_path,
                params=SmartTurnParams(
                    stop_secs=2.0, pre_speech_ms=0.0, max_duration_secs=8.0
                ),
            ),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
