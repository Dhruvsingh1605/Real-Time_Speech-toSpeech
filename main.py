import io
import asyncio
import logging
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
import whisper
from sentence_transformers import SentenceTransformer
from openvoice import OpenVoice
import webrtcvad
import soundfile as sf

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
phi_model.eval()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load models at startup
dpt_model = whisper.load_model("base")  # Fast Whisper model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tts = OpenVoice()

# Voice Activity Detector (VAD) setup
vad = webrtcvad.Vad(2)  # aggressiveness: 0-3
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000

# Simple in-memory conversation context per connection
MAX_CONTEXT_LEN = 5  # number of turns to keep

async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using Whisper."""
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        result = await run_in_threadpool(dpt_model.transcribe, audio_buffer)
        text = result.get("text", "").strip()
        logger.info(f"Transcribed: {text}")
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

async def embed_text(text: str):
    """Compute semantic embeddings of text."""
    try:
        return await run_in_threadpool(embed_model.encode, text)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

async def generate_response(context: deque) -> str:
    """Generate a response using the microsoft/phi-2 model."""
    try:
        # Combine last few user/assistant turns into a prompt
        prompt = "You are a helpful assistant.\n"
        for turn in context:
            if "user" in turn:
                prompt += f"User: {turn['user']}\n"
            if "assistant" in turn:
                prompt += f"Assistant: {turn['assistant']}\n"
        prompt += "Assistant:"

        # Tokenize and generate
        inputs = phi_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = await run_in_threadpool(
                lambda: phi_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=phi_tokenizer.eos_token_id
                )
            )

        response = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after last 'Assistant:'
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return "Sorry, I encountered an error generating a response."


async def synthesize_speech(text: str) -> bytes:
    """Synthesize speech from text using OpenVoice TTS."""
    try:
        audio_data = await run_in_threadpool(tts.synthesize, text)
        return audio_data
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return b""

def is_speech(audio_bytes: bytes) -> bool:
    """Simple VAD that checks if a frame contains speech."""
    try:
        pcm, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='int16')
        # Ensure correct rate
        if sample_rate != SAMPLE_RATE:
            return True  # fallback to processing
        frame_bytes = pcm.tobytes()
        return vad.is_speech(frame_bytes, SAMPLE_RATE)
    except Exception as e:
        logger.warning(f"VAD error: {e}")
        return True

@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    context = deque(maxlen=MAX_CONTEXT_LEN)
    logger.info("Client connected")
    try:
        while True:
            # Receive binary audio from client
            audio_bytes = await websocket.receive_bytes()

            # Voice activity detection
            if not is_speech(audio_bytes):
                logger.debug("Silence detected, skipping processing.")
                continue

            # Transcription
            user_text = await transcribe_audio(audio_bytes)
            if not user_text:
                continue

            context.append({'user': user_text})

            # Embedding (unused in stub)
            _ = await embed_text(user_text)

            # Generate response
            response_text = await generate_response(context)
            context.append({'assistant': response_text})
            logger.info(f"Responding: {response_text}")

            # TTS synthesis
            response_audio = await synthesize_speech(response_text)
            if response_audio:
                await websocket.send_bytes(response_audio)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Connection error: {e}")
        await websocket.close(code=1011, reason="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
