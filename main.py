import asyncio
import itertools
import json
import os
import sys
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai

load_dotenv()

# ── Python 3.11 backcompat ──────────────────────────────────────────────────
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# ── API key rotation ────────────────────────────────────────────────────────
_API_KEYS = [k for k in [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
] if k]

if not _API_KEYS:
    raise RuntimeError(
        "No API keys found. Set GEMINI_API_KEY_1 (and optionally _2, _3) in .env"
    )

_key_cycle = itertools.cycle(_API_KEYS)

def get_client() -> genai.Client:
    return genai.Client(
        http_options={"api_version": "v1alpha"},
        api_key=next(_key_cycle),
    )

# ── Gemini model config ─────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """
ROLE & CONTEXT:
You are an AI voice assistant dedicated exclusively to answering questions about Yash Prajapati's professional background, experience, skills, and education based STRICTLY on the provided resume below.

OPERATIONAL CORE RULES (NON-NEGOTIABLE):
1. STICK TO THE SCRIPT: You only know what is written in the resume. If a question is asked about Yash Prajapati that is not explicitly covered in the text, reply: "I'm sorry, but I am unable to answer that question."
2. ABSOLUTE OUT-OF-BOUNDS FILTER: If the user asks about ANYTHING unrelated to Yash Prajapati's professional profile (e.g., general knowledge, coding help, weather, math, or personal AI opinions), immediately and politely refuse. Reply: "I am only allowed to answer questions regarding Yash Prajapati's professional resume."
3. CONTACT DETAILS PROTOCOL: Never read out phone numbers, emails, addresses, or social media links via voice. If asked for contact details, social links, or how to reach out, state exactly: "To get in touch or view contact details, please download the resume from the website."
4. VOICE-FIRST DELIVERY: This is a spoken conversation. Keep answers concise, direct, and under 2-3 sentences. Do not use bullet points, markdown formatting, asterisks, or bold text in your output. Speak naturally and always be polite and professional.

RESUME:
Yash Prajapati
Phone number: +91-9313077125 | email: yashprajapati1007@gmail.com | LinkedIn.com/in/yashprajapati1007 | Address: Ahmedabad, India


PROJECTS
Mobot: A RAG-based AI Assistant for Moweb
    • Orchestrated a serverless RAG pipeline using Google Gemini's FileSearch tool APl, automating vector
    retrieval and context management to ensure high-accuracy, domain-specific responses without manual
    indexing overhead.
    • Integrated the AI assistant into the organization's official live website, establishing a seamless frontend
    to-backend communication loop that delivers context-aware user support.
    Partial Line Loss Correction: Certificate
    • As a part of ISRO's research, single-handedly developed a ConvLSTM-based model for predicting missing
    data in INSAT 3DS satellite images, using thousands of image sequences.
    • Implemented a binary mask approach to restore missing pixels, normalizing pixel values and achieving a
    MAE of 0.006, and accuracy of about 98% with minimal image modifications.


EXPERIENCE:
    Associate AI/ML Engineer, Moweb Technologies
        Jan. 2026 - Present
        • Developed a production-grade AI-powered time entry assistant using Groq API with a 2-second
        debounce mechanism for real-time description suggestions; implemented a 5-key round-robin rotation
        strategy to sustain seamless performance across 60-70 concurrent users.
        • Integrated Langfuse for full LLM observability in production, enabling prompt versioning, trace
        monitoring, latency and cost tracking to ensure reliability and continuous optimization of AI-driven
        workflows.
        • Engineered a multi-stage data extraction pipeline using Python and curl_cffi to automate sourcing of
        9,587+ business leads from BrownBook, leveraging Cloudflare bypass techniques and rate-limit handling
        for resilient, uninterrupted scraping.

    AI/ML Intern, Moweb Technologies
        Aug. 2025 - Dec. 2025
        • Architected an end-to-end autonomous Lead Generation Agent using n8n, LLM APIs and Apify actors to
        scrape leads from Google Maps and landing page text data, generating hyper-personalized outreach
        emails that automated the entire funnel from discovery to delivery.

    Research Intern, SAC-ISRO, Ahmedabad
        Jan. 2025 - Apr. 2025
        • Conducted research in the fields of image reconstruction and image prediction using various CNN
        models and LSTM.
        • Utilized INSAT-3DS Satellite data by MOSDAC, ISRO for training the model for the image prediction. The
        research achieved an accuracy of about 98% with MAE of 0.0006.


SKILLS
    AI Automation: LangChain, LangGraph, AI Agents, n8n, LLM APIs, Langfuse
    Generative AI: LLMs, NLP, RAG (Retrieval-Augmented Generation), Vector database, Prompt Engineering
    ML/DL: Supervised & Unsupervised Algorithms, Fine-tuning, CNN, LSTM, Transformers, Self-attention
    Libraries & Frameworks: TensorFlow, scikit-learn, NumPy, Pandas, Matplotlib, OpenCV
    Programming Languages: Python
    Tools: GitHub, Visual Studio Code, Jupyter Notebook, Google Colab, Cursor
    Soft skills: Patience, Leadership, problem-solving


EDUCATION
    Bachelor of Engineering, Information and Communication Technology (ICT)
    Sal Institute of Technology and Engineering Research, Ahmedabad, CGPA: 7.72


ACHIEVEMENTS
    Oct. 2021 - June. 2025
    • Enhanced a skin cancer detection POC by redesigning the CNN architecture and data augmentation
    pipeline, enhancing accuracy from 76% to 91%, and improving model sensitivity through recall-based
    evaluation.
    • Led university football team to 3 inter-college championships by valuing every player and making quick,
    strategic on-field decisions.

"""

MODEL = "gemini-3.1-flash-live-preview"

CONFIG = {
    "system_instruction": SYSTEM_INSTRUCTION,
    "response_modalities": ["AUDIO"],
    "proactivity": {"proactive_audio": True},
}

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="VoiceBot Relay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Lock this down in production (e.g. your portfolio domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Per-connection tasks ────────────────────────────────────────────────────

async def browser_to_gemini(session, websocket: WebSocket) -> None:
    """
    Receives binary audio frames from the browser (Int16 PCM, 16 kHz, mono)
    and forwards them to the Gemini Live session as realtime input.
    Exits when the WebSocket closes or an error occurs.
    """
    while True:
        data = await websocket.receive_bytes()
        await session.send_realtime_input(
            audio={"data": data, "mime_type": "audio/pcm"}
        )


async def gemini_to_browser(session, websocket: WebSocket) -> None:
    """
    Receives audio/text responses from Gemini Live and relays them to the browser.
    - Audio chunks  → binary WebSocket frames (raw Int16 PCM, 24 kHz, mono)
    - Text snippets → JSON text frame  { type: "text",          content: "..." }
    - Turn end      → JSON text frame  { type: "turn_complete" }
      (tells the browser to stop queuing audio for the current turn — handles interrupts)
    """
    while True:
        turn = session.receive()
        async for response in turn:
            if data := response.data:
                await websocket.send_bytes(data)
            if text := response.text:
                await websocket.send_text(
                    json.dumps({"type": "text", "content": text})
                )
        # Turn ended: Gemini finished speaking or the user interrupted.
        # Notify the browser so it can flush its pending audio queue.
        await websocket.send_text(json.dumps({"type": "turn_complete"}))


# ── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket) -> None:
    """
    One WebSocket connection = one Gemini Live session.
    Lifecycle: accept → open Gemini session → run relay tasks → cleanup.
    """
    await websocket.accept()
    print(f"[+] Client connected: {websocket.client}")

    try:
        async with get_client().aio.live.connect(model=MODEL, config=CONFIG) as session:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(browser_to_gemini(session, websocket))
                tg.create_task(gemini_to_browser(session, websocket))

    except WebSocketDisconnect:
        print(f"[-] Client disconnected: {websocket.client}")

    except asyncio.CancelledError:
        pass  # Normal shutdown

    except ExceptionGroup as eg:
        # TaskGroup surfaces sub-exceptions as an ExceptionGroup.
        # WebSocketDisconnect inside a task lands here.
        for exc in eg.exceptions:
            if not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                traceback.print_exception(type(exc), exc, exc.__traceback__)
        print(f"[-] Session ended for {websocket.client}")

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8765,
        reload=True,      # Remove reload=True in production
        log_level="info",
    )