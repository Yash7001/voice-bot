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
+91-9313077125 | yashprajapati1007@gmail.com| LinkedIn.com/in/yashprajapati1007
SUMMARY											           
AI/ML Engineer experienced in building Generative AI applications, LLM-powered conversational AI, RAG pipelines, prompt engineering and AI workflows. Hands-on experience with LangChain, Gemini, ChromaDB, FastAPI, Python, LLM evaluation, monitoring, and AI application optimization.
PROJECTS											           
Mobot: A RAG-based AI Assistant for Moweb
•	Orchestrated a serverless RAG pipeline using Google Gemini’s FileSearch tool API, automating vector retrieval and context management to ensure high-accuracy, domain-specific responses without manual indexing overhead.
•	Integrated the conversational AI assistant into the organization’s official live website via FastAPI based REST API, establishing a seamless frontend-to-backend communication loop that delivers context-aware user support.
•	Followed Agile development practices with Git-based version control for iterative pipeline development and cross-functional teams collaboration.
Partial Line Loss Correction: Certificate
•	As a part of ISRO’s research, single-handedly developed a ConvLSTM-based model for predicting missing data in INSAT 3DS satellite images, using thousands of image sequences.
•	Implemented a binary mask approach to restore missing pixels, normalizing pixel values and achieving a MAE of 0.006, and accuracy of 98% with minimal image modifications.

EXPERIENCE											           
Associate AI/ML Engineer, Moweb Technologies	     			  	          Jan. 2026 – Present
•	Engineered a RAG ingestion pipeline that processes 50+ markdown knowledge-base documents into 500+ semantically segmented chunks using heading-aware section parsing, 900-character chunk sizing with 180-character overlap, SentenceTransformers embeddings, and ChromaDB persistence with per-chunk metadata for filename, section, and chunk index.
•	Designed a hybrid retrieval system combining dense vector search with lexical keyword, domain-specific query expansion, and prompt engineering techniques, evaluates 18 retrieval candidates per query and custom re-ranking to return top-6 context chunks and automatically falls back to full-index lexical search when semantic retrieval fails.
•	Implemented an end-to-end LLM observability and evaluation pipeline in Langfuse - prompt versioning, traces, cost monitoring, latency, token usage, evaluating 1000+ production traces at an estimated compute cost below $0.25 while surfacing failure patterns to guide iterative pipeline improvements and MLOps-driven model deployment decisions.
AI/ML Intern, Moweb Technologies	     			  	                     Aug. 2025 – Dec. 2025
•	Experience mentioned in the projects and achievement section
Research Intern, SAC-ISRO, Ahmedabad: Certificate				      Jan. 2025 – Apr. 2025
•	Conducted research in the fields of image reconstruction and image prediction using various CNN architectures and LSTM algorithm.
•	Utilized INSAT-3DS Satellite data by MOSDAC, ISRO for training the model for the image reconstruction.
SKILLS											           
Generative AI: LLMs, NLP, TF-IDF, RAG (Retrieval-Augmented Generation), Vector database, chromaDB, Prompt Engineering, MLOps, Model Deployment, Gemini API, Conversational AI, 
ML/DL/Fine-tuning: Supervised & Unsupervised Algorithms, CNN, LSTM, Transformers, LoRA, QLoRA
Libraries & Frameworks: LangChain, TensorFlow, scikit-learn, NumPy, Matplotlib, OpenCV, FastAPI, MLflow
Programming Languages: Python | Tools: Git, Github, Cursor, VS Code, Jupyter Notebook, Google Colab
Soft Skills: Problem-solving, Patient, Excellent Communicator, Continuous learner
EDUCATION											           
Bachelor of Engineering, Information and Communication Technology (ICT)		Oct. 2021 – June. 2025
Sal Institute of Technology and Engineering Research, Ahmedabad, CGPA: 7.72
ACHIEVEMENTS										           
•	Enhanced skin cancer detection POC by redesigning the CNN architecture and data augmentation pipeline, enhancing accuracy from 76% to 91% and model sensitivity through recall-based evaluation.

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
    - Interrupted   → JSON text frame  { type: "interrupted" }
      (user barged in — browser must stop playback and drop stale audio)
    - Turn end      → JSON text frame  { type: "turn_complete" }
    """
    while True:
        turn = session.receive()
        async for response in turn:
            sc = response.server_content
            if sc and sc.interrupted:
                await websocket.send_text(json.dumps({"type": "interrupted"}))
            if data := response.data:
                await websocket.send_bytes(data)
            if text := response.text:
                await websocket.send_text(
                    json.dumps({"type": "text", "content": text})
                )
        await websocket.send_text(json.dumps({"type": "turn_complete"}))

# ── Health check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "alive"}

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
        host="0.0.0.0",
        port=8765,
        reload=True,      # Remove reload=True in production
        log_level="info",
    )
