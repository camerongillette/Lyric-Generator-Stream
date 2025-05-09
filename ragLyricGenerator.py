import asyncio
import os
import json
from fastapi import FastAPI, Request
from langchain_text_splitters import CharacterTextSplitter
from param import String
from pydantic import BaseModel
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from slowapi.util import get_remote_address
from SecurityHeaderMiddleware import SecurityHeadersMiddleware;
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import Request

LYRIC_FILE_NAME="cam_lyrics"

load_dotenv()
frontend_urls = os.getenv('FRONTEND_URLS', 'MISSING FRONTEND_URLS ENV VAR')
print(f"Frontend URL: {frontend_urls}")
frontend_urls = [url.strip() for url in frontend_urls.split(',') if url.strip()]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_urls,  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)

# ---- RATE LIMITING ----
limiter = Limiter(key_func=get_remote_address)

# ───────────────────────────────
# 1. Load lyrics from JSON
def load_lyrics(json_path=LYRIC_FILE_NAME+".json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure it's a list of strings
    if isinstance(data, list):
        return [str(lyric).strip() for lyric in data if str(lyric).strip()]
    else:
        raise ValueError("Expected a JSON array of strings in lyrics.json")


# ───────────────────────────────
# 2. Connect to Weaviate (v4+ REST only)
def connect_weaviate():
    return weaviate.connect_to_local(
        port=8080
    )

# ───────────────────────────────
# 3. Setup vector store (create or reuse)
def get_vectorstore(client, index_name=LYRIC_FILE_NAME):
    embedding_model = OpenAIEmbeddings()

    # DEBUG MODE ON!!
    if index_name in client.collections.list_all():
        print(f"📦 Using existing Weaviate index: {index_name}")
        return WeaviateVectorStore(
            client=client,
            embedding=embedding_model,
            index_name=index_name,
            by_text=False
        )
    else:
        print(f"🆕 Creating Weaviate index: {index_name}")
        lyrics = load_lyrics()
        docs = [Document(page_content=lyric) for lyric in lyrics]
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        return WeaviateVectorStore.from_documents(chunks, embeddings, client=client)

# ───────────────────────────────
# 4. Build prompt for Ollama
def build_prompt(chunks, topic):
    context = "\n\n".join(chunks)
    return f"""Write a new song based on the TOPIC below. Use the LYRICS as a guide for style, emotion, word choice, rhyme scheme, line length, variety between, match chorus style to [chorus] and verse style to [verse]. Do not include a header or footer explaining the song, only write the lyrics themselves and section titles like [chorus]

LYRICS:
{context}

TOPIC:
{topic}
"""

def build_prompt_existing(chunks, topic, existing_lyrics):
    print("Building prompt")
    context = "\n\n".join(chunks)
    prompt=""
    if(existing_lyrics):
        prompt += f"""Complete the song by finishing any unfinished sections. If a section is incomplete, finish it fully before moving on to the next section. Once the current section is completed, proceed to the next logical section of the song, using appropriate section labels (e.g., "[Verse 2]", "[Bridge]"). Do not repeat any part of the song that already exists, including section labels. 
    """
    else :
        prompt += f"""You are a professional songwriter and outputting only lyrics."""
    prompt += f"""**Do not add any introductory phrases or explanations.** Only output the lyrics, with the section labels and lyrics as appropriate. Only provide the **finished lyrics**—no extra text, no "here's my attempt," no "completing the song," just the song itself.
    - Do not generate messages about the lyrics. Only output lyrics. 
- If a section is unfinished, complete it.
- If the current section is finished, move on to the next section (e.g., from chorus to verse or bridge).
"""
    if(existing_lyrics):prompt+=f"""
    Existing Song:
    {existing_lyrics}"""

    prompt += f"""
    The song will be about {topic}

    Once you finish the current section, proceed to the next logical part of the song. Only output the next parts of the song. Use the INSPIRATION LYRICS FOR STYLE as a general guide for style, emotion, word choice, rhyme scheme, line length, variety between, match chorus style to [chorus] and verse style to [verse]. Don't copy exact phrases from LYRICS, only use it as inspiration
    Do not output a title or a song title at all.
    INSPIRATION LYRICS FOR STYLE:
    {context}

    """
    return prompt


    prompt = "You are a songwriter"
    if(existing_lyrics): 
        prompt += f"""EXISTING_SONG:```{existing_lyrics}```\n \n

    Continue writing the song defined in triple ticks EXISTING_SONG. Analyze the song EXISTING_SONG and write only lyrics that would make sense in that context. 
    'Sections' of a song are defined by the following keywords: [Verse 1],[Verse], [Chorus], [Verse 2], [Prechorus], [Bridge], [Bridge 2], [Intro], [Outro]. 
    First analyze the what's written in the last section of EXISTING_SONG and finish that section before creating a new one. 
    If a section is not present, you can add it. If a section is present, you can change it.
    If a section already exists, do not repeat or rewrite them. Make sure to complete sections before starting a new section. Do not repeat sections. Avoid duplicate sections. Detect if we have finished a section before starting a new section"
    Duplicate existing chorus sections. Duplicate rhythm of existing verse sections.
    Avoid one line sections and duplicate sections. Do not have sections like this :
    [Verse 1]
    In darkness, whispers carved in pain
    [Verse 1]
    Shadows on the wall, whispers small

    Here is an example of failing to finish the chorus section before starting the verse section. In this example, the chorus section needs to be extended before starting the next section
    [Chorus]
    We're fragments of a broken design
    [Verse]
    Shadows carve their mark, in the darkest part
    """;
    else:
        prompt += f"""Write only lyrics """

    prompt = f""" based on the TOPIC below. Use the LYRICS as a general guide for style, emotion, word choice, rhyme scheme, line length, variety between, match chorus style to [chorus] and verse style to [verse]. Don't copy exact phrases from LYRICS, only use it as inspiration
     Do not include a header or footer explaining the song, only write the lyrics themselves and section titles like [chorus].
    Use the LYRICS as guide for Section organization. Do not create a new section until the previous one is finished. Always finish the current line before starting a new section.
    """;
    
    prompt +=f"""
    LYRICS:
    {context}

    TOPIC:
    {topic}

    Do not include any explanation or commentary. Only write lyrics. Don't output any phrases like 'Here is rewritten lyrics'
    Be creative.
    """
    return prompt

# ───────────────────────────────
# Used to stream output from the llm 
class CustomStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put(token)

    async def stream_tokens(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield f"{token}"
        yield "data: [DONE]\n\n"  # Explicit end-of-stream message

class ContinueSongRequest(BaseModel):
    topic: str
    existing_lyrics: str

@app.post("/lyricgenerator/songwriter")
@limiter.limit("10/minute")
async def generate_lyrics_endpoint(request: Request, payload: ContinueSongRequest):
    topic = payload.topic
    existing_lyrics = payload.existing_lyrics
    client = connect_weaviate()
    vectorstore = get_vectorstore(client)
    try:
        print("Song RAG Generator (LangChain + OpenAI + Ollama)")
        #while True:
        print("🔍 Retrieving relevant lyrics...")
        chunks = vectorstore.similarity_search(topic, k=4)
        lyrics_texts = [doc.page_content for doc in chunks]
        handler = CustomStreamHandler()

        #print(prompt)
        llm = ChatOllama(model="llama3", streaming=True, callbacks=[handler])
        prompt = build_prompt_existing(lyrics_texts, topic, existing_lyrics)
        print("Generated Prompt\n\n " + prompt)

        async def event_stream():
            try:
                task = asyncio.create_task(llm.ainvoke(prompt))
                async for token in handler.stream_tokens():
                    if await request.is_disconnected():
                        print("❌ Client disconnected")
                        break
                    yield token
            finally:
                await handler.queue.put(None)  # Signal the handler to stop streaming
                await task
                print("✅ Stream finished")

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    finally:
        print("🔌 Closing Weaviate connection...")
        client.close()

@app.get("/lyricgenerator/topic")
@limiter.limit("10/minute")
async def generate_lyrics_endpoint(request: Request, topic: str):
    client = connect_weaviate()
    vectorstore = get_vectorstore(client)
    try:
        print("Song RAG Generator (LangChain + OpenAI + Ollama)")
        #while True:
        print("🔍 Retrieving relevant lyrics...")
        chunks = vectorstore.similarity_search(topic, k=4)
        lyrics_texts = [doc.page_content for doc in chunks]

        prompt = build_prompt(lyrics_texts, topic)
        handler = CustomStreamHandler()

        print("\n🧠 Prompt being sent to Ollama:\n")
        print(prompt)
        llm = ChatOllama(model="llama2-uncensored", streaming=True, callbacks=[handler])
        
        print("🔍 Retrieving relevant lyrics...")
        chunks = vectorstore.similarity_search(topic, k=4)
        lyrics_texts = [doc.page_content for doc in chunks]
        prompt = build_prompt(lyrics_texts, topic)

        async def event_stream():
            try:
                task = asyncio.create_task(llm.ainvoke(prompt))
                async for token in handler.stream_tokens():
                    if await request.is_disconnected():
                        print("❌ Client disconnected")
                        break
                    yield token
            finally:
                await handler.queue.put(None)  # Signal the handler to stop streaming
                await task
                print("✅ Stream finished")

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    finally:
        print("🔌 Closing Weaviate connection...")
        client.close()