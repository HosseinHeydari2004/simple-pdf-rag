import tempfile
from unittest import result

from chromadb.utils import lru_cache
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline

from src.loaders.pipeline import _loader
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path

app = FastAPI(
    title="simple pdf rag",
    version="1.0.0",
)


class IngestResponse(BaseModel):
    filename: str
    chunk_added: int


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


class SourceItem(BaseModel):
    index: int
    text: str
    source: str
    page: int | None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in list(_loader.keys()):
        raise HTTPException(status_code=400, detail="File type not supported")

    content = await file.read()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(content)

        try:
            chunk_added = get_pipeline().ingest(
                file_path=tmp_path
            )
        except Exception as ex:
            raise HTTPException(status_code=500, detail=str(ex)) from ex
    return IngestResponse(filename=file.filename, chunk_added=chunk_added)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest)-> ChatResponse:
    try:
        pipeline = get_pipeline()
    except ValueError as ve:
        raise HTTPException(status_code=503, detail=str(ve)) from ve

    result = pipeline.query(question=request.question)
    return ChatResponse(
        **result
    )

@app.delete("/reset")
def reset_knowledge_base():
    try:
        pipeline = get_pipeline()
    except ValueError as ve:
        raise HTTPException(status_code=503, detail=str(ve)) from ve

    pipeline.reset()
    return {"status": "cleared"}



