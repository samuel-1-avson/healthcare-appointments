# src/api/routes/rag_routes.py
"""
RAG API Routes
==============
FastAPI endpoints for RAG-based policy Q&A.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field

# RAG imports
from src.llm.rag import (
    load_policy_documents,
    create_vector_store,
    VectorStoreManager,
    get_vector_store,
    create_retriever
)
from src.llm.rag.chains import (
    RAGChain,
    ConversationalRAGChain,
    CitationRAGChain,
    create_rag_chain
)
from src.llm.rag.evaluation import RAGEvaluator, create_healthcare_golden_set
from ..auth import get_current_user, User
from ..cache import cache_response


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


# ==================== Schemas ====================

class PolicyQuestion(BaseModel):
    """Policy question request."""
    question: str = Field(description="Question about policies")
    session_id: Optional[str] = Field(default=None, description="Session for conversation continuity")
    include_sources: bool = Field(default=False, description="Include source documents")
    use_citations: bool = Field(default=False, description="Include citations in answer")


class PolicyAnswer(BaseModel):
    """Policy answer response."""
    answer: str = Field(description="Generated answer")
    question: str = Field(description="Original question")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Source documents")
    citations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Citation references")
    session_id: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUpload(BaseModel):
    """Document upload request."""
    content: str = Field(description="Document content")
    filename: str = Field(description="Filename for reference")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class IndexStatus(BaseModel):
    """Vector store index status."""
    initialized: bool
    document_count: int
    chunk_count: int
    last_updated: Optional[str]
    embeddings_model: str


# ==================== State Management ====================

# Global state (in production, use proper state management)
_rag_chain: Optional[RAGChain] = None
_conversational_rag: Optional[ConversationalRAGChain] = None
_citation_rag: Optional[CitationRAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get or create RAG chain."""
    global _rag_chain
    
    if _rag_chain is None:
        vector_store = get_vector_store()
        
        # Check if initialized
        if vector_store.vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized. Please index documents first."
            )
        
        _rag_chain = RAGChain(vector_store)
    
    return _rag_chain


def get_conversational_rag() -> ConversationalRAGChain:
    """Get or create conversational RAG."""
    global _conversational_rag
    
    if _conversational_rag is None:
        vector_store = get_vector_store()
        
        if vector_store.vector_store is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        _conversational_rag = ConversationalRAGChain(vector_store)
    
    return _conversational_rag


# ==================== Endpoints ====================

@router.post(
    "/ask",
    response_model=PolicyAnswer,
    summary="Ask Policy Question",
    description="Ask a question about healthcare appointment policies"
)
async def ask_policy_question(
    request: PolicyQuestion,
    user: User = Depends(get_current_user)
) -> PolicyAnswer:
    """
    Ask a question about appointment policies.
    
    The system will:
    1. Search the policy documents
    2. Retrieve relevant sections
    3. Generate an answer based on the policies
    
    Use session_id for follow-up questions that need context.
    """
    try:
        if request.session_id:
            # Use conversational RAG
            rag = get_conversational_rag()
            result = rag.ask(
                session_id=request.session_id,
                question=request.question,
                return_sources=request.include_sources
            )
        elif request.use_citations:
            # Use citation RAG
            vector_store = get_vector_store()
            citation_rag = CitationRAGChain(vector_store)
            result = citation_rag.ask(request.question)
        else:
            # Use basic RAG
            rag = get_rag_chain()
            result = rag.ask(
                question=request.question,
                return_sources=request.include_sources
            )
        
        return PolicyAnswer(
            answer=result["answer"],
            question=request.question,
            sources=result.get("sources"),
            citations=result.get("citations"),
            session_id=request.session_id,
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/ask/batch",
    summary="Batch Policy Questions",
    description="Ask multiple policy questions at once"
)
async def ask_batch_questions(
    questions: List[str] = Body(..., description="List of questions"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Ask multiple questions and get all answers."""
    rag = get_rag_chain()
    
    results = []
    for question in questions:
        try:
            result = rag.ask(question)
            results.append({
                "question": question,
                "answer": result["answer"],
                "success": True
            })
        except Exception as e:
            results.append({
                "question": question,
                "answer": None,
                "success": False,
                "error": str(e)
            })
    
    return {
        "results": results,
        "total": len(questions),
        "successful": sum(1 for r in results if r["success"])
    }


@router.get(
    "/search",
    summary="Search Policies",
    description="Search policy documents without generating an answer"
)
@cache_response(expire=3600)
async def search_policies(
    query: str = Query(..., description="Search query"),
    k: int = Query(default=5, ge=1, le=20, description="Number of results"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search policy documents.
    
    Returns relevant document chunks without generating an answer.
    Useful for exploring what's in the policy database.
    """
    vector_store = get_vector_store()
    
    if vector_store.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    results = vector_store.search_with_scores(query, k=k)
    
    return {
        "query": query,
        "results": [
            {
                "content": doc.page_content,
                "score": float(score),
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", "")
            }
            for doc, score in results
        ],
        "count": len(results)
    }


# ==================== Index Management ====================

@router.post(
    "/index/create",
    summary="Create Index",
    description="Create vector index from policy documents"
)
async def create_index(
    background_tasks: BackgroundTasks,
    documents_path: str = Query(default="data/documents", description="Path to documents"),
    chunk_size: int = Query(default=1000, ge=100, le=4000),
    chunk_overlap: int = Query(default=200, ge=0, le=500),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create or rebuild the vector index.
    
    This loads all documents from the specified path,
    chunks them, and creates embeddings.
    """
    try:
        # Load documents
        docs = load_policy_documents(documents_path)
        
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found")
        
        # Create vector store
        manager = get_vector_store()
        manager.create_from_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Save index
        manager.save("default")
        
        # Reset cached chains
        global _rag_chain, _conversational_rag
        _rag_chain = None
        _conversational_rag = None
        
        return {
            "status": "success",
            "documents_loaded": len(docs),
            "chunk_count": manager._metadata.get("chunk_count", 0),
            "message": "Index created successfully"
        }
        
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/index/load",
    summary="Load Index",
    description="Load existing vector index from disk"
)
async def load_index(
    name: str = Query(default="default", description="Index name"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Load a previously saved index."""
    try:
        manager = get_vector_store()
        manager.load(name)
        
        # Reset cached chains
        global _rag_chain, _conversational_rag
        _rag_chain = None
        _conversational_rag = None
        
        return {
            "status": "success",
            "index_name": name,
            "metadata": manager._metadata
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index '{name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/index/status",
    response_model=IndexStatus,
    summary="Index Status",
    description="Get current index status"
)
async def get_index_status(
    user: User = Depends(get_current_user)
) -> IndexStatus:
    """Get the status of the vector index."""
    manager = get_vector_store()
    stats = manager.get_stats()
    metadata = stats.get("metadata", {})
    
    return IndexStatus(
        initialized=stats["initialized"],
        document_count=metadata.get("document_count", 0),
        chunk_count=metadata.get("chunk_count", 0),
        last_updated=metadata.get("created_at"),
        embeddings_model=stats.get("embeddings", {}).get("model_info", {}).get("model", "unknown")
    )


@router.post(
    "/index/add-document",
    summary="Add Document",
    description="Add a new document to the index"
)
async def add_document(
    document: DocumentUpload,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Add a new document to the existing index."""
    from langchain_core.documents import Document
    
    manager = get_vector_store()
    
    if manager.vector_store is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    # Create document
    doc = Document(
        page_content=document.content,
        metadata={
            "source": document.filename,
            "filename": document.filename,
            **(document.metadata or {})
        }
    )
    
    # Add to index
    ids = manager.add_documents([doc])
    
    # Save updated index
    manager.save("default")
    
    return {
        "status": "success",
        "document_id": ids[0] if ids else None,
        "chunks_added": len(ids)
    }


# ==================== Evaluation ====================

@router.post(
    "/evaluate",
    summary="Evaluate RAG",
    description="Run evaluation on RAG pipeline"
)
async def evaluate_rag(
    use_golden_set: bool = Query(default=True, description="Use predefined golden set"),
    questions: Optional[List[str]] = Body(default=None, description="Custom questions"),
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Evaluate the RAG pipeline quality.
    
    Uses either the golden evaluation set or custom questions.
    """
    evaluator = RAGEvaluator()
    rag = get_rag_chain()
    
    if use_golden_set:
        golden = create_healthcare_golden_set()
        eval_questions, ground_truths = golden.to_eval_format()
    else:
        if not questions:
            raise HTTPException(status_code=400, detail="Provide questions or use golden set")
        eval_questions = questions
        ground_truths = None
    
    # Run questions through RAG
    evaluator.add_samples_from_chain(
        rag,
        eval_questions,
        ground_truths
    )
    
    # Evaluate
    results = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(f"evals/rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    return results


# ==================== Sessions ====================

@router.post(
    "/sessions",
    summary="Create Session",
    description="Create a new conversation session for RAG"
)
async def create_session(
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Create a new conversation session."""
    rag = get_conversational_rag()
    session_id = rag.create_session()
    
    return {"session_id": session_id}


@router.get(
    "/sessions/{session_id}/history",
    summary="Get Session History",
    description="Get conversation history for a RAG session"
)
async def get_session_history(
    session_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get conversation history."""
    rag = get_conversational_rag()
    
    try:
        history = rag.get_history(session_id)
        return {
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete(
    "/sessions/{session_id}",
    summary="Clear Session",
    description="Clear conversation history for a session"
)
async def clear_session(
    session_id: str,
    user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """Clear session history."""
    rag = get_conversational_rag()
    rag.clear_history(session_id)
    
    return {"status": "cleared", "session_id": session_id}