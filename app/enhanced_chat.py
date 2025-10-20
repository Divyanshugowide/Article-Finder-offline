"""
Enhanced Chat API with Streaming, RBAC, and ChatGPT-like Experience
"""
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.retrieval import Retriever

# =====================================
# Configuration
# =====================================
class ChatConfig:
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "qwen2.5:7b"  # Better conversational model
    FALLBACK_MODEL = "phi3"
    MAX_HISTORY = 15
    MAX_CONTEXT_LENGTH = 8000
    REQUEST_TIMEOUT = 300
    STREAM_DELAY = 0.02  # Delay between chunks for typing effect

# =====================================
# Pydantic Models
# =====================================
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_role: str = "staff"  # RBAC role
    use_documents: bool = True
    model: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    sources: Optional[List[Dict[str, Any]]] = None
    model_used: str
    timestamp: str

# =====================================
# Enhanced Chat Application
# =====================================
app = FastAPI(title="Article Finder - ChatGPT-like Experience", version="2.0")

# Initialize retriever
try:
    # Try both possible FAISS file names
    faiss_path = "data/idx/faiss.index"
    if not Path(faiss_path).exists():
        faiss_path = "data/idx/mE5.faiss"
    
    retriever = Retriever(
        bm25_path="data/idx/bm25.pkl",
        faiss_path=faiss_path, 
        meta_path="data/idx/meta.json",
        alpha=0.45
    )
    print(f"✅ Document retriever initialized with {faiss_path}")
except Exception as e:
    print(f"⚠️ Warning: Retriever failed to initialize: {e}")
    retriever = None

# Conversation memory with enhanced tracking
conversations: Dict[str, Dict[str, Any]] = {}

# =====================================
# Utility Functions
# =====================================
def is_search_query(text: str) -> bool:
    """Enhanced search intent detection"""
    search_indicators = [
        # Direct search terms
        "find", "search", "look for", "show me", "where is", "locate",
        # Document-specific terms
        "article", "section", "clause", "page", "document", "pdf", "policy",
        # Query patterns
        "what does", "what is", "how does", "according to", "mentioned in",
        "requirements for", "process of", "rules about", "information on"
    ]
    
    text_lower = text.lower().strip()
    return any(indicator in text_lower for indicator in search_indicators)

async def get_available_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ChatConfig.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
    except Exception:
        pass
    return [ChatConfig.DEFAULT_MODEL, ChatConfig.FALLBACK_MODEL]

def filter_sources_by_role(sources: List[Dict], user_role: str) -> List[Dict]:
    """Enhanced RBAC filtering for document sources"""
    if not sources:
        return []
    
    # Define role hierarchy
    role_hierarchy = {
        "admin": ["admin", "legal", "staff", "public"],
        "legal": ["legal", "staff", "public"],  
        "staff": ["staff", "public"],
        "public": ["public"]
    }
    
    allowed_roles = role_hierarchy.get(user_role.lower(), ["public"])
    
    filtered_sources = []
    for source in sources:
        source_roles = source.get("roles", ["public"])
        if isinstance(source_roles, str):
            source_roles = [source_roles]
            
        # Check if user has access to at least one of the source's roles
        if any(role.lower() in [r.lower() for r in allowed_roles] for role in source_roles):
            filtered_sources.append(source)
    
    return filtered_sources

async def stream_ollama_response(
    messages: List[Dict[str, str]], 
    model: str = ChatConfig.DEFAULT_MODEL
) -> AsyncGenerator[str, None]:
    """Stream response from Ollama with typing effect"""
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        async with httpx.AsyncClient(timeout=ChatConfig.REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST", 
                f"{ChatConfig.OLLAMA_BASE_URL}/api/chat", 
                json=payload
            ) as response:
                
                if response.status_code != 200:
                    yield f"data: {json.dumps({'error': 'Model connection failed'})}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk_data = json.loads(line)
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                content = chunk_data["message"]["content"]
                                if content:
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                                    await asyncio.sleep(ChatConfig.STREAM_DELAY)
                            
                            if chunk_data.get("done"):
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                break
                                
                        except json.JSONDecodeError:
                            continue
                            
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# =====================================
# API Endpoints
# =====================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models": await get_available_models()}

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Main streaming chat endpoint with RBAC and document integration"""
    
    # Initialize conversation
    conv_id = request.conversation_id or str(uuid.uuid4())
    if conv_id not in conversations:
        conversations[conv_id] = {
            "messages": [],
            "user_role": request.user_role,
            "created_at": datetime.now().isoformat()
        }
    
    # Add user message
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    }
    conversations[conv_id]["messages"].append(user_message)
    
    # Keep conversation history manageable
    if len(conversations[conv_id]["messages"]) > ChatConfig.MAX_HISTORY:
        conversations[conv_id]["messages"] = conversations[conv_id]["messages"][-ChatConfig.MAX_HISTORY:]
    
    # Determine if document search is needed
    needs_search = request.use_documents and is_search_query(request.message)
    
    # Build context from documents if needed
    context_info = ""
    sources = []
    
    if needs_search and retriever:
        try:
            search_results = retriever.search(
                request.message, 
                roles=[request.user_role], 
                topk=5
            )
            
            # Filter sources by user role (RBAC)
            filtered_results = filter_sources_by_role(
                search_results.get("results", []), 
                request.user_role
            )
            
            if filtered_results:
                sources = filtered_results[:3]  # Top 3 most relevant
                context_parts = []
                for source in sources:
                    context_parts.append(
                        f"[Document: {source.get('doc_id', 'Unknown')} | "
                        f"Pages {source.get('page_start', '?')}-{source.get('page_end', '?')}]\n"
                        f"{source.get('excerpt', '')}"
                    )
                
                context_info = "\n\n".join(context_parts)
                
        except Exception as e:
            print(f"Document search failed: {e}")
    
    # Build messages for Ollama
    system_message = {
        "role": "system",
        "content": (
            "You are Crystal, a helpful AI assistant for the Article Finder system. "
            "You have access to a database of PDF documents with role-based access control. "
            "Be conversational, friendly, and helpful. When you have document context, "
            "use it to provide accurate, well-sourced answers. If you don't have relevant "
            "context, engage naturally in conversation. Always be concise but thorough."
        )
    }
    
    ollama_messages = [system_message]
    
    # Add conversation history (last few messages for context)
    recent_messages = conversations[conv_id]["messages"][-6:]  # Last 6 messages
    ollama_messages.extend([
        {"role": msg["role"], "content": msg["content"]} 
        for msg in recent_messages if msg["role"] in ["user", "assistant"]
    ])
    
    # Add document context if available
    if context_info:
        context_message = {
            "role": "system",
            "content": f"Relevant document context for this query:\n\n{context_info}"
        }
        ollama_messages.append(context_message)
    
    # Stream response
    async def generate_response():
        try:
            # Send conversation ID first
            yield f"data: {json.dumps({'conversation_id': conv_id, 'sources': sources})}\n\n"
            
            # Stream the actual response
            model_to_use = request.model or ChatConfig.DEFAULT_MODEL
            full_response = ""
            
            async for chunk in stream_ollama_response(ollama_messages, model_to_use):
                yield chunk
                
                # Extract content for storage
                if chunk.startswith("data: "):
                    try:
                        chunk_data = json.loads(chunk[6:])
                        if "content" in chunk_data:
                            full_response += chunk_data["content"]
                    except:
                        pass
            
            # Store assistant response
            if full_response:
                assistant_message = {
                    "role": "assistant",
                    "content": full_response.strip(),
                    "timestamp": datetime.now().isoformat(),
                    "sources": sources
                }
                conversations[conv_id]["messages"].append(assistant_message)
                
        except Exception as e:
            error_msg = f"data: {json.dumps({'error': f'Chat failed: {str(e)}'})}\n\n"
            yield error_msg
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")

@app.get("/chat/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get conversation history"""
    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conv_id,
        "messages": conversations[conv_id]["messages"],
        "user_role": conversations[conv_id].get("user_role", "staff")
    }

@app.delete("/chat/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation"""
    if conv_id in conversations:
        del conversations[conv_id]
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/models")
async def list_models():
    """Get available models"""
    return {"models": await get_available_models()}

# Mount static files and serve the enhanced UI
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the Optimized ChatGPT-like interface with better performance"""
    ui_file = static_dir / "optimized_chat_ui.html"
    if ui_file.exists():
        return HTMLResponse(content=ui_file.read_text(encoding='utf-8'))
    else:
        # Fallback to premium UI
        fallback_file = static_dir / "premium_chat_ui.html"
        if fallback_file.exists():
            return HTMLResponse(content=fallback_file.read_text(encoding='utf-8'))
        # Final fallback
        final_fallback = static_dir / "chatgpt_ui.html"
        if final_fallback.exists():
            return HTMLResponse(content=final_fallback.read_text(encoding='utf-8'))
        return HTMLResponse(
            content="<h1>UI not found</h1><p>Please ensure optimized_chat_ui.html exists in the static directory.</p>",
            status_code=404
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
