#!/usr/bin/env python3
"""
Enhanced Article Finder Chatbot - ChatGPT-like Experience
Run this script to start your improved local chatbot with Ollama integration
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.enhanced_chat import app

if __name__ == "__main__":
    print("🚀 Starting Article Finder - ChatGPT Experience...")
    print("📋 Features enabled:")
    print("   • Streaming responses with typing effects")
    print("   • Role-based access control (RBAC)")
    print("   • Document search integration")
    print("   • Conversation memory")
    print("   • Qwen2.5:7b model for better conversations")
    print()
    print("🌐 Access your chatbot at: http://127.0.0.1:8001")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="info"
    )