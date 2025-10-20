# 🤖 Enhanced Article Finder Chatbot

A ChatGPT-like conversational AI that runs entirely locally using Ollama, with document search capabilities and role-based access control.

## ✨ Features

- **ChatGPT-like Experience**: Streaming responses with typing effects
- **Local AI**: Powered by Ollama (Qwen2.5:7b model recommended)
- **Document Search**: Intelligent search through your PDFs with citations
- **Role-Based Access Control (RBAC)**: Different access levels (Public, Staff, Legal, Admin)
- **Conversation Memory**: Maintains context across messages
- **Modern UI**: Clean, responsive interface similar to ChatGPT
- **Real-time Streaming**: See responses as they're generated
- **Source Citations**: Shows document sources when answering from PDFs

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have:
- Python 3.8+ installed
- Ollama installed and running
- Your PDF documents processed (see main README.md)

### 2. Install Better Model

For improved conversational experience, install Qwen2.5:
```bash
ollama pull qwen2.5:7b
```

Alternative models you can try:
```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:medium
```

### 3. Start the Enhanced Chatbot

Simply run:
```bash
python start_chatbot.py
```

Or manually:
```bash
python -m app.enhanced_chat
```

### 4. Access Your Chatbot

Open your browser and go to:
**http://127.0.0.1:8001**

## 🎯 Usage Examples

### General Conversation
```
User: Hello! How are you?
Crystal: Hi! I'm doing great, thank you for asking! I'm Crystal, your AI assistant for the Article Finder system. I'm here to help you search through documents, answer questions, or just have a friendly chat. What would you like to know today?
```

### Document Search
```
User: What are the requirements for equipment maintenance?
Crystal: Based on the maintenance policy document, here are the key equipment maintenance requirements:

[Document response with citations showing specific pages and sources]
```

### Role-Based Access
- **Public**: Access to general documents only
- **Staff**: Access to staff-level documents + public
- **Legal**: Access to legal documents + staff + public  
- **Admin**: Access to all documents including restricted

## 🛠️ API Endpoints

The enhanced chatbot provides these endpoints:

- `GET /` - Main chat interface
- `POST /chat/stream` - Streaming chat endpoint
- `GET /chat/conversations/{id}` - Get conversation history
- `DELETE /chat/conversations/{id}` - Delete conversation
- `GET /models` - List available Ollama models
- `GET /health` - Health check

## 🎨 UI Features

### Typing Effects
- Real-time streaming response display
- ChatGPT-like typing animation
- Smooth message transitions

### Loading States
- Spinning loader during processing
- Typing indicator while AI thinks
- Progressive response building

### Interactive Elements
- Auto-resizing input textarea
- Role selector for RBAC
- Responsive design for mobile/desktop
- Modern scrollbars and animations

## 🔧 Configuration

### Model Selection
Edit `app/enhanced_chat.py` to change the default model:
```python
class ChatConfig:
    DEFAULT_MODEL = "qwen2.5:7b"  # Change this
    FALLBACK_MODEL = "phi3"
```

### RBAC Roles
Modify role hierarchy in the `filter_sources_by_role` function:
```python
role_hierarchy = {
    "admin": ["admin", "legal", "staff", "public"],
    "legal": ["legal", "staff", "public"],  
    "staff": ["staff", "public"],
    "public": ["public"]
}
```

### Conversation Settings
Adjust memory and context in `ChatConfig`:
```python
class ChatConfig:
    MAX_HISTORY = 15  # Number of messages to remember
    MAX_CONTEXT_LENGTH = 8000  # Token limit for context
    STREAM_DELAY = 0.02  # Typing effect speed
```

## 📱 Mobile Support

The interface is fully responsive and works great on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablets (iPad, Android tablets)
- Mobile phones (iOS Safari, Android Chrome)

## 🔍 Search Intelligence

The chatbot automatically detects when you need document search based on:
- Keywords like "find", "search", "what is", "requirements"
- Document-specific terms like "article", "section", "policy"
- Question patterns that typically need citations

## 🔐 Security Features

- **RBAC Integration**: Documents are filtered by user role
- **Local Processing**: All AI processing happens locally
- **No Data Leakage**: Conversations stay on your machine
- **Session Isolation**: Each conversation is independent

## 🚨 Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull required model
ollama pull qwen2.5:7b
```

### Port Already in Use
Change the port in `start_chatbot.py`:
```python
uvicorn.run(app, host="127.0.0.1", port=8002)  # Use different port
```

### Documents Not Loading
Ensure your document indices are built:
```bash
python scripts/01_chunk_pdfs.py
python scripts/02_build_bm25.py  
python scripts/03_build_faiss.py
```

## 🔄 Comparing with Original

| Feature | Original | Enhanced |
|---------|----------|----------|
| UI Style | Basic HTML | ChatGPT-like modern UI |
| Responses | Static | Real-time streaming |
| Model | phi3 | Qwen2.5:7b (better) |
| Conversation | Limited | Full memory |
| RBAC | Basic | Advanced filtering |
| Mobile | Poor | Fully responsive |
| Typing Effects | None | ChatGPT-style |
| Error Handling | Basic | Comprehensive |

## 🎉 Next Steps

1. **Start chatting**: Launch the bot and try various queries
2. **Test RBAC**: Switch between different user roles
3. **Upload more PDFs**: Add documents to `data/raw_pdfs/` and rebuild indices
4. **Customize appearance**: Modify CSS in `static/chatgpt_ui.html`
5. **Try different models**: Experiment with various Ollama models

## 📞 Support

If you encounter issues:
1. Check that Ollama is running: `ollama list`
2. Verify document indices exist in `data/idx/`
3. Check console logs for error messages
4. Try restarting both Ollama and the chatbot

---

**Enjoy your new ChatGPT-like local AI assistant! 🚀**