from flask import Flask, request, jsonify, render_template_string
import requests
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# 🔥 STEP 1: Load Documents Automatically
def load_documents():
    documents = []

    if not os.path.exists("documents"):
        print("❌ documents folder not found.")
        return []

    for file in os.listdir("documents"):
        path = os.path.join("documents", file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

    return documents


# 🔥 STEP 2: Build Vectorstore (Auto)
def build_vectorstore():
    documents = load_documents()

    if not documents:
        print("❌ No documents found.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    print("✅ Vectorstore created successfully!")
    return vectorstore


# 🔥 Create vectorstore when app starts
vectorstore = build_vectorstore()

# 🔒 Coding-only filter
CODING_KEYWORDS = [
    "python", "java", "c", "c++", "javascript", "html", "css",
    "sql", "mysql", "postgresql", "mongodb", "flask", "spring",
    "react", "node", "api", "backend", "frontend",
    "function", "class", "object", "loop", "array", "string",
    "database", "query", "algorithm", "data structure",
    "error", "exception", "bug", "debug", "compile",
    "code", "program", "build", "develop", "implement"
]

def is_allowed(message: str) -> bool:
    msg = message.lower()
    return any(word in msg for word in CODING_KEYWORDS)

@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Coding RAG Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
            color: white;
            display: flex;
            justify-content: center;
        }

        .chat-container {
            width: 100%;
            max-width: 800px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            padding: 20px;
            font-size: 22px;
            font-weight: bold;
            border-bottom: 1px solid #333;
            background: #111;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 75%;
            padding: 12px 16px;
            border-radius: 15px;
            line-height: 1.5;
            animation: fadeIn 0.2s ease-in-out;
        }

        .user {
            align-self: flex-end;
            background: #2563eb;
            border-bottom-right-radius: 5px;
        }

        .bot {
            align-self: flex-start;
            background: #2a2a2a;
            border-bottom-left-radius: 5px;
        }

        .input-area {
            display: flex;
            padding: 15px;
            border-top: 1px solid #333;
            background: #111;
        }

        input {
            flex: 1;
            padding: 12px;
            border-radius: 10px;
            border: none;
            outline: none;
            background: #1f1f1f;
            color: white;
            font-size: 15px;
        }

        button {
            margin-left: 10px;
            padding: 12px 18px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            background: #2563eb;
            color: white;
            font-weight: bold;
            transition: 0.2s;
        }

        button:hover {
            background: #1d4ed8;
        }

        .typing {
            font-style: italic;
            opacity: 0.6;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(5px);}
            to {opacity: 1; transform: translateY(0);}
        }

        @media (max-width: 600px) {
            .message { max-width: 90%; }
        }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="header">💻 Coding RAG Assistant</div>
    <div class="messages" id="messages"></div>

    <div class="input-area">
        <input type="text" id="message" placeholder="Ask coding question..." onkeydown="handleKey(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>

function handleKey(e) {
    if (e.key === "Enter") {
        sendMessage();
    }
}

function addMessage(text, type) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", type);
    msgDiv.innerHTML = text.replace(/\\n/g, "<br>");
    document.getElementById("messages").appendChild(msgDiv);
    document.getElementById("messages").scrollTop =
        document.getElementById("messages").scrollHeight;
}

function sendMessage() {
    const input = document.getElementById("message");
    const msg = input.value.trim();
    if (!msg) return;

    addMessage(msg, "user");
    input.value = "";

    const typingDiv = document.createElement("div");
    typingDiv.classList.add("message", "bot", "typing");
    typingDiv.id = "typing";
    typingDiv.innerHTML = "Typing...";
    document.getElementById("messages").appendChild(typingDiv);

    fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: msg})
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("typing").remove();
        addMessage(data.reply, "bot");
    })
    .catch(() => {
        document.getElementById("typing").remove();
        addMessage("⚠️ Error connecting to server.", "bot");
    });
}

</script>

</body>
</html>
""")



@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    if not is_allowed(user_msg):
        return jsonify({
            "reply": "❌ This assistant only supports coding-related questions."
        })

    if vectorstore is None:
        return jsonify({
            "reply": "⚠️ No documents loaded. Please add documents inside 'documents' folder."
        })

    # 🔍 RAG retrieval
    docs = vectorstore.similarity_search(user_msg, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert coding assistant. Use ONLY the provided context to answer. If not in context, say you don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{user_msg}"
            }
        ]
    }

    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Error: {str(e)}"})


if __name__ == "__main__":
    app.run(debug=True, port=5002)
