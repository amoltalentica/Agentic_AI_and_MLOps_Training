"""
Agentic AI – FastAPI application wrapping all LangGraph agents.
Exposes REST endpoints so the agents can be used without interactive terminal sessions.
Designed for containerised / Kubernetes (Minikube) deployment.
"""

import os
import uuid
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic AI API",
    description=(
        "LangGraph-powered AI Agents exposed as REST endpoints. "
        "Agents: agent-bot, memory, react, drafter, rag."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session storage
# ---------------------------------------------------------------------------

memory_sessions: Dict[str, List[BaseMessage]] = {}
drafter_sessions: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ReactRequest(BaseModel):
    query: str


class ReactResponse(BaseModel):
    result: str


class DrafterRequest(BaseModel):
    instruction: str
    session_id: Optional[str] = None


class DrafterResponse(BaseModel):
    response: str
    document_content: str
    session_id: str
    is_saved: bool = False


class RAGRequest(BaseModel):
    query: str


class RAGResponse(BaseModel):
    answer: str


# ===========================================================================
# AGENT BOT  –  stateless single-turn chatbot
# ===========================================================================


class _AgentBotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


_agent_bot_graph = None


def _get_agent_bot():
    global _agent_bot_graph
    if _agent_bot_graph is None:
        llm = ChatOpenAI(model="gpt-4o")

        def _process(state: _AgentBotState) -> _AgentBotState:
            return {"messages": [llm.invoke(list(state["messages"]))]}

        g = StateGraph(_AgentBotState)
        g.add_node("process", _process)
        g.add_edge(START, "process")
        g.add_edge("process", END)
        _agent_bot_graph = g.compile()
    return _agent_bot_graph


# ===========================================================================
# REACT AGENT  –  math tool agent (add / subtract / multiply)
# ===========================================================================


class _ReactState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


_react_tools = [add, subtract, multiply]
_react_graph = None


def _get_react_agent():
    global _react_graph
    if _react_graph is None:
        model = ChatOpenAI(model="gpt-4o").bind_tools(_react_tools)

        def _model_call(state: _ReactState) -> _ReactState:
            sp = SystemMessage(
                content="You are an AI assistant. Answer queries to the best of your ability."
            )
            return {"messages": [model.invoke([sp] + list(state["messages"]))]}

        def _should_continue(state: _ReactState) -> str:
            last = state["messages"][-1]
            return "continue" if getattr(last, "tool_calls", None) else "end"

        g = StateGraph(_ReactState)
        g.add_node("agent", _model_call)
        g.add_node("tools", ToolNode(tools=_react_tools))
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent", _should_continue, {"continue": "tools", "end": END}
        )
        g.add_edge("tools", "agent")
        _react_graph = g.compile()
    return _react_graph


# ===========================================================================
# DRAFTER AGENT  –  session-based document drafting with update / save tools
# Handled with direct LLM calls (no graph) to keep session state simple.
# ===========================================================================

# Module-level tool definitions – bodies are irrelevant because tool calls
# are handled manually in the endpoint; these objects only provide the JSON
# schemas that the LLM sees.


@tool
def update_document(content: str) -> str:
    """Update the working document with the provided full content."""
    return content


@tool
def save_document(filename: str) -> str:
    """Save the current document to a file with the given filename."""
    return filename


_drafter_tools = [update_document, save_document]
_drafter_llm = None


def _get_drafter_llm():
    global _drafter_llm
    if _drafter_llm is None:
        _drafter_llm = ChatOpenAI(model="gpt-4o").bind_tools(_drafter_tools)
    return _drafter_llm


# ===========================================================================
# RAG AGENT  –  PDF Q&A over Stock Market Performance 2024 document
# Lazily initialised; returns None when the PDF is not mounted.
# ===========================================================================

_rag_graph = None


def _get_rag_agent():
    global _rag_graph
    if _rag_graph is not None:
        return _rag_graph

    pdf_path = os.environ.get(
        "PDF_PATH", "/app/Agents/Stock_Market_Performance_2024.pdf"
    )
    if not os.path.exists(pdf_path):
        return None

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        pages = PyPDFLoader(pdf_path).load()
        docs = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(pages)

        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "/app/chroma_db")
        os.makedirs(persist_dir, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="stock_market",
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        @tool
        def retriever_tool(query: str) -> str:
            """Search the Stock Market Performance 2024 document for relevant information."""
            result_docs = retriever.invoke(query)
            if not result_docs:
                return "No relevant information found in the document."
            return "\n\n".join(
                f"Chunk {i + 1}:\n{d.page_content}"
                for i, d in enumerate(result_docs)
            )

        rag_tools = [retriever_tool]
        rag_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(rag_tools)
        rag_tools_dict = {t.name: t for t in rag_tools}

        rag_system_prompt = (
            "You are an AI assistant answering questions about Stock Market Performance "
            "in 2024 based on an uploaded PDF document. Use the retriever_tool to find "
            "relevant information. Always cite specific chunks from the document."
        )

        class _RAGState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        def _call_llm(state: _RAGState) -> _RAGState:
            msgs = [SystemMessage(content=rag_system_prompt)] + list(state["messages"])
            return {"messages": [rag_llm.invoke(msgs)]}

        def _take_action(state: _RAGState) -> _RAGState:
            results = []
            for tc in state["messages"][-1].tool_calls:
                if tc["name"] in rag_tools_dict:
                    result = rag_tools_dict[tc["name"]].invoke(
                        tc["args"].get("query", "")
                    )
                else:
                    result = "Requested tool not found."
                results.append(
                    ToolMessage(
                        tool_call_id=tc["id"], name=tc["name"], content=str(result)
                    )
                )
            return {"messages": results}

        def _rag_should_continue(state: _RAGState) -> bool:
            last = state["messages"][-1]
            return bool(getattr(last, "tool_calls", None))

        g = StateGraph(_RAGState)
        g.add_node("llm", _call_llm)
        g.add_node("action", _take_action)
        g.add_conditional_edges(
            "llm", _rag_should_continue, {True: "action", False: END}
        )
        g.add_edge("action", "llm")
        g.set_entry_point("llm")
        _rag_graph = g.compile()
        return _rag_graph

    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Failed to initialise RAG agent: {exc}")
        return None


# ===========================================================================
# API Endpoints
# ===========================================================================


@app.get("/health", tags=["Health"])
async def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "service": "Agentic AI API"}


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Agentic AI API",
        "version": "1.0.0",
        "agents": ["agent-bot", "memory", "react", "drafter", "rag"],
        "interactive_docs": "/docs",
    }


# ---- Agent Bot ---------------------------------------------------------------


@app.post("/api/agent-bot/chat", response_model=ChatResponse, tags=["Agent Bot"])
async def agent_bot_chat(request: ChatRequest):
    """
    Single-turn stateless chatbot.

    Send a message and receive a response. No conversation history is retained.
    """
    try:
        graph = _get_agent_bot()
        result = graph.invoke({"messages": [HumanMessage(content=request.message)]})
        return ChatResponse(
            response=result["messages"][-1].content,
            session_id=str(uuid.uuid4()),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---- Memory Agent ------------------------------------------------------------


@app.post("/api/memory/chat", response_model=ChatResponse, tags=["Memory Agent"])
async def memory_chat(request: ChatRequest):
    """
    Multi-turn chatbot with persistent conversation memory.

    Omit `session_id` on the first call – you'll receive one in the response.
    Pass the same `session_id` on subsequent calls to continue the conversation.
    """
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in memory_sessions:
        memory_sessions[session_id] = []

    memory_sessions[session_id].append(HumanMessage(content=request.message))

    try:
        llm = ChatOpenAI(model="gpt-4o")
        response = llm.invoke(memory_sessions[session_id])
        memory_sessions[session_id].append(AIMessage(content=response.content))
        return ChatResponse(response=response.content, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/api/memory/sessions/{session_id}",
    tags=["Memory Agent"],
)
async def get_session_history(session_id: str):
    """Return full conversation history for a memory session."""
    if session_id not in memory_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    history = [
        {
            "role": "human" if isinstance(m, HumanMessage) else "ai",
            "content": m.content,
        }
        for m in memory_sessions[session_id]
    ]
    return {"session_id": session_id, "history": history}


@app.delete("/api/memory/sessions/{session_id}", tags=["Memory Agent"])
async def clear_memory_session(session_id: str):
    """Clear the conversation history for a session."""
    memory_sessions.pop(session_id, None)
    return {"message": f"Session {session_id} cleared."}


# ---- ReAct Agent -------------------------------------------------------------


@app.post("/api/react/solve", response_model=ReactResponse, tags=["ReAct Agent"])
async def react_solve(request: ReactRequest):
    """
    ReAct math agent.

    Supports add, subtract, and multiply operations.
    Example query: *"Add 40 and 12, then multiply the result by 6."*
    """
    try:
        graph = _get_react_agent()
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.query)]}
        )
        return ReactResponse(result=result["messages"][-1].content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---- Drafter Agent -----------------------------------------------------------


@app.post("/api/drafter/chat", response_model=DrafterResponse, tags=["Drafter Agent"])
async def drafter_chat(request: DrafterRequest):
    """
    Document-drafting agent.

    - Provide an instruction (e.g. *"Write an introduction about AI"*).
    - The agent will update the in-session document and return the new content.
    - Say *"save the document as report.txt"* to finalise.

    Omit `session_id` on the first call to start a new drafting session.
    """
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in drafter_sessions:
        drafter_sessions[session_id] = {"messages": [], "document_content": ""}

    session = drafter_sessions[session_id]
    doc_content: str = session["document_content"]
    model = _get_drafter_llm()

    system_prompt = f"""You are Drafter, a helpful writing assistant. Help the user create and refine documents.

Rules:
- Use the `update_document` tool to create or update the document.  Always pass the FULL document content.
- Use the `save_document` tool when the user wants to save and finish.
- After using a tool, summarise what changed in plain language.

Current document content:
{doc_content if doc_content else "(empty – no content yet)"}"""

    user_msg = HumanMessage(content=request.instruction)
    all_messages = (
        [SystemMessage(content=system_prompt)]
        + session["messages"]
        + [user_msg]
    )

    try:
        response = model.invoke(all_messages)
        is_saved = False

        if getattr(response, "tool_calls", None):
            tool_results: List[ToolMessage] = []
            for tc in response.tool_calls:
                if tc["name"] == "update_document":
                    doc_content = tc["args"].get("content", doc_content)
                    result_text = (
                        f"Document updated successfully.\n\nNew content:\n{doc_content}"
                    )
                elif tc["name"] == "save_document":
                    filename = tc["args"].get("filename", "document.txt")
                    result_text = f"Document saved to '{filename}'."
                    is_saved = True
                else:
                    result_text = "Unknown tool requested."

                tool_results.append(
                    ToolMessage(
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        content=result_text,
                    )
                )

            # Ask the LLM to summarise after tool execution
            final_msgs = all_messages + [response] + tool_results
            final_response = model.invoke(final_msgs)
            ai_content = final_response.content or tool_results[-1].content

            session["messages"] = (
                session["messages"] + [user_msg, response] + tool_results + [final_response]
            )
        else:
            ai_content = response.content
            session["messages"] = session["messages"] + [user_msg, response]

        session["document_content"] = doc_content

        return DrafterResponse(
            response=ai_content,
            document_content=doc_content,
            session_id=session_id,
            is_saved=is_saved,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/drafter/sessions/{session_id}", tags=["Drafter Agent"])
async def clear_drafter_session(session_id: str):
    """Clear a drafting session (document content + history)."""
    drafter_sessions.pop(session_id, None)
    return {"message": f"Drafter session {session_id} cleared."}


# ---- RAG Agent ---------------------------------------------------------------


@app.post("/api/rag/query", response_model=RAGResponse, tags=["RAG Agent"])
async def rag_query(request: RAGRequest):
    """
    Retrieval-Augmented Generation agent.

    Answers questions about the *Stock Market Performance 2024* PDF.

    **Requirements:** Mount the PDF at the path specified by the `PDF_PATH`
    environment variable (default: `/app/data/Stock_Market_Performance_2024.pdf`).
    The agent initialises ChromaDB lazily on the first query.
    """
    graph = _get_rag_agent()
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG agent is not available. "
                "Mount the PDF at the location specified by PDF_PATH "
                "(default: /app/data/Stock_Market_Performance_2024.pdf) and restart the pod."
            ),
        )
    try:
        result = graph.invoke({"messages": [HumanMessage(content=request.query)]})
        return RAGResponse(answer=result["messages"][-1].content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entrypoint (for local testing without Docker)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
