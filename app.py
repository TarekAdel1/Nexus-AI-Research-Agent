import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# Set User-Agent headers for web scrapers and Wikipedia BEFORE loading tools
os.environ["USER_AGENT"] = "StreamlitResearchBot/1.0 (contact: test@example.com)"

import wikipedia
wikipedia.set_user_agent("StreamlitResearchBot/1.0 (contact: test@example.com)")

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Nexus | AI Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal UI styles
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        color: #31333F;
        font-size: 0.8rem;
        font-family: monospace;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Per-browser session isolation (UUID & Thread ID)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Persistent Checkpointer (cached per Streamlit session)
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

# Chat display history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I am your research agent with live web search and Wikipedia access. What would you like to explore today?",
        }
    ]

# Sidebar configuration
with st.sidebar:
    st.title("⚡ Settings")
    st.caption("Active configuration and session space.")

    st.markdown("**Active Model**")
    st.code("openai/gpt-oss-120b", language="text")

    st.divider()

    st.markdown("**Session Information**")
    st.markdown(f"<span class='badge'>Thread ID</span> `{st.session_state.thread_id[:8]}...`", unsafe_allow_html=True)
    st.caption("Each browser window operates on its own isolated memory thread.")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Session cleared! New memory thread started. What can I do for you?",
            }
        ]
        st.rerun()

# Tool setup with safety fallbacks
class SafeWikiWrapper(WikipediaAPIWrapper):
    def run(self, query: str) -> str:
        try:
            return super().run(query)
        except Exception as e:
            return f"Wikipedia error: {e}"

wiki_tool = WikipediaQueryRun(
    api_wrapper=SafeWikiWrapper(top_k_results=1, doc_content_chars_max=200),
    handle_tool_error=True,
)
search_tool = DuckDuckGoSearchRun(name="web_search", handle_tool_error=True)

tools = [search_tool, wiki_tool]

# App header
st.title("⚡ Nexus Research Agent")
st.caption("Powered by `openai/gpt-oss-120b` with persistent memory & live token streaming.")

# Render previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User interaction
if user_prompt := st.chat_input("Ask a question, search current news, or lookup facts..."):
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to begin chatting.")
        st.stop()

    # Append and render user input
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Initialize model locked to openai/gpt-oss-120b
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="openai/gpt-oss-120b",
        temperature=0.2,
        streaming=True,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are Nexus, a helpful, precise research assistant. "
            "Use Wikipedia for factual knowledge and encyclopedia queries. "
            "Use web_search for real-time information, current events, or internet lookup. "
            "Maintain conversational context from previous user turns."
        ),
        checkpointer=st.session_state.checkpointer,
    )

    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Stream agent execution
    with st.chat_message("assistant"):
        status_box = st.status("Agent thinking...", expanded=True)
        tools_used = set()

        def agent_token_generator():
            for token, metadata in agent.stream(
                {"messages": [("user", user_prompt)]},
                config=thread_config,
                stream_mode="messages",
            ):
                # Detect and report tool calls inside the status container
                if hasattr(token, "tool_call_chunks") and token.tool_call_chunks:
                    for chunk in token.tool_call_chunks:
                        tool_name = chunk.get("name")
                        if tool_name and tool_name not in tools_used:
                            tools_used.add(tool_name)
                            status_box.write(f"⚡ Calling tool: `{tool_name}`")
                            status_box.update(label=f"Querying {tool_name}...", state="running")

                # Yield text tokens directly to the outer chat bubble
                if metadata.get("langgraph_node") == "model" and token.content:
                    status_box.update(
                        label=f"Completed using: {', '.join(tools_used)}" if tools_used else "Direct answer",
                        state="complete",
                        expanded=False,
                    )
                    yield token.content

        # Stream directly into the chat message container
        full_response = st.write_stream(agent_token_generator())

        # Ensure status box closes if no tokens were streamed
        if not tools_used:
            status_box.update(label="Direct answer", state="complete", expanded=False)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
