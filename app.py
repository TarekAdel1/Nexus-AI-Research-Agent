```python
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

# User-Agent for web scraping / Wikipedia
os.environ["USER_AGENT"] = (
    "StreamlitResearchBot/1.0 (contact: your-email@example.com)"
)

import wikipedia

wikipedia.set_user_agent(
    "StreamlitResearchBot/1.0 (contact: your-email@example.com)"
)

# ---------------------------------------------------------
# LangChain / LangGraph
# ---------------------------------------------------------

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Nexus | AI Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Get Groq API key from Streamlit Secrets
# ---------------------------------------------------------

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to your Streamlit Cloud Secrets."
    )
    st.stop()


# ---------------------------------------------------------
# Session / Memory
# ---------------------------------------------------------

# Unique conversation/thread ID for each browser session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# LangGraph checkpointer
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()


# Visible chat history for Streamlit UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I am Nexus, your AI research agent. "
                "I can search the web and Wikipedia to help you "
                "research questions and find current information."
            ),
        }
    ]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("⚡ Settings")

    st.caption(
        "Active configuration and conversation session."
    )

    st.markdown("**Active Model**")

    st.code(
        "openai/gpt-oss-120b",
        language="text",
    )

    st.divider()

    st.markdown("**Session Information**")

    st.markdown(
        f"""
        <span class='badge'>Thread ID</span>
        `{st.session_state.thread_id[:8]}...`
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Each browser session has its own isolated conversation memory."
    )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):

        # Create a completely new memory thread
        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Session cleared! 🔄 "
                    "A new memory thread has been started."
                ),
            }
        ]

        st.rerun()


# ---------------------------------------------------------
# Safe Wikipedia Wrapper
# ---------------------------------------------------------

class SafeWikiWrapper(WikipediaAPIWrapper):

    def run(self, query: str) -> str:

        try:
            return super().run(query)

        except Exception as e:
            return f"Wikipedia error: {str(e)}"


# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------

wiki_tool = WikipediaQueryRun(
    api_wrapper=SafeWikiWrapper(
        top_k_results=1,
        doc_content_chars_max=200,
    ),
    handle_tool_error=True,
)


search_tool = DuckDuckGoSearchRun(
    name="web_search",
    handle_tool_error=True,
)


tools = [
    search_tool,
    wiki_tool,
]


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------

st.title("⚡ Nexus Research Agent")

st.caption(
    "Powered by `openai/gpt-oss-120b` "
    "with web search, Wikipedia, persistent memory "
    "and real-time token streaming."
)


# ---------------------------------------------------------
# Display Previous Messages
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

user_prompt = st.chat_input(
    "Ask a question, search current news, or lookup facts..."
)


if user_prompt:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)


    # -----------------------------------------------------
    # Initialize LLM
    # -----------------------------------------------------

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0.2,
        streaming=True,
    )


    # -----------------------------------------------------
    # Create Agent
    # -----------------------------------------------------

    agent = create_agent(
        model=llm,

        tools=tools,

        system_prompt=(
            "You are Nexus, a helpful and precise AI research assistant. "

            "Use Wikipedia when the user asks about factual, "
            "encyclopedic, or historical information. "

            "Use web_search when the user asks for current information, "
            "recent events, news, live information, or internet research. "

            "Use tools whenever they provide more accurate or "
            "up-to-date information than your internal knowledge. "

            "Maintain conversational context from previous messages. "

            "Give clear and concise answers and do not mention "
            "internal implementation details unless asked."
        ),

        checkpointer=st.session_state.checkpointer,
    )


    # -----------------------------------------------------
    # Thread configuration
    # -----------------------------------------------------

    thread_config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


    # -----------------------------------------------------
    # Stream Agent Response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        status_box = st.status(
            "🤔 Agent thinking...",
            expanded=True,
        )

        tools_used = set()


        def agent_token_generator():

            try:

                for token, metadata in agent.stream(
                    {
                        "messages": [
                            (
                                "user",
                                user_prompt,
                            )
                        ]
                    },

                    config=thread_config,

                    stream_mode="messages",
                ):

                    # -----------------------------------------
                    # Detect tool calls
                    # -----------------------------------------

                    if (
                        hasattr(token, "tool_call_chunks")
                        and token.tool_call_chunks
                    ):

                        for chunk in token.tool_call_chunks:

                            tool_name = chunk.get("name")

                            if (
                                tool_name
                                and tool_name not in tools_used
                            ):

                                tools_used.add(tool_name)

                                status_box.write(
                                    f"⚡ Calling tool: `{tool_name}`"
                                )

                                status_box.update(
                                    label=(
                                        f"Querying {tool_name}..."
                                    ),
                                    state="running",
                                )


                    # -----------------------------------------
                    # Stream model output
                    # -----------------------------------------

                    if (
                        metadata.get("langgraph_node") == "model"
                        and token.content
                    ):

                        yield token.content


                # -----------------------------------------
                # Finish status
                # -----------------------------------------

                if tools_used:

                    status_box.update(
                        label=(
                            "Completed using: "
                            + ", ".join(tools_used)
                        ),
                        state="complete",
                        expanded=False,
                    )

                else:

                    status_box.update(
                        label="Direct answer",
                        state="complete",
                        expanded=False,
                    )


            except Exception as e:

                status_box.update(
                    label="Agent error",
                    state="error",
                    expanded=True,
                )

                yield (
                    f"\n\n❌ **Error:** {str(e)}"
                )


        # ---------------------------------------------
        # Stream directly into chat
        # ---------------------------------------------

        full_response = st.write_stream(
            agent_token_generator()
        )


    # -----------------------------------------------------
    # Save Assistant Response
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )
```

### Your `requirements.txt`

I'd also change yours to this:

```txt
langgraph
langchain
langsmith
python-dotenv
ipykernel

langchain-community
langchain-text-splitters
langchain-openai
langchain-groq
langchain-huggingface
langchain-chroma

pypdf
pymupdf
beautifulsoup4
arxiv
wikipedia

chromadb
faiss-cpu
sentence-transformers

pandas
openai
duckdb

ddgs
streamlit
```

**Remove:**

```txt
duckduckgo-search
```

And make sure you don't have `pymupdf` twice.

### Streamlit Cloud Secret

In **Streamlit Cloud → Manage app → Settings → Secrets**, add:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxx"
```

You **do not** put that key in GitHub.

One more important thing: if Streamlit keeps deploying with **Python 3.14**, set the app's Python version to **3.12** in the deployment settings if available. That will make this LangChain stack cons

