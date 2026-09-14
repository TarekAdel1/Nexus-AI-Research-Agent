# AI Research Chatbot 🤖

An AI-powered research chatbot that can search the web and Wikipedia to answer questions with up-to-date information. The chatbot maintains conversation memory, supports multiple users through unique session IDs, and streams responses in real time.

## 🚀 Features

* 🔎 **Web Search** — Searches the web for current information.
* 📚 **Wikipedia Search** — Retrieves information from Wikipedia.
* 🧠 **Conversation Memory** — Remembers previous messages and context during a session.
* 👥 **Multi-User Sessions** — Creates a unique session ID for each user.
* ⚡ **Streaming Responses** — Displays the AI response as it is generated.
* 🛠️ **Tool Calling** — The AI decides when to use available research tools.
* 💬 **Interactive Chat UI** — Built with Streamlit.

## 🧰 Tech Stack

* **Python**
* **LangChain**
* **LangGraph**
* **Streamlit**
* **Groq / LLM API**
* **Web Search**
* **Wikipedia**
* **Python-dotenv**

## 🏗️ How It Works

The user sends a question through the Streamlit interface. The AI agent analyzes the question and decides whether it needs to use external tools such as web search or Wikipedia.

The conversation state is stored using a unique session ID, allowing the chatbot to remember previous messages and maintain context throughout the conversation.

The response is then streamed back to the user in real time.

```text
User
  ↓
Streamlit Chat UI
  ↓
LangGraph Agent
  ↓
 ┌───────────────┐
 │  LLM Reasoning │
 └───────┬───────┘
         ↓
   ┌─────┴─────┐
   ↓           ↓
Web Search   Wikipedia
   ↓           ↓
   └─────┬─────┘
         ↓
   Final Response
         ↓
   Streaming UI
```

## 📁 Project Structure

```text
.
├── app.py              # Streamlit application
├── agent.py             # Agent and workflow logic
├── tools.py             # External tools
├── requirements.txt     # Project dependencies
├── .env                 # API keys and environment variables
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/TarekAdel1/<repository-name>.git
cd <repository-name>
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Add any additional API keys required by the tools you are using.

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL provided by Streamlit in your browser.

## 🎯 Project Type

This project is an **Agentic AI application** built around an LLM agent that can use external tools, maintain conversation state, and dynamically decide how to handle user queries.

## 📌 Future Improvements

* Add more research tools
* Add persistent long-term memory
* Add document/PDF research
* Improve agent planning and reasoning
* Add authentication and user accounts
* Deploy the application publicly
