import os

import streamlit as st

from rag import BOT_NAME, KnowledgeBase

st.set_page_config(page_title=BOT_NAME, page_icon="🤖")


def get_setting(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


api_key = get_setting("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it in Streamlit secrets or as an environment variable.")
    st.stop()


@st.cache_resource(show_spinner="Reading the CVs and building the index...")
def load_kb():
    return KnowledgeBase(
        api_key=api_key,
        data_dir="data",
        embed_model=get_setting("EMBED_MODEL", "gemini-embedding-001"),
        chat_model=get_setting("CHAT_MODEL", "gemini-3.5-flash-lite"),
    )


kb = load_kb()

st.title(f"🤖 {BOT_NAME}")
st.caption("Ask me about Muhammad Zain or Uzair Bin Ahmad: education, experience, projects and skills.")

with st.sidebar:
    st.header(BOT_NAME)
    st.write("A RAG chatbot built on the CVs of Muhammad Zain and Uzair Bin Ahmad.")
    st.subheader("Try asking")
    samples = [
        "Who is Muhammad Zain?",
        "What projects has Zain built?",
        "What are his technical skills?",
        "Where does he work and what does he study?",
        "Tell me about his NLP project.",
        "Who is Uzair Bin Ahmad?",
        "What are Uzair's technical skills?",
        "What projects has Uzair built?",
    ]
    clicked = None
    for s in samples:
        if st.button(s, use_container_width=True):
            clicked = s
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input(f"Ask {BOT_NAME} a question...") or clicked

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, hits = kb.answer(question, history)
            except Exception as e:
                answer, hits = f"Sorry, something went wrong: {e}", []
        st.markdown(answer)
        if hits:
            with st.expander("Retrieved context"):
                for chunk, score in hits:
                    st.caption(f"Similarity: {score:.2f}")
                    st.text(chunk)
    st.session_state.messages.append({"role": "assistant", "content": answer})
