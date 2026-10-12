import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser

prompt = """
You are a helpful assistant who replies in simple english and on the topic.
"""

USER_ICON = ":material/person:"
AI_ICON = ":material/smart_toy:"
MODEL_OPTIONS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite"
]

st.set_page_config(page_title="AI Chat Assistant", page_icon=":material/forum:")
st.title(":material/auto_awesome: AI Chat Assistant")
st.caption("Ask me anything")

with st.sidebar:
    st.header(":material/tune: Options")
    selected_model = st.selectbox("Choose a Model", MODEL_OPTIONS)
    if st.button("Clear chat", icon=":material/delete:"):
        st.session_state.chat_history = [SystemMessage(content=prompt)]
        st.rerun()

if selected_model == "gpt-4.1-mini":
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    connect_llm = ChatOpenAI(api_key=api_key,model=selected_model)
else:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    connect_llm = ChatGoogleGenerativeAI(api_key=api_key,model=selected_model)   

chain = connect_llm | StrOutputParser()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=prompt)]

chat_history = st.session_state.chat_history
for message in chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar=USER_ICON):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar=AI_ICON):
            st.write(message.content)

user_input = st.chat_input("Type your message here...")

if user_input:
    if user_input.strip() == "":
        st.warning("Please ask a question", icon=":material/warning:")
    else:
        with st.chat_message("user", avatar=USER_ICON):
            st.write(user_input)

        chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("assistant", avatar=AI_ICON):
            with st.spinner("Thinking..."):
                ai_response = chain.invoke(chat_history)
            st.write(ai_response)

        chat_history.append(AIMessage(content=ai_response))
