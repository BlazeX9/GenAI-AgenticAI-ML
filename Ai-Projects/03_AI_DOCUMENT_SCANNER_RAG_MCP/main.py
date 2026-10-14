import asyncio
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from google import genai
from google.genai import types
from pdf_store import pdf_vector_store
from excel_store import excel_vector_store
import configuration

load_dotenv()

retry_settings = types.HttpRetryOptions(attempts=6, initial_delay=2)
http_settings = types.HttpOptions(retry_options=retry_settings)
llm_connection = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"), http_options=http_settings)

server = StdioServerParameters(command="python", args=["mcp_server.py"])

ICONS = {
    "user": ":material/account_circle:",
    "assistant": ":material/smart_toy:",
}

def show_message(message):
    with st.chat_message(message["role"], avatar=ICONS[message["role"]]):
        st.caption(f":material/schedule: {message['time']}")
        st.write(message["content"])

async def callAgent(user_input, model_name):
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            function_list = []

            for tool in mcp_tools.tools:
                function_list.append(types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.inputSchema
                ))

            gemini_tools = [types.Tool(function_declarations=function_list)]

            instructions = """
            You are a helpful assistant who replies in simple English and stays on topic.
            You can use tools provided by the MCP server.
            If the user's question is about the uploaded document or its contents, use the search tool to retrieve relevant information before answering.
            Base document-related answers on the retrieved content.
            If the search results do not contain the answer, clearly say that the document does not provide enough information.
            Do not invent facts.
            """

            first_prompt = instructions + "\nUser question: " + user_input

            first_response = llm_connection.models.generate_content(
                model=model_name,
                contents=first_prompt,
                config=types.GenerateContentConfig(tools=gemini_tools)
            )

            first_part = first_response.candidates[0].content.parts[0]

            if first_part.function_call is None:
                return first_response.text

            tool_name = first_part.function_call.name
            tool_args = dict(first_part.function_call.args)

            tool_result = await session.call_tool(tool_name, tool_args)
            document_text = tool_result.content[0].text

            final_prompt = instructions + "\nDocument content:\n" + document_text + "\n\nUser question: " + user_input

            final_response = llm_connection.models.generate_content(
                model=model_name,
                contents=final_prompt
            )
            return final_response.text


st.title(":material/description: AI Document Scanner")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF, Excel or CSV file", type=["pdf", "xlsx", "csv"])
    selected_model = st.selectbox(
        "Choose model",
        [configuration.GEMINI_MODEL_1, configuration.GEMINI_MODEL_2],
        index=1
    )
    clear_clicked = st.button("Clear chat", icon=":material/delete:")

if uploaded_file is None:
    st.info("Please upload a PDF, Excel or CSV file to start")
    st.stop()

file_id = uploaded_file.name + str(uploaded_file.size)

if st.session_state.get("pdf_id") != file_id:
    with st.spinner("Reading the file..."):
        if uploaded_file.name.lower().endswith(".pdf"):
            pdf_vector_store(uploaded_file)
        else:
            excel_vector_store(uploaded_file)
    st.session_state.pdf_id = file_id
    st.session_state.messages = []

if clear_clicked:
    st.session_state.messages = []

for message in st.session_state.messages:
    show_message(message)

user_input = st.chat_input("Ask a question about the document")

if user_input:
    asked_time = datetime.now().strftime("%I:%M:%S %p")
    user_message = {"role": "user", "content": user_input, "time": asked_time}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.spinner("Thinking..."):
        try:
            answer = asyncio.run(callAgent(user_input, selected_model))
        except Exception:
            answer = "The AI model is busy right now. Please try again in a few seconds."

    answered_time = datetime.now().strftime("%I:%M:%S %p")
    bot_message = {"role": "assistant", "content": answer, "time": answered_time}
    st.session_state.messages.append(bot_message)
    show_message(bot_message)