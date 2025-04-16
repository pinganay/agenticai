from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_community.agent_toolkits import GmailToolkit
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from streamlit import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()

toolkit = GmailToolkit()

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)

api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY)

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=False,
)

#Page config
st.set_page_config(page_title="Gmail Assistant", page_icon="✉️", layout="wide")

#Title and description

st.title("Your personal Gmail assistant")
st.markdown("Ask an agent to help with your gmail")

# Sidebar 

with st.sidebar:
    st.header("Content Settings")

    topic = st.text_area(
        "Enter Your Command",
        height=100,
        placeholder="Enter the Command"
    )
    
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    with st.expander("How to use"):
        st.markdown("""
            1. Enter your desired text
            2. Play with temperature
            3. Click 'Generate' button
            4. Wait for AI to generate content
            5. Download the result
        """)

def invoke_agent(topic):
    result = agent_executor.invoke(
        {
            "input": "Read the first 4 emails in my inbox and show me in multiple lines."
        }
    )

    output = result["output"]

    return output

if generate_button:
    with st.spinner("Generating Content... This may take a moment."):
        try:
            result = invoke_agent(topic)
            st.markdown("### Generated Content")
            st.markdown(result)

        except Exception as err:
            st.error(f"An error has occured: {str(err)}")

st.markdown("---")
st.markdown("Built with CrewAI, ChatGPT, and Streamlit")