from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_community.agent_toolkits import GmailToolkit
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent

import os
from dotenv import load_dotenv

load_dotenv()
#os.environ['TAVILY_API_KEY'] = load_dotenv("TAVILY_API_KEY")

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
    # This is set to False to prevent information about my email showing up on the screen
    # Normally, it is helpful to have it set to True however.
    verbose=False,
)

print(agent_executor.invoke(
    {
        "input": "Read the first 4 emails in my inbox and show me in multiple lines."
    }
))