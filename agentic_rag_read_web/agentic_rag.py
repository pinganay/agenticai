import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

import os
from dotenv import load_dotenv

load_dotenv()
#USER_AGENT=os.getenv("USER_AGENT")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

#Testing llm model
#print(llm.invoke("hello, how are you?"))

urls = [
    "https://www.soccer.com/guide/rules-of-soccer-guide",
    "https://www.sportsengine.com/soccer/rules-soccer-offsides-explained#:~:text=The%20offside%20rule%20is%20one,defender%2C%20not%20including%20the%20goalkeeper."
]

docs = [WebBaseLoader(url).load() for url in urls]
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=5)
docs_list = [item for sublist in docs for item in sublist]
doc_splits=text_splitter.split_documents(docs_list)

vectorstore=Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chrome",
    embedding=embeddings   
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search these urls for soccer data and rules .You are a specialized assistant. Use the 'retriever_tool' **only** when the query explicitly relates soccer. For all other queries, respond directly without using any tool. For simple queries like 'hi', 'hello', or 'how are you', provide a normal response.",
)

tools=[retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def AI_Asisstant(state: AgentState):
    print("---CALL AGENT---")
    messages=state['messages']
    llm_with_tool=llm.bind_tools(tools)
    response=llm_with_tool.invoke(messages)
    return {"messages": [response]}

# def retrieve(state):
#     pass

def rewrite(state:AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    print(f"here is message from rewrite: {messages}")
    
    message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning. 
                    Here is the initial question: {question} 
                    Formulate an improved question: """)
       ]
    response = llm.invoke(message)
    return {"messages": [response]}

def generate(state:AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = prompt | llm

    response = rag_chain.invoke({"context": docs, "question": question})
    print(f"this is my response:{response}")
    
    return {"messages": [response]}

class grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no' ")

def grade_documents(state:AgentState)->Literal["Output_Generator", "Query_Rewriter"]:
    llm_with_structure_op=llm.with_structured_output(grade)
    
    prompt=PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
                    Here is the document: {context}
                    Here is the user’s question: {question}
                    If the document talks about or contains information related to the user’s question, mark it as relevant. 
                    Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
                    input_variables=["context", "question"]
                    )
    chain = prompt | llm_with_structure_op
    
    messages = state["messages"]
    print(f"message from the grader: {messages}")
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generator" #this should be a node name
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewriter" #this should be a node name

workflow=StateGraph(AgentState)

workflow.add_node("ai_assistant", AI_Asisstant)
retrieve=ToolNode([retriever_tool])
workflow.add_node("retriever", retrieve)
workflow.add_node("rewriter", rewrite)
workflow.add_node("generator", generate)

workflow.add_edge(START, "ai_assistant")
workflow.add_conditional_edges(
    "ai_assistant", 
    tools_condition,
    {"tools": "retriever", END: END,}
    )
workflow.add_conditional_edges(
    "retriever", 
    grade_documents,
    {"rewriter": "rewriter", "generator": "generator",}
    )
workflow.add_edge("generator", END)
workflow.add_edge("rewriter", "ai_assistant")

app=workflow.compile()

print(app.invoke({"messages":["How do you make chocolate?"]}))