from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from streamlit import streamlit as st

from dotenv import load_dotenv

load_dotenv()

#Page config
st.set_page_config(page_title="Content Writer and Researcher", page_icon="🖊️", layout="wide")

#Title and description

st.title("Content Writer and Researcher, powered by CrewAI")
st.markdown("Generate blog posts about any topic using Ai Agents")

# Sidebar 

with st.sidebar:
    st.header("Content Settings")
    
    topic = st.text_area(
        "Enter Your Topic",
        height=100,
        placeholder="Enter the topic"
    )

    st.markdown("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    st.markdown("---")

    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    with st.expander("How to use"):
        st.markdown("""
            1. Enter your desired text
            2. Play with temperature
            3. Click 'Generate' button
            4. Wait for AI to generate content
            5. Download the result
        """)

def generate_content(topic):
    llm = LLM(model="gpt-4")

    search_tool = SerperDevTool(n=1)

    senior_research_analyst = Agent(
        role = "Senior Research Analyst",
        goal = f"Research, analyze, and synthesize comprehensive information on {topic} from reliable sources",
        backstory = "You are an expert research analyst with advanced web research skills. You excel at finding, analyzing, and synthesizing "
                    "information from across the internet using search tools. You are skilled at dishtinguishing reliable sources from "
                    "unreliable ones, fact checking, cross-referencing information, and indentifing well-organized research briefs with proper "
                    "citations and source verification. Your analysis includes both raw data and interpreted insights, making complex information "
                    "accesible and actionable.",
        allow_delegation = False,
        verbose = True,
        tools = [search_tool],
        llm = llm,
    )

    content_writer = Agent(
        role = "Content Writer",
        goal = "Transform research findings into engaging blog posts while maintaining accuracy",
        backstory = "You are a skilled content writer specialized in creating engaging, accesible content from technical research. You work closely "
                    "with the Senior Research Analyst and excel at maintaining the perfect balance between informative and enertaining writing, "
                    "while ensuring all facts and citations from the research are properly incorparated. You have a talent for making complex topics"
                    "approachable without oversimplyfying them.",
        allow_delegation = False,
        verbose = True,
        tools = [search_tool],
        llm = llm,
    )

    research_tasks = Task(
        description=("""
            1. Conduct comprehensive research on {topic} including:
                - Recent developments and news
                - Key industry trends and inovations
                - Expert opinions and analysis
                - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources
        """),
        expected_output=("""
            A detailed research report containing the following:
                - Excecutive summary of key findings
                - Comprehensive analysis of current trends and developments
                - List of verified facts and statistics
                - All citations and links to original sources
                - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference.
        """),
        agent=senior_research_analyst
    )

    writing_task = Task(
        description=("""
            Using the research brief provided, create an engaging blog post that:
                1. Transforms technicalinformation into accesible content
                2. Maintains all factual accuracy and citations from the research
                3. Includes: 
                    - Attention-Grabbing introduction
                    - Well-structured body sections with clear headings
                    - Compelling conclusion
                4. Preserves all source citations in [Source: URL] format
                5. Includes a References section at the end
        """),
        expected_output=("""
            A polished blog post in markdown format that:
                - Engages readers while maintaining accuracy
                - Contains properly structured sections
                - Includes Inline Citations with hyperlink to original source url
                - Presents information in an accesible yet informative way
                - Follows proper markdown formatting, use h1 for the title and h3 for the sub-sections
        """),
        agent=content_writer
    )

    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_tasks, writing_task],
        verbose=True
    )

    return crew.kickoff({"topic": topic})

if generate_button:
    with st.spinner("Generating Content... This may take a moment."):
        try:
            result = generate_content(topic)
            st.markdown("### Generated Content")
            st.markdown(result)

            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.lower().replace(" ", "_")}_article.md",
                mime="text/markdown"
            )
        except Exception as err:
            st.error(f"An error has occured: {str(err)}")

st.markdown("---")
st.markdown("Built with CrewAI, ChatGPT, and Streamlit")