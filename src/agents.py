# src/agents.py
from langchain.agents import initialize_agent, AgentType


def create_agent_executor(prompt, llm_with_tools, tools):
    """
    Initialize and return a LangChain agent executor using the provided prompt,
    LLM, and tools.
    """
    agent = initialize_agent(
        tools=tools,
        llm=llm_with_tools,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        prompt=prompt
    )
    return agent
