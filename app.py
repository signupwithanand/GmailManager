import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI LLM with the API key
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Define the prompt template
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# Create a chain that links the prompt template to the LLM
llm_chain = prompt | llm

# Define a dummy tool as an example (replace with actual tools as needed)
class DummyTool(BaseTool):
    name = "dummy_tool"
    description = "This is a dummy tool for demonstration purposes."
    
    def _run(self, input: str) -> str:
        return "Dummy response"

    async def _arun(self, input: str) -> str:
        return "Dummy response"

dummy_tool_instance = DummyTool()
tools = [Tool(name=dummy_tool_instance.name, func=dummy_tool_instance._run, description=dummy_tool_instance.description)]

# Initialize the agent
agent = initialize_agent(tools=tools, llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Main function to interact with the agent
def main():
    user_input = input("Ask your AI agent a question: ")
    response = agent.run(user_input)
    print(f"AI Agent: {response}")

if __name__ == "__main__":
    main()
