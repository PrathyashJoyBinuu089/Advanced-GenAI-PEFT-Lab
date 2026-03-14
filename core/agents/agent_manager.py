from typing import List, Optional, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

class AgentConfig(BaseModel):
    model_name: str = Field("gpt-4-turbo", description="OpenAI model for the agent")
    temperature: float = Field(0.0, description="Sampling temperature")
    system_message: str = Field(
        "You are a helpful GenAI assistant that can use tools to answer questions.",
        description="The agent's personality and instructions"
    )

class AgentManager:
    """
    Orchestrates Agentic workflows using LangChain.
    """
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = ChatOpenAI(model=self.config.model_name, temperature=self.config.temperature)
        self.tools = self.setup_tools()
        self.agent_executor = self.setup_agent()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_tools(self) -> List[BaseTool]:
        """Configures the agent's available tools."""
        search = DuckDuckGoSearchRun()
        
        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful for when you need to answer questions about current events or general knowledge."
            )
        ]
        return tools

    def setup_agent(self) -> AgentExecutor:
        """Initializes the OpenAI Functions agent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, input_text: str):
        """Runs the agent on a given user input."""
        chat_history = self.memory.chat_memory.messages
        response = self.agent_executor.invoke({
            "input": input_text,
            "chat_history": chat_history
        })
        
        # Manually update memory since AgentExecutor won't do it automatically here
        self.memory.chat_memory.add_user_message(input_text)
        self.memory.chat_memory.add_ai_message(response["output"])
        
        return response["output"]

if __name__ == "__main__":
    # Example usage
    # config = AgentConfig()
    # manager = AgentManager(config)
    print("Agent Manager logic ready.")
