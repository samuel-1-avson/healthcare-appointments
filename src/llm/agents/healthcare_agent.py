# src/llm/agents/healthcare_agent.py
"""
Healthcare Agent
================
Autonomous agent with tools for healthcare appointment management.

The agent can:
- Make predictions using the ML API
- Explain predictions in natural language
- Recommend interventions
- Answer policy questions
- Maintain conversation context
"""

import logging
from typing import Dict, Any, Optional, List, Sequence
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

from ..langchain_config import get_chat_model
from ..tools.prediction_tool import PredictionTool, create_batch_prediction_tool
from ..memory.conversation_memory import get_memory_manager, InMemoryHistory


logger = logging.getLogger(__name__)


# ==================== Agent System Prompt ====================

HEALTHCARE_AGENT_PROMPT = """You are a Healthcare Appointment Assistant, an AI agent that helps healthcare staff manage patient appointments and reduce no-shows.

## Your Tools

You have access to these tools:
1. **predict_noshow**: Get a no-show risk prediction for a patient appointment
2. **batch_predict_noshows**: Get predictions for multiple appointments at once

## When to Use Tools

USE the predict_noshow tool when:
- User asks about a specific patient's risk
- User provides patient details (age, appointment date, etc.)
- User wants to know if intervention is needed

DO NOT use tools when:
- User asks general questions about no-shows
- User asks about policies or best practices
- User wants explanations of past predictions

## Response Guidelines

1. When making predictions:
   - Call the tool with the provided information
   - Explain the results in plain language
   - Provide specific recommendations

2. When explaining concepts:
   - Be clear and educational
   - Use examples when helpful
   - Avoid medical jargon

3. Always:
   - Be empathetic and patient-focused
   - Acknowledge uncertainty
   - Recommend human review for critical decisions

## Current Context
Today's date: {current_date}
You are helping staff at a healthcare clinic manage their appointment schedule.
"""

REACT_AGENT_PROMPT = """You are a Healthcare Appointment Assistant.
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# ==================== Agent Implementation ====================

class HealthcareAgent:
    """
    Healthcare Assistant Agent with tool-calling capabilities.
    
    This agent can autonomously decide when to:
    - Call the prediction API for risk assessment
    - Provide explanations from its knowledge
    - Recommend interventions
    
    Example
    -------
    >>> agent = HealthcareAgent()
    >>> session_id = agent.create_session()
    >>> response = agent.chat(
    ...     session_id,
    ...     "What's the risk for a 45-year-old patient with an appointment in 2 weeks?"
    ... )
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        """
        Initialize the healthcare agent.
        
        Parameters
        ----------
        model_name : str, optional
            LLM model to use
        temperature : float
            Sampling temperature
        tools : list, optional
            Custom tools (default: prediction tools)
        max_iterations : int
            Max tool-calling iterations
        verbose : bool
            Enable verbose logging
        """
        self.model = get_chat_model(model_name, temperature)
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize tools
        self.tools = tools or self._get_default_tools()
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager()
        
        # Build agent
        self._agent_executor = self._build_agent()
        
        # Tracking
        self._call_count = 0
        self._tool_calls = 0
        
        self.logger.info(f"HealthcareAgent initialized with {len(self.tools)} tools")
    
    def _get_default_tools(self) -> List[BaseTool]:
        """Get default tools for the agent."""
        return [
            PredictionTool(),
            create_batch_prediction_tool()
        ]
    
    def _build_agent(self) -> AgentExecutor:
        """Build the agent executor."""
        
        agent = None
        
        # Try to create tool calling agent first
        try:
            if hasattr(self.model, "bind_tools"):
                # Create prompt for tool calling agent
                prompt = ChatPromptTemplate.from_messages([
                    ("system", HEALTHCARE_AGENT_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
                ])
                
                # Create agent with tool calling
                # This might raise NotImplementedError if bind_tools is present but not implemented
                agent = create_tool_calling_agent(
                    llm=self.model,
                    tools=self.tools,
                    prompt=prompt
                )
        except (NotImplementedError, Exception) as e:
            self.logger.warning(f"Tool calling not supported ({e}), falling back to ReAct agent")
            agent = None

        # Fallback to ReAct agent
        if agent is None:
            self.logger.info("Using ReAct agent fallback")
            # Create prompt for ReAct agent
            prompt = PromptTemplate.from_template(REACT_AGENT_PROMPT)
            
            # Create ReAct agent
            agent = create_react_agent(
                llm=self.model,
                tools=self.tools,
                prompt=prompt
            )
        
        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        return executor
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        return self.memory_manager.create_session(session_id)
    
    def chat(
        self,
        session_id: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message to the agent.
        
        Parameters
        ----------
        session_id : str
            Conversation session ID
        message : str
            User message
        
        Returns
        -------
        dict
            Response with output and metadata
        """
        # Ensure session exists
        self.memory_manager.get_or_create_session(session_id)
        
        # Get chat history
        history = self.memory_manager.get_conversation_messages(session_id)
        
        # Prepare input
        agent_input = {
            "input": message,
            "chat_history": history,
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Invoke agent
        start_time = datetime.utcnow()
        
        try:
            result = self._agent_executor.invoke(agent_input)
            
            # Extract output
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Track tool usage
            tool_calls = len(intermediate_steps)
            self._tool_calls += tool_calls
            self._call_count += 1
            
            # Save to memory
            self.memory_manager.add_exchange(session_id, message, output)
            
            # Build response
            response = {
                "output": output,
                "tool_calls": self._format_tool_calls(intermediate_steps),
                "metadata": {
                    "session_id": session_id,
                    "latency_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "tools_used": tool_calls,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            raise
    
    async def achat(
        self,
        session_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Async chat with the agent."""
        self.memory_manager.get_or_create_session(session_id)
        history = self.memory_manager.get_conversation_messages(session_id)
        
        agent_input = {
            "input": message,
            "chat_history": history,
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        result = await self._agent_executor.ainvoke(agent_input)
        output = result.get("output", "")
        
        self.memory_manager.add_exchange(session_id, message, output)
        
        return {
            "output": output,
            "tool_calls": self._format_tool_calls(result.get("intermediate_steps", []))
        }
    
    def _format_tool_calls(self, steps: List) -> List[Dict[str, Any]]:
        """Format intermediate steps for response."""
        tool_calls = []
        
        for action, observation in steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": str(observation)[:500]  # Truncate long outputs
            })
        
        return tool_calls
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        messages = self.memory_manager.get_conversation_messages(session_id)
        
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in messages
        ]
    
    def clear_history(self, session_id: str) -> None:
        """Clear session history."""
        self.memory_manager.clear_session(session_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_calls": self._call_count,
            "total_tool_calls": self._tool_calls,
            "tools_available": [t.name for t in self.tools],
            "active_sessions": len(self.memory_manager.list_sessions())
        }


# ==================== Factory Functions ====================

def create_healthcare_agent(
    model_name: Optional[str] = None,
    include_batch_tool: bool = True,
    verbose: bool = False
) -> HealthcareAgent:
    """
    Create a configured healthcare agent.
    
    Parameters
    ----------
    model_name : str, optional
        LLM model to use
    include_batch_tool : bool
        Include batch prediction tool
    verbose : bool
        Enable verbose mode
    
    Returns
    -------
    HealthcareAgent
        Configured agent instance
    """
    tools = [PredictionTool()]
    
    if include_batch_tool:
        tools.append(create_batch_prediction_tool())
    
    return HealthcareAgent(
        model_name=model_name,
        tools=tools,
        verbose=verbose
    )