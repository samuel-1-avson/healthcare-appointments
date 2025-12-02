# src/llm/callbacks/logging_callback.py
"""
Logging Callback
================
Logs LLM calls for debugging and monitoring.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

logger = logging.getLogger("healthcare_llm")


class LoggingCallback(BaseCallbackHandler):
    """
    Callback handler that logs all LLM interactions.
    
    Logs:
    - Start/end of LLM calls
    - Prompts and responses
    - Token usage
    - Errors
    
    Example
    -------
    >>> from langchain_openai import ChatOpenAI
    >>> callback = LoggingCallback()
    >>> llm = ChatOpenAI(callbacks=[callback])
    """
    
    def __init__(
        self,
        log_prompts: bool = True,
        log_responses: bool = True,
        log_level: int = logging.INFO,
        max_content_length: int = 500
    ):
        """
        Initialize logging callback.
        
        Parameters
        ----------
        log_prompts : bool
            Whether to log prompts
        log_responses : bool
            Whether to log responses
        log_level : int
            Logging level
        max_content_length : int
            Maximum content length to log
        """
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.log_level = log_level
        self.max_content_length = max_content_length
        
        # Track calls
        self._call_stack: Dict[str, Dict] = {}
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        """Log when LLM starts."""
        run_key = str(run_id)
        
        self._call_stack[run_key] = {
            "start_time": datetime.utcnow(),
            "model": serialized.get("name", "unknown")
        }
        
        log_data = {
            "event": "llm_start",
            "run_id": run_key[:8],
            "model": serialized.get("name", "unknown"),
            "prompt_count": len(prompts)
        }
        
        if self.log_prompts and prompts:
            prompt_preview = prompts[0][:self.max_content_length]
            log_data["prompt_preview"] = prompt_preview
        
        logger.log(self.log_level, f"LLM Start: {json.dumps(log_data)}")
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ):
        """Log when chat model starts."""
        run_key = str(run_id)
        
        self._call_stack[run_key] = {
            "start_time": datetime.utcnow(),
            "model": serialized.get("name", "unknown")
        }
        
        message_count = sum(len(batch) for batch in messages)
        
        log_data = {
            "event": "chat_start",
            "run_id": run_key[:8],
            "model": serialized.get("name", "unknown"),
            "message_count": message_count
        }
        
        if self.log_prompts and messages:
            last_msg = messages[0][-1].content if messages[0] else ""
            log_data["last_message_preview"] = last_msg[:self.max_content_length]
        
        logger.log(self.log_level, f"Chat Start: {json.dumps(log_data)}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ):
        """Log when LLM completes."""
        run_key = str(run_id)
        call_data = self._call_stack.pop(run_key, {})
        
        # Calculate latency
        start_time = call_data.get("start_time", datetime.utcnow())
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Get token usage
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        
        log_data = {
            "event": "llm_end",
            "run_id": run_key[:8],
            "model": call_data.get("model", "unknown"),
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0)
        }
        
        if self.log_responses and response.generations:
            first_gen = response.generations[0][0] if response.generations[0] else None
            if first_gen:
                content = first_gen.text[:self.max_content_length]
                log_data["response_preview"] = content
        
        logger.log(self.log_level, f"LLM End: {json.dumps(log_data)}")
    
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ):
        """Log LLM errors."""
        run_key = str(run_id)
        self._call_stack.pop(run_key, None)
        
        log_data = {
            "event": "llm_error",
            "run_id": run_key[:8],
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        logger.error(f"LLM Error: {json.dumps(log_data)}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs
    ):
        """Log chain start."""
        log_data = {
            "event": "chain_start",
            "run_id": str(run_id)[:8],
            "chain_name": serialized.get("name", "unknown"),
            "input_keys": list(inputs.keys())
        }
        logger.log(self.log_level, f"Chain Start: {json.dumps(log_data)}")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs
    ):
        """Log chain end."""
        log_data = {
            "event": "chain_end",
            "run_id": str(run_id)[:8],
            "output_keys": list(outputs.keys()) if isinstance(outputs, dict) else ["output"]
        }
        logger.log(self.log_level, f"Chain End: {json.dumps(log_data)}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs
    ):
        """Log tool start."""
        log_data = {
            "event": "tool_start",
            "run_id": str(run_id)[:8],
            "tool_name": serialized.get("name", "unknown"),
            "input_preview": input_str[:200]
        }
        logger.log(self.log_level, f"Tool Start: {json.dumps(log_data)}")
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs
    ):
        """Log tool end."""
        log_data = {
            "event": "tool_end",
            "run_id": str(run_id)[:8],
            "output_preview": output[:200] if isinstance(output, str) else str(output)[:200]
        }
        logger.log(self.log_level, f"Tool End: {json.dumps(log_data)}")