import os

file_path = r"C:\Users\samue\Desktop\NSP\healthcare-appointments\src\llm\memory\__init__.py"
content = """# src/llm/memory/__init__.py
\"\"\"
Conversation Memory Module
==========================
Memory implementations for maintaining conversation context.
\"\"\"

from .conversation_memory import (
    ConversationMemoryManager,
    create_memory,
    get_memory_for_session
)

__all__ = [
    "ConversationMemoryManager",
    "create_memory",
    "get_memory_for_session"
]
"""

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Successfully overwrote {file_path}")
with open(file_path, "r", encoding="utf-8") as f:
    print("New content:")
    print(f.read())
