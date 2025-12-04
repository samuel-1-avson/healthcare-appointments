import sys
try:
    import pydantic
    print(f"Pydantic version: {pydantic.VERSION}")
except ImportError:
    print("Pydantic not installed")

try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
except ImportError:
    print("LangChain not installed")

try:
    import langchain_core
    print(f"LangChain Core version: {langchain_core.__version__}")
except ImportError:
    print("LangChain Core not installed")

try:
    from langchain_core.messages import ToolMessage
    print("ToolMessage imported successfully")
except Exception as e:
    print(f"Error importing ToolMessage: {e}")
