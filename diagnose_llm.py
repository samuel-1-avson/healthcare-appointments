
import sys
import logging
import asyncio
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        print("Importing config...")
        from src.llm.config import get_llm_config
        config = get_llm_config()
        print(f"Config loaded. Provider: {config.default_provider}, Model: {config.default_model}")

        print("Importing langchain_config...")
        from src.llm.langchain_config import get_chat_model
        model = get_chat_model()
        print(f"Chat model created: {model}")

        print("Importing orchestrator...")
        from src.llm.chains.orchestrator import HealthcareOrchestrator
        orchestrator = HealthcareOrchestrator()
        print("Orchestrator instantiated.")

        print("Running process (synchronous call inside async function)...")
        # This simulates what happens in the API endpoint
        result = orchestrator.process("Hello", session_id="test-session")
        print(f"Result: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
