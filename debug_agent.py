from src.llm.langchain_config import get_chat_model
from src.llm.tools.prediction_tool import PredictionTool

try:
    print("Initializing model...")
    model = get_chat_model()
    print(f"Model type: {type(model)}")
    
    print("Attempting to bind tools...")
    tools = [PredictionTool()]
    if hasattr(model, "bind_tools"):
        model_with_tools = model.bind_tools(tools)
        print("bind_tools successful")
    else:
        print("Model does not have bind_tools method")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
