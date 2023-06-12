from guidance.tools import huggingface_tool

tool = huggingface_tool.load_huggingface_tool("lysandre/hf-model-downloads")

print(f"{tool.name}: {tool.description}")

print("The most downloaded model for depth-estimation : ", tool.run("depth-estimation"))

print("The most downloaded model for text-classification : ", tool.run("text-classification"))
