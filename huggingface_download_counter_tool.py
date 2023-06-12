from guidance.tools import huggingface_tool

tool = huggingface_tool.load_huggingface_tool("lysandre/hf-model-downloads")

print(f"{tool.name}: {tool.description}")

print("The most downloaded model for depth-estimation : ", tool.run("depth-estimation"))

#Try for "text-classification" --->> facebook/bart-large-mnli
