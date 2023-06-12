from typing import Any, Dict, List, Optional, Callable, Tuple
from .base import BaseTool, Tool

def load_huggingface_tool(
    task_or_repo_id: str,
    model_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    remote: bool = False,
    **kwargs: Any,
) -> BaseTool:
    try:
        from transformers import load_tool
    except ImportError:
        raise ValueError(
            "HuggingFace tools require the libraries `transformers>=4.29.0`"
            " and `huggingface_hub>=0.14.1` to be installed."
            " Please install it with"
            " `pip install --upgrade transformers huggingface_hub`."
        )
    hf_tool = load_tool(
        task_or_repo_id,
        model_repo_id=model_repo_id,
        token=token,
        remote=remote,
        **kwargs,
    )
    outputs = hf_tool.outputs
    if set(outputs) != {"text"}:
        raise NotImplementedError("Multimodal outputs not supported yet.")
    inputs = hf_tool.inputs
    if set(inputs) != {"text"}:
        raise NotImplementedError("Multimodal inputs not supported yet.")
    return Tool.from_function(
        hf_tool.__call__, name=hf_tool.name, description=hf_tool.description
    )