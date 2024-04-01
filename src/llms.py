"""
LLM
"""
import os
from typing import Optional

from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.openai import OpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler

from src import CFG
from src.helpers.findenv import load_env


def build_llm():
    """Builds LLM defined in config."""
    if CFG.USE_OPENAI:
        load_env()
        return build_openai(
            model_name=CFG.LLM_PATH,
            config={
                "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
            },
        )
    elif CFG.USE_CTRANSFORMERS:
        return build_ctransformers(
            os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
            config={
                "max_new_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
                "temperature": CFG.LLM_CONFIG.TEMPERATURE,
                "repetition_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
                "context_length": CFG.LLM_CONFIG.CONTEXT_LENGTH,
            },
        )
    config = {
        "max_tokens": CFG.LLM_CONFIG.MAX_NEW_TOKENS,
        "temperature": CFG.LLM_CONFIG.TEMPERATURE,
        "repeat_penalty": CFG.LLM_CONFIG.REPETITION_PENALTY,
        "n_ctx": CFG.LLM_CONFIG.CONTEXT_LENGTH,
    }
    if CFG.DEVICE != "cpu":
        config["n_gpu_layers"] = CFG.LLM_CONFIG.N_GPU_LAYERS

    return build_llamacpp(
        os.path.join(CFG.MODELS_DIR, CFG.LLM_PATH),
        config=config,
    )


def build_ctransformers(
    model_path: str, config: Optional[dict] = None, debug: bool = False, **kwargs
):
    """Builds LLM using CTransformers."""
    if config is None:
        config = {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "repetition_penalty": 1.1,
            "context_length": 1024,
        }

    llm = CTransformers(
        model=model_path,
        config=config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm

def build_openai(model_name: str, config: Optional[dict] = None, **kwargs):
    """Builds LLM using OpenAI."""
    if config is None:
        config = {
            "max_tokens": 512,
            "temperature": 0.2,
            "repetition_penalty": 1.1,
        }

    llm = OpenAI(model_name=model_name, **config, **kwargs)
    return llm


def build_llamacpp(
    model_path: str, config: Optional[dict] = None, debug: bool = False, **kwargs
):
    """Builds LLM using LlamaCpp."""
    if config is None:
        config = {
            "max_tokens": 512,
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "n_ctx": 2048,
            "n_gpu_layers": -1
        }

    llm = LlamaCpp(
        model_path=model_path,
        **config,
        callbacks=[StreamingStdOutCallbackHandler()] if debug else None,
        **kwargs,
    )
    return llm
