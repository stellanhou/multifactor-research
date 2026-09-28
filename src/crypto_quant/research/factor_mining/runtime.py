"""Shared LangChain role calls and LangGraph execution policy.

Research evidence remains in the domain stores; graphs do not add a second
checkpoint database or retry mutations, market reads, or validation stages.
"""
import sys
from importlib.metadata import version

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda


GRAPH_CONFIG = {"recursion_limit": sys.maxsize}


def runtime_versions():
    return {name: version(name) for name in ("langchain-core", "langchain-openai", "langgraph")}


def role_chain(model, *, max_output_tokens, session_id):
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])

    def generate(value):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        messages = [{"role": roles[item.type], "content": item.content} for item in value.to_messages()]
        return model.complete(messages, max_output_tokens=max_output_tokens, session_id=session_id)

    return prompt | RunnableLambda(generate, name="model_transport")


def invoke_role(model, messages, *, max_output_tokens, session_id):
    return role_chain(model, max_output_tokens=max_output_tokens, session_id=session_id).invoke(
        {"messages": messages}, config={"run_name": session_id})
