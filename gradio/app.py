from langchain_ollama import ChatOllama

import gradio as gr

chatbot = ChatOllama(model="llama3.2", base_url="http://ollama:11434")


def chatbot_interface(text: str) -> str:
    """Chatbot interface.

    Args:
        text (str): The input text

    Returns:
        str: The chatbot response
    """
    response = chatbot.invoke(text)
    return response.content


gr.Interface(fn=chatbot_interface, inputs="text", outputs="text").launch()
