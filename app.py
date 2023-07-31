import chainlit as cl
import os
from getpass import getpass
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

# API token from user
api_token=getpass("Enter your HuggingFace API token:")
os.environ['HUGGINGFACEHUB_API_TOKEN']=api_token

# Define llm model with configuration parameters like temperatures, max_new_tokens and etc.
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', 
                     model_kwargs={"temperature":0.5, "max_new_tokens":512, "top_k":10, "top_p":0.95})

template = """

Briefly explain about {person}?

"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["person"])
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

#"on_message" decorator will react to messages coming from the UI. The decorated function is called every time a new message is received.    
@cl.on_message                          
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    # Callback handlers ensures seamless interaction between your LangChain agent and the Chainlit UI.
    res = await cl.make_async(llm_chain)(message, callbacks=[cl.LangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()
    return llm_chain
