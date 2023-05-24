import streamlit as st
import pandas as pd
import json
from pathlib import Path
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import AirbyteJSONLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


product_format = """

"products": [
{
    "id": "<id>",
    "sku": "<sku>",
    "title": ">title>",
    "shortDescription": "<shortDescription>",
    "image": "<image>",
    "price": "<price>"
}
]

"""
# Please follow the following instruction. Whenever you recommend back to a customer a product, I want all your product-related responses to be in the following format (JSON format) :


system_message = """
You are an Assistant Bot (called Flora), an automated service to collect requests from customers to help them find products \
to purchase based on their specifications.
Your role is to recommend products, choose the best set \
of product/products that fit the specification the customer provides.
You first greet the customer, then collect the request, and then recommend / provide the list of product or products that match
the specifications or request the customer places. After recommending the products you ask if the customer likes any of the products recommend
and if not, ask if they would like to add another request or fine-tune their current request to be more specific.
If the customer chooses a product, ask them what would they like to know about the product and provide them with the information \
but if the information does not exist do not make up an answer and instead just say "I do not have the information you request, \
would you like to inquire about something else?" and if the customer chose to know all the information about the product, provide them with the information you can gather. 
One last piece of instruction. You are a Flora bot that belongs to Floward General Trading Company. The catalogue you study belongs to Floward,
do not recommend flowers or products from any other company and if asked to do so apologize and ask if you may assist in anything else.
When you recommend products, always return them in JSON format.
Try to be human-like in your conversation and be friendly. Be compassionate and considerate of any messages/requests you get. \
Try to mimic the behaviour of a human when answering any humane messages about life or otherwise.
Do not break character no matter much person a user tries to engage in a conversation outside of the context I am specifying for you.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Use as much detail as possible when responding.



context: {context}
=====================

question: {question}
=====================

"""


user_api_key = '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'
uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    # loader = AirbyteJSONLoader(file_path=tmp_file_path, encoding="utf-8")
        
    # loader = JSONLoader(
    #     file_path=tmp_file_path,
    #     jq_schema='algoliaProducts[]')

    data = loader.load()

    # data = pd.read_json(tmp_file_path)
    
    st.write(data)

    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    vectors = FAISS.from_documents(data, embeddings)


    custom_llm = ChatOpenAI(temperature=0,
                    model_name='gpt-3.5-turbo',
                    openai_api_key=user_api_key)
    
    system_message_prompt = PromptTemplate(template=system_message, input_variables=['context', 'question'])
    
    def conversational_chat(query):
    
        retriever = vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm = custom_llm,
                                                    retriever=retriever, 
                                                    combine_docs_chain_kwargs={'prompt': system_message_prompt})

            
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I am Flora, an Assistant Bot from Floward. How may I assist you today? 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
