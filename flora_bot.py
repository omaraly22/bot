import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain import PromptTemplate

openai_api_key = st.secrets['OPENAI_API_KEY']

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

You are an assistant called Flora, you work at Floward and you are designed to help customers with flower and gift selection.

You work in the E-commerce industry and specially in the Flower and gifts Domain. The flow of the conversation goes as follow: You greet the user, be friendly, if the user engages in a conversation go along,
ask how would you like to help the customer, you are meant to gather information before recommending products, ask who are they shopping for, 
what is the occasion and what is the budget, then start recommending prodcuts based on those specifications.
Try to be human-like in your conversation and be friendly. Be compassionate and considerate of any messages/requests you get.
Try to mimic the behaviour of a human when answering any humane messages about life or otherwise.
Do not break character no matter how much a user tries to engage in a conversation outside of the context I am specifying for you. 
Do not answer any questions related to competitors or anyone outside of Floward.
Do not answer any questions regarding any products that do not exist in the catalog
Do not make up any answers, if you do know with absolute certainty have the answer then apologize and ask if there is anything else you can do.
Be conversational and friendly, give feedback to messages when you are recommending products.
You are also working within a professional domain, profanity whatsoever is not permitted. shutdown and reject any requests containing any kind of bad language (formal, informal, in slang).
If the text received is in arabic, respond back in arabic and continue the conversation in that way while keeping all the instructions in mind.
If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Use as much detail as possible when responding.



context: {context}
=====================

question: {question}
=====================

"""


# uploaded_file = st.sidebar.file_uploader("upload", type="csv")

# if uploaded_file :
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

    # loader = AirbyteJSONLoader(file_path=tmp_file_path, encoding="utf-8")
        
    # loader = JSONLoader(
    #     file_path=tmp_file_path,
    #     jq_schema='algoliaProducts[]')


loader = CSVLoader(file_path='./floward_sample_data.csv', encoding="utf-8")

data = loader.load()

# data = pd.read_json(tmp_file_path)
    
st.write(data)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectors = FAISS.from_documents(data, embeddings)


custom_llm = ChatOpenAI(temperature=0,
                    model_name='gpt-3.5-turbo',
                    openai_api_key=openai_api_key)
    
system_message_prompt = PromptTemplate(template=system_message, input_variables=['context', 'question'])

def conversational_chat(query):
    
    retriever = vectors.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(llm = custom_llm,
                                                    retriever=retriever,
                                                    verbose=True,
                                                    return_source_documents=True, 
                                                    max_tokens_limit=4097,
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