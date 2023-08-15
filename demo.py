from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import json
from langchain.chains.summarize import load_summarize_chain

import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

path = ["attention-is-all-you-need.pdf", "qlora.pdf"]


print('indexing documents...')

for i in path:
    loader = PyMuPDFLoader(i)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=" ")
    documents = text_splitter.split_documents(pages)
        
db = Chroma.from_documents(documents = documents, embedding = OpenAIEmbeddings())
print('indexing done...') 

llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

query = input('>>>>: Hello, I am your personal bot. How can I help you?')

template = """
            You are a personal assistant. You answer questions related to the context provided to you only. 
            If the user asks about something which is not present in the context provided, please say "I'm sorry,
            I'm not sure if I can answer that question"
            Only stick to the context provided to you and nothing else.
        
            Use a formal language. You should answer all queries which is related to
            the context which has been given to you.

            Context: {context}

            Conversation History: {history}.
            
            Question: {human_input}
            
"""

prompt = PromptTemplate(
    input_variables=["history", "human_input", "context"], 
    template=template
)

memory = ConversationBufferMemory(memory_key="history", input_key="human_input")
chain = load_qa_chain(llm = llm, chain_type="stuff", memory=memory, prompt=prompt, verbose=True)

while True:
    
    if query == 'No' or query == 'no' or query == 'NO':
        print('Cool. Have a nice day')
        break
        
    else:
        docs = db.similarity_search(query, k=3, search_type = "approximate_search", space_type = "cosinesimil")
        result = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
        output_text = result['output_text']
        print(result['output_text'])
        print('Any further questions?')
        print('\n')
    query = input('>>>>: ')