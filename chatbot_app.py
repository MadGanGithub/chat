import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 

import chromadb
from chromadb.config import Settings 
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)


#set model to be run on CPU rather than GPU
device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

persist_directory = "db"

def data_ingestion(filepath):
    loader=PDFMinerLoader(os.path.abspath(f"file/{filepath}"))
                
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


def main():
    filepath=input("Enter file name: ")
    filepath=filepath+".pdf"
    ingested_data = data_ingestion(filepath)

    while(True):
        user_input = input("Enter the response:")

        if user_input:
            answer = process_answer({'query': user_input})
            response = answer
            print("BOT reply: "+response)

if __name__ == "__main__":
    main()