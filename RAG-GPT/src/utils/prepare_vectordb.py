from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#preparing Vectordb

class PrepareVectorDb:
    """A class that vec"""

    def __init__(self,data_directory:str,persist_directory:str,embedding_model_engine:str,chunk_size:int,chunk_overlap:int) -> None:
        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                            chunk_overlap=chunk_overlap,
                                                            separators=["\n\n", "\n", " ", ""])
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding=GoogleGenerativeAIEmbeddings(model=self.embedding_model_engine)
        
    
        