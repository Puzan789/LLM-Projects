from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
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
        self.embedding=GoogleGenerativeAIEmbeddings(model=self.embedding_model_engine,maxBatchSize=100)
    
    def __load_all_documents(self):
        doc_counter=0
        if isinstance(self.data_directory,list):
            print("uploaded documents loading\n")
            docs=[]
            for doc in self.data_directory:
                docs.extend(PyPDFLoader(doc).load())
                doc_counter+=1
            print(f"loaded {doc_counter} documents\n")
            print(f"Number of pages : {len(docs)}\n")
        else:
            print("Loading documents manually ")
            current_path = os.getcwd()
            print(f"Current path: {current_path}")
            docs=PyPDFDirectoryLoader(self.data_directory).load()
            
        return docs
    
    def __create_chunk(self,docs:List)->List:
        print("Creating chunk \n")
        chunked_documents=self.text_splitter.split_documents(docs)
        return chunked_documents
    
    def Save_in_vector_db(self):
        documents=self.__load_all_documents()
        chunked_documents=self.__create_chunk(documents)
        print("preparing vectored database \n \n")
        vector_store=Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
        )
        print("vector DB is created and saved \n")
        print("VectorDB is created and saved.")
        print("Number of vectors in vectordb:",
              vector_store._collection.count(), "\n\n")
        return vector_store
