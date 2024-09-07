import os 
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import TextLoader

def Load_text(file_path):
    loader = TextLoader(file_path)
    documents= loader.load()
    return documents



def Split_Documents(documents):
    sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
    sent_docs = sent_splitter.split_documents(documents)
    return sent_docs



if __name__ == "__main__":
    file_path = "text/constitution.txt"
    documents = Load_text(file_path)
    texts=Split_Documents(documents)
    print(len(texts))
