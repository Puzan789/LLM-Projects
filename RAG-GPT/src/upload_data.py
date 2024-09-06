import os
from utils.prepare_vectordb import PrepareVectorDb
from utils.load_config import LoadConfig
CONFIG=LoadConfig()

def upload_data_manually() -> None:
    prepare_vector_db_instance=PrepareVectorDb(
        data_directory=CONFIG.data_directory,
        persist_directory=CONFIG.persist_directory,
        embedding_model_engine=CONFIG.embedding_model_engine,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    if not len(os.listdir(CONFIG.persist_directory)) != 0:
        prepare_vector_db_instance.Save_in_vector_db()
    else:
        print(f"VectorDB already exists in {CONFIG.persist_directory}")
    return None

if __name__ == "__main__":
    upload_data_manually()