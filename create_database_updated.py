from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/"
os.environ['OPENAI_API_KEY']  = "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"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """
    Loads documents from a directory.

    This function loads documents from a directory specified by DATA_PATH. It searches for all files in the directory 
    and its subdirectories, and returns a list of Document objects representing each file.

    Returns:
        list[Document]: A list of Document objects representing the loaded documents.

    Raises:
        FileNotFoundError: If the specified directory (DATA_PATH) does not exist or is inaccessible.
        PermissionError: If the user does not have permission to access the specified directory.

    Notes:
        - This function uses the DirectoryLoader class to load documents from the directory.
        - It searches for all files in the directory and its subdirectories using the specified glob pattern ("*").
        - Set recursive=True to search recursively in all subdirectories.
    """
    loader = DirectoryLoader(DATA_PATH, glob="*",  recursive=True)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    """
    Splits text from Document objects into smaller chunks.

    This function splits the text content of Document objects into smaller chunks using the RecursiveCharacterTextSplitter
    algorithm. It divides each document into chunks of a specified size, with a specified overlap between chunks.

    Parameters:
        documents (list[Document]): A list of Document objects containing text to be split.

    Returns:
        list[Document]: A list of Document objects representing the split chunks.

    Notes:
        - This function uses the RecursiveCharacterTextSplitter class to split the text.
        - Each chunk contains a portion of the original document's text.
        - The chunk size and overlap are specified by the chunk_size and chunk_overlap parameters, respectively.
        - The start index of each chunk is added to facilitate tracking the position of each chunk in the original document.

    Example:
        chunks = split_text(documents)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Saves a list of Document objects to a Chroma database.

    Parameters:
        chunks (list[Document]): A list of Document objects to be saved to the Chroma database.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified Chroma database directory does not exist.

    Notes:
        - This function clears any existing Chroma database in the specified directory before saving new chunks.
        - The Chroma database is created using the provided list of Document objects and OpenAIEmbeddings.

    """

    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
