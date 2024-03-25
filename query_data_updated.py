import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

os.environ['OPENAI_API_KEY']  = "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"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    """
    Main function for performing a search and generating a response.

    This function reads a query text from the command line, searches for relevant documents in a Chroma database,
    generates a prompt based on the search results and the query text, and uses a ChatOpenAI model to predict a response.

    Returns:
        None

    Notes:
        - This function relies on the presence of a Chroma database persisted at the location specified by CHROMA_PATH.
        - It uses an embedding function (OpenAIEmbeddings) to process document embeddings.
        - The search is performed using the similarity_search_with_relevance_scores method of the Chroma database.
        - If no matching results are found or the relevance score of the top result is below a threshold (0.7), the function prints a message and returns.
        - It constructs a prompt template using a predefined template (PROMPT_TEMPLATE) and fills it with the context text and the query text.
        - The ChatOpenAI model is used to generate a response based on the constructed prompt.
        - The response, along with the sources of the relevant documents, is printed to the console.

    Example:
        To run the main function from the command line:
        ```
        python script.py "query_text"
    """
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    #print(response_text)

if __name__ == "__main__":
    main()
