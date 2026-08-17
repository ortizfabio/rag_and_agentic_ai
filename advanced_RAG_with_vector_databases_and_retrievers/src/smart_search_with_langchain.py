import logging

from langchain.chains.query_constructor.base import AttributeInfo, StructuredQueryOutputParser
from langchain.retrievers import SelfQueryRetriever, ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_llm import llm, model_embedding

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)




def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def email_policy(vector_db: Chroma):
    print("#" * 20 + "email_policy" + "#" * 20)
    query = "email policy"
    retriever = vector_db.as_retriever()
    email_policy_docs = retriever.invoke(query)
    print(email_policy_docs)

    retriever = vector_db.as_retriever(search_kwargs={"k": 1})
    email_policy_docs = retriever.invoke(query)
    print(email_policy_docs)
    # MMR Search
    retriever = vector_db.as_retriever(search_type="mmr")
    email_policy_docs = retriever.invoke(query)
    print(email_policy_docs)

    # Similarity score threshold retrieval
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
    )
    email_policy_docs = retriever.invoke(query)
    print(email_policy_docs)


def langchain_paper_search(vector_db: Chroma):
    print("#" * 20 + "langchain_paper_search" + "#" * 20)
    loader = PyPDFLoader(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf")
    # loader = PyPDFLoader("/home/db2admin/dev/coursera/advanced_RAG_with_vector_databases_and_retrievers/langchain_paper.pdf")
    pdf_data = loader.load()
    print(pdf_data[1])

    # Split
    chunks_pdf = text_splitter(pdf_data, 500, 20)

    # VectorDB
    ids = vector_db.get()["ids"]
    vector_db.delete(
        ids)  # We need to delete existing embeddings from previous documents and then store current document embeddings in.
    vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=model_embedding())

    query = "What does the paper say about langchain?"

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm()
    )

    lang_chain_docs = retriever.invoke(query)
    print(lang_chain_docs)


def book_search():
    print("#" * 20 + "book_search" + "#" * 20)
    docs = [
        Document(
            page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone, three men walk out of the Zone",
            metadata={
                "year": 1979,
                "director": "Andrei Tarkovsky",
                "genre": "thriller",
                "rating": 9.9,
            },
        ),
    ]

    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="float"
        ),
    ]
    chroma_db = Chroma.from_documents(docs, model_embedding())
    document_content_description = "Brief summary of a movie."
    # 1. Create the default query parser, enabling autofix
    output_parser = StructuredQueryOutputParser.from_components(fix_invalid=True)

    retriever = SelfQueryRetriever.from_llm(
        llm(),
        chroma_db,
        document_content_description,
        metadata_field_info,
        query_constructor_parser=output_parser,  # Forces the retriever to fix bad JSON formats
    )
    try:
        # This example only specifies a filter
        retriever.invoke("I want to watch a movie rated higher than 8.5")
    except Exception as e:
        print(e)
    try:
        # This example specifies a query and a filter
        retriever.invoke("Has Greta Gerwig directed any movies about women")
    except Exception as e:
        print(e)
    try:
        # This example specifies a composite filter
        retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
    except Exception as e:
        print(e)
    # This example specifies a query with filter# You might encouter some errors or blank content when run the
    # following code. It is becasue LLM cannot get the answer at first. Don't worry, re-run it several times
    # you will get the answer.
    vectordb = Chroma.from_documents(docs, model_embedding())
    retriever = SelfQueryRetriever.from_llm(
        llm(),
        vectordb,
        document_content_description,
        metadata_field_info,
    )

    # This example specifies a query with filter
    retriever.invoke(
        "I want to watch a movie directed by Christopher Nolan"
    )


def smoking_policy_search(txt_data):
    print("#" * 20 + "smoking_policy_search" + "#" * 20)
    # Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
    parent_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
    child_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='\n')
    vectordb = Chroma(
        collection_name="split_parents", embedding_function=model_embedding()
    )
    # vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(txt_data)
    # These are the number of large chunks:
    len(list(store.yield_keys()))
    sub_docs = vectordb.similarity_search("smoking policy")
    print(sub_docs[0].page_content)

    retrieved_docs = retriever.invoke("smoking policy")
    print(retrieved_docs[0].page_content)
    chunks_txt = text_splitter(txt_data, 200, 20)
    vectordb = Chroma.from_documents(documents=chunks_txt, embedding=model_embedding())
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    query = "smoking policy"
    docs = retriever.invoke(query)
    print(docs)


def main():
    # Download the data
    # !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt"

    txt_loader = TextLoader("companypolicies.txt")
    txt_data = txt_loader.load()

    print(txt_data[0])
    print(len(txt_data))

    chunks_txt = text_splitter(txt_data, 200, 20)

    vector_db: Chroma = Chroma.from_documents(chunks_txt, model_embedding())
    try:
        email_policy(vector_db)
        print("#" * 20 + "email_policy end " + "#" * 20)
    except Exception as e:
        print(f"Exception in email_policy function: {e}")
    try:
        langchain_paper_search(vector_db)
        print("#" * 20 + "langchain_paper_search end " + "#" * 20)
    except Exception as e:
        print(f"Exception in langchain_paper_search function: {e}")
    try:
        book_search()
        print("#" * 20 + "book_search end " + "#" * 20)
    except Exception as e:
        print(f"Exception in book_search function: {e}")
    try:
        smoking_policy_search(txt_data)
        print("#" * 20 + "smoking_policy_search end " + "#" * 20)
    except Exception as e:
        print(f"Exception in smoking_policy_search function: {e}")


if __name__ == "__main__":
    main()
