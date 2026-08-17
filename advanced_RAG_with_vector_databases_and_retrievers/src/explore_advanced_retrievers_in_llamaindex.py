# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)
# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

from llamaindex_llm import model_embedding, simple_llm

# Statistical libraries for fusion techniques
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some advanced fusion features will be limited")

# Sample data for the lab - AI/ML focused documents
SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout the lab
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning",
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}


class AdvancedRetrieversLab():
    def __init__(self):
        print("🚀 Initializing Advanced Retrievers Lab...")
        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes = SentenceSplitter().get_nodes_from_documents(self.documents, embed_model=model_embedding())

        print("📊 Creating indexes...")
        # Create various indexes
        self.vector_index = VectorStoreIndex.from_documents(self.documents, embed_model=model_embedding())

        # 2. Bind your local components globally to Settings
        Settings.llm = simple_llm()
        Settings.embed_model = model_embedding()
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)

        self.keyword_index = KeywordTableIndex.from_documents(self.documents, embed_model=model_embedding())

        print("✅ Advanced Retrievers Lab Initialized!")
        print(f"📄 Loaded {len(self.documents)} documents")
        print(f"🔢 Created {len(self.nodes)} nodes")


def search_with_vector_index(lab: AdvancedRetrieversLab):
    print("=" * 60)
    print("1. VECTOR INDEX RETRIEVER")
    print("=" * 60)

    # Basic vector retriever
    vector_retriever = VectorIndexRetriever(
        index=lab.vector_index,
        similarity_top_k=3
    )

    # Alternative creation method
    alt_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

    query = DEMO_QUERIES["basic"]  # "What is machine learning?"
    nodes = vector_retriever.retrieve(query)

    print(f"Query: {query}")
    print(f"Retrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()


def search_with_vector_index_retriever(lab: AdvancedRetrieversLab):
    print("=" * 60)
    print("1. VECTOR INDEX RETRIEVER")
    print("=" * 60)

    # Basic vector retriever
    vector_retriever = VectorIndexRetriever(
        index=lab.vector_index,
        similarity_top_k=3
    )

    # Alternative creation method
    alt_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

    query = DEMO_QUERIES["basic"]  # "What is machine learning?"
    nodes = vector_retriever.retrieve(query)

    print(f"Query: {query}")
    print(f"Retrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()


def search_with_bm25_retriever(lab: AdvancedRetrieversLab):
    print("=" * 60)
    print("2. BM25 RETRIEVER")
    print("=" * 60)

    try:
        import Stemmer

        # Create BM25 retriever with default parameters
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=lab.nodes,
            similarity_top_k=3,
            stemmer=Stemmer.Stemmer("english"),
            language="english"
        )

        query = DEMO_QUERIES["technical"]  # "neural networks deep learning"
        nodes = bm25_retriever.retrieve(query)

        print(f"Query: {query}")
        print("BM25 analyzes exact keyword matches with sophisticated scoring")
        print(f"Retrieved {len(nodes)} nodes:")

        for i, node in enumerate(nodes, 1):
            score = node.score if hasattr(node, 'score') and node.score else 0
            print(f"{i}. BM25 Score: {score:.4f}")
            print(f"   Text: {node.text[:100]}...")

            # Highlight which query terms appear in the text
            text_lower = node.text.lower()
            query_terms = query.lower().split()
            found_terms = [term for term in query_terms if term in text_lower]
            if found_terms:
                print(f"   → Found terms: {found_terms}")
            print()

        print("BM25 vs TF-IDF Comparison:")
        print("TF-IDF Problem: Linear term frequency scaling")
        print("  Example: 10 occurrences → score of 10, 100 occurrences → score of 100")
        print("BM25 Solution: Saturation function")
        print("  Example: 10 occurrences → high score, 100 occurrences → slightly higher score")
        print()
        print("TF-IDF Problem: No document length consideration")
        print("  Example: Long documents dominate results")
        print("BM25 Solution: Length normalization (b parameter)")
        print("  Example: Scores adjusted based on document length vs. average")
        print()
        print("Key BM25 Parameters:")
        print("- k1 ≈ 1.2: Term frequency saturation (how quickly scores plateau)")
        print("- b ≈ 0.75: Document length normalization (0=none, 1=full)")
        print("- IDF weighting: Rare terms get higher scores")

    except ImportError:
        print("⚠️ BM25Retriever requires 'pip install PyStemmer'")
        print("Demonstrating BM25 concepts with fallback vector search...")

        fallback_retriever = lab.vector_index.as_retriever(similarity_top_k=3)
        query = DEMO_QUERIES["technical"]
        nodes = fallback_retriever.retrieve(query)

        print(f"Query: {query}")
        print("(Using vector fallback to demonstrate BM25 concepts)")

        for i, node in enumerate(nodes, 1):
            print(f"{i}. Vector Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")

            # Demonstrate TF-IDF concept manually
            text_lower = node.text.lower()
            query_terms = query.lower().split()
            found_terms = [term for term in query_terms if term in text_lower]

            if found_terms:
                print(f"   → BM25 would boost this result for terms: {found_terms}")
            print()

        print("BM25 Concept Demonstration:")
        print("1. TF-IDF Foundation:")
        print("   - Term Frequency: How often words appear in document")
        print("   - Inverse Document Frequency: How rare words are across collection")
        print("   - TF-IDF = TF × IDF (balances frequency vs rarity)")
        print()
        print("2. BM25 Improvements:")
        print("   - Saturation: Prevents over-scoring repeated terms")
        print("   - Length normalization: Prevents long document bias")
        print("   - Tunable parameters: k1 (saturation) and b (length adjustment)")
        print()
        print("3. Real-world Usage:")
        print("   - Elasticsearch default scoring function")
        print("   - Apache Lucene/Solr standard")
        print("   - Used in 83% of text-based recommender systems")
        print("   - Developed by Robertson & Spärck Jones at City University London")


def search_with_document_retriever_index(lab: AdvancedRetrieversLab):
    print("=" * 60)
    print("3. DOCUMENT SUMMARY INDEX RETRIEVERS")
    print("=" * 60)

    # LLM-based document summary retriever
    doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
        lab.document_summary_index,
        choice_top_k=3  # Number of documents to select
    )

    # Embedding-based document summary retriever
    doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
        lab.document_summary_index,
        similarity_top_k=3  # Number of documents to select
    )

    query = DEMO_QUERIES["learning_types"]  # "different types of learning"

    print(f"Query: {query}")

    print("\nA) LLM-based Document Summary Retriever:")
    print("Uses LLM to select relevant documents based on summaries")
    try:
        nodes_llm = doc_summary_retriever_llm.retrieve(query)
        print(f"Retrieved {len(nodes_llm)} nodes")
        for i, node in enumerate(nodes_llm[:2], 1):
            print(f"{i}. Score: {node.score:.4f}" if hasattr(node,
                                                             'score') and node.score else f"{i}. (Document summary)")
            print(f"   Text: {node.text[:80]}...")
            print()
    except Exception as e:
        print(f"LLM-based retrieval demo: {str(e)[:100]}...")

    print("B) Embedding-based Document Summary Retriever:")
    print("Uses vector similarity between query and document summaries")
    try:
        nodes_emb = doc_summary_retriever_embedding.retrieve(query)
        print(f"Retrieved {len(nodes_emb)} nodes")
        for i, node in enumerate(nodes_emb[:2], 1):
            print(f"{i}. Score: {node.score:.4f}" if hasattr(node,
                                                             'score') and node.score else f"{i}. (Document summary)")
            print(f"   Text: {node.text[:80]}...")
            print()
    except Exception as e:
        print(f"Embedding-based retrieval demo: {str(e)[:100]}...")

    print("Document Summary Index workflow:")
    print("1. Generates summaries for each document using LLM")
    print("2. Uses summaries to select relevant documents")
    print("3. Returns full content from selected documents")


def search_with_recursive_retriever(lab: AdvancedRetrieversLab):
    print("=" * 60)
    print("5. RECURSIVE RETRIEVER")
    print("=" * 60)

    # Create documents with references
    docs_with_refs = []
    for i, doc in enumerate(lab.documents):
        # Add reference metadata
        ref_doc = Document(
            text=doc.text,
            metadata={
                "doc_id": f"doc_{i}",
                "references": [f"doc_{j}" for j in range(len(lab.documents)) if j != i][:2]
            }
        )
        docs_with_refs.append(ref_doc)

    # Create index with referenced documents
    ref_index = VectorStoreIndex.from_documents(docs_with_refs, embed_model=model_embedding())

    # Create retriever mapping
    retriever_dict = {
        f"doc_{i}": ref_index.as_retriever(similarity_top_k=1)
        for i in range(len(docs_with_refs))
    }

    # Base retriever
    base_retriever = ref_index.as_retriever(similarity_top_k=2)

    # Add the root retriever to the dictionary
    retriever_dict["vector"] = base_retriever

    # Recursive retriever
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict=retriever_dict,
        query_engine_dict={},
        verbose=True
    )

    query = DEMO_QUERIES["applications"]  # "What are the applications of AI?"
    try:
        nodes = recursive_retriever.retrieve(query)
        print(f"Query: {query}")
        print(f"Recursively retrieved {len(nodes)} nodes")
        for i, node in enumerate(nodes[:3], 1):
            print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Recursive)")
            print(f"   Text: {node.text[:100]}...")
            print()
    except Exception as e:
        print(f"Query: {query}")
        print(f"Recursive retriever demo: {str(e)}")
        print("Note: Recursive retriever requires specific node reference setup")

        # Fallback to basic retrieval for demonstration
        print("\nFalling back to basic retrieval demonstration...")
        base_nodes = base_retriever.retrieve(query)
        for i, node in enumerate(base_nodes[:2], 1):
            print(f"{i}. Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()


def search_with_recursive_retriever_index(lab: AdvancedRetrieversLab):
    print("=" * 60)
    print("6.1 RECIPROCAL RANK FUSION MODE DEMONSTRATION")
    print("=" * 60)

    # Create QueryFusionRetriever with RRF mode
    base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

    print("Testing QueryFusionRetriever with reciprocal_rerank mode:")
    print("This demonstrates how RRF works within the query fusion framework")

    # Use the same query for consistency across all fusion modes
    query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

    try:
        # Create query fusion retriever with RRF mode
        rrf_query_fusion = QueryFusionRetriever(
            [base_retriever],
            similarity_top_k=3,
            num_queries=3,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True
        )

        print(f"\nQuery: {query}")
        print("QueryFusionRetriever will:")
        print("1. Generate query variations using LLM")
        print("2. Retrieve results for each variation")
        print("3. Apply Reciprocal Rank Fusion")

        nodes = rrf_query_fusion.retrieve(query)

        print(f"\nRRF Query Fusion Results:")
        for i, node in enumerate(nodes, 1):
            print(f"{i}. Final RRF Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()

        print("RRF Benefits in Query Fusion Context:")
        print("- Automatically handles query variations of different quality")
        print("- No bias toward queries that return higher raw scores")
        print("- Stable performance across diverse query formulations")

    except Exception as e:
        print(f"QueryFusionRetriever error: {e}")
        print("Demonstrating RRF concept manually with query variations...")

        # Manual demonstration with query variations derived from the main query
        query_variations = [
            DEMO_QUERIES["comprehensive"],  # Original query
            "machine learning approaches and methods",
            "different ML techniques and algorithms"
        ]

        print("Manual RRF with Query Variations:")
        all_results = {}

        for i, query_var in enumerate(query_variations):
            print(f"\nQuery variation {i + 1}: {query_var}")
            nodes = base_retriever.retrieve(query_var)

            # Apply RRF scoring
            for rank, node in enumerate(nodes):
                node_id = node.node.node_id
                if node_id not in all_results:
                    all_results[node_id] = {
                        'node': node,
                        'rrf_score': 0,
                        'query_ranks': []
                    }

                # Calculate RRF contribution: 1 / (rank + k)
                k = 60  # Standard RRF parameter
                rrf_contribution = 1.0 / (rank + 1 + k)
                all_results[node_id]['rrf_score'] += rrf_contribution
                all_results[node_id]['query_ranks'].append((i, rank + 1))

        # Sort by final RRF score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        print(f"\nCombined RRF Results (top 3):")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. Final RRF Score: {result['rrf_score']:.4f}")
            print(f"   Query ranks: {result['query_ranks']}")
            print(f"   Text: {result['node'].text[:100]}...")
            print()

        print("RRF Formula Demonstration:")
        print("For each document: RRF_score = Σ(1 / (rank + 60))")
        print("- Rank 1 in query: 1/(1+60) = 0.0164")
        print("- Rank 2 in query: 1/(2+60) = 0.0161")
        print("- Rank 3 in query: 1/(3+60) = 0.0159")
        print("Documents appearing in multiple queries get higher combined scores")


def main():
    print(f"📄 Loaded {len(SAMPLE_DOCUMENTS)} sample documents")
    print(f"🔍 Prepared {len(DEMO_QUERIES)} consistent demo queries")
    for i, doc in enumerate(SAMPLE_DOCUMENTS[:3], 1):
        print(f"{i}. {doc}")
    print("...")
    lab = AdvancedRetrieversLab()
    try:
        search_with_vector_index(lab)
    except Exception as e:
        print(f"Error occurred while searching with vector index: {e}")
    try:
        search_with_vector_index_retriever(lab)
    except Exception as e:
        print(f"Error occurred while searching with vector index retriever: {e}")
    try:
        search_with_bm25_retriever(lab)
    except Exception as e:
        print(f"Error occurred while searching with BM25 retriever: {e}")

    try:
        search_with_document_retriever_index(lab)
    except Exception as e:
        print(f"Error occurred while searching with document retriever index: {e}")


if __name__ == "__main__":
    main()
