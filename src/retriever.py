"""
RAG Retriever

Handles searching ChromaDB for relevant context from Stefan's reviews.
"""

from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from . import config


class CopyChiefRetriever:
    """Retrieves relevant context from Stefan's past reviews."""

    def __init__(self):
        """Initialize the retriever with ChromaDB connection."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.COLLECTION_NAME
            )
            self.is_ready = True
        except:
            self.collection = None
            self.is_ready = False

        # Initialize OpenAI for embeddings
        if config.OPENAI_API_KEY and OPENAI_AVAILABLE:
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.openai_client = None

    def get_stats(self) -> Dict:
        """Get statistics about the indexed data."""
        if not self.is_ready:
            return {"status": "not_initialized", "count": 0}

        count = self.collection.count()
        return {
            "status": "ready",
            "count": count,
            "collection_name": config.COLLECTION_NAME,
        }

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for query text."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        response = self.openai_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text[:8000]  # Truncate if too long
        )
        return response.data[0].embedding

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        niche_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The copy or question to find context for
            top_k: Number of results to return
            niche_filter: Optional niche to filter by

        Returns:
            List of relevant chunks with metadata
        """
        if not self.is_ready:
            return []

        top_k = top_k or config.TOP_K_RESULTS

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Build where filter if niche specified
        where_filter = None
        if niche_filter:
            where_filter = {"niche": niche_filter}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                })

        return formatted_results

    def retrieve_diverse(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Retrieve diverse results by querying multiple ways.

        Returns results from:
        1. Direct semantic similarity
        2. Feedback-focused query
        3. Hook-focused query
        """
        if not self.is_ready:
            return []

        top_k = top_k or config.TOP_K_RESULTS
        per_query_k = max(3, top_k // 3)

        all_results = []
        seen_ids = set()

        # Query 1: Direct similarity to the copy
        results1 = self.retrieve(query[:2000], top_k=per_query_k)
        for r in results1:
            chunk_id = f"{r['metadata'].get('filename', '')}_{r['metadata'].get('chunk_index', 0)}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_results.append(r)

        # Query 2: Look for hook-related feedback
        hook_query = f"hook lead feedback critique {query[:500]}"
        results2 = self.retrieve(hook_query, top_k=per_query_k)
        for r in results2:
            chunk_id = f"{r['metadata'].get('filename', '')}_{r['metadata'].get('chunk_index', 0)}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_results.append(r)

        # Query 3: Look for rewrite examples
        rewrite_query = f"rewrite suggestion improvement {query[:500]}"
        results3 = self.retrieve(rewrite_query, top_k=per_query_k)
        for r in results3:
            chunk_id = f"{r['metadata'].get('filename', '')}_{r['metadata'].get('chunk_index', 0)}"
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_results.append(r)

        # Sort by relevance (lower distance = more relevant)
        all_results.sort(key=lambda x: x.get('distance', 999))

        return all_results[:top_k]

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into a context string for the LLM."""
        if not results:
            return "No relevant past reviews found."

        context_parts = []

        for i, result in enumerate(results, 1):
            filename = result['metadata'].get('filename', 'Unknown')
            niche = result['metadata'].get('niche', 'general')

            context_parts.append(
                f"--- Example {i} (from: {filename}, niche: {niche}) ---\n"
                f"{result['text']}\n"
            )

        return "\n".join(context_parts)


# Singleton instance
_retriever = None


def get_retriever() -> CopyChiefRetriever:
    """Get or create the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = CopyChiefRetriever()
    return _retriever
