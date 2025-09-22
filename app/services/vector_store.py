# In app/services/vector_store.py

from typing import List
from app.utils.logger import logger

class EnhancedHybridVectorStore:
    """Enhanced hybrid vector storage that receives initialized models."""
    
    def __init__(self, embedding_model, pinecone_index):
        # Store the provided models as instance attributes
        self.embedding_model = embedding_model
        self.pinecone_index = pinecone_index
        self.namespace = "insurance_docs"

    def search(self, query: str, document_id: str, limit: int = 15) -> List[str]:
        """Primary Pinecone search with document filtering."""
        try:
            # Use the instance's embedding model
            query_embedding = self.embedding_model.encode([query])[0]
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=self.namespace,
                filter={"document_id": {"$eq": document_id}},
                include_metadata=True
            )
            
            documents = [match.metadata.get("text", "") for match in results.matches if "text" in match.metadata]
            logger.info(f" Search found {len(documents)} chunks for document {document_id}")
            return documents
        except Exception as e:
            logger.error(f" Search failed: {e}")
            return []

    def add_to_pinecone_fallback(self, chunks: List[str], document_id: str):
        """Add chunks to Pinecone with document_id in metadata."""
        try:
            # Use the instance's embedding model
            embeddings = self.embedding_model.encode(chunks)
            batch_size = 20
            
            for batch_idx in range(0, len(chunks), batch_size):
                batch_end = min(batch_idx + batch_size, len(chunks))
                batch_chunks = chunks[batch_idx:batch_end]
                batch_embeddings = embeddings[batch_idx:batch_end]
                
                vectors = []
                for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    vectors.append({
                        "id": f"chunk_{document_id}_{batch_idx + i}",
                        "values": embedding if isinstance(embedding, list) else embedding.tolist(),
                        "metadata": {
                            "text": chunk,
                            "chunk_id": batch_idx + i,
                            "text_length": len(chunk),
                            "document_id": document_id
                        }
                    })
                
                self.pinecone_index.upsert(vectors=vectors, namespace=self.namespace)
            
            logger.info(f" Added {len(chunks)} chunks to Pinecone for document {document_id}")
        except Exception as e:
            logger.error(f" Failed to add to Pinecone fallback: {e}")
            # Re-raise the exception to be caught by the endpoint

            raise e

