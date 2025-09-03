from typing import List
from app.utils.logger import logger
from app.config import embedding_model, pinecone_index


class EnhancedHybridVectorStore:
    """Enhanced hybrid vector storage with improved search strategies"""
    
    def __init__(self):
        self.namespace = "insurance_docs"

    def search_pinecone_enhanced(self, query: str, document_id: str, limit: int = 10) -> List[str]:
        """Enhanced Pinecone search with metadata filtering"""
        try:
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=limit,
                namespace=self.namespace,
                filter={"document_id": {"$eq": document_id}},
                include_metadata=True
            )
            
            documents = [match.metadata.get("text", "") for match in results.matches if "text" in match.metadata]
            logger.info(f" Pinecone search found {len(documents)} chunks for query '{query[:50]}...'")
            return documents
            
        except Exception as e:
            logger.error(f" Pinecone search failed: {e}")
            return []

    def generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        variations = []
        key_terms = self.extract_key_terms(query)
        
        if key_terms:
            # Full phrase with key terms
            variations.append(" ".join(key_terms))
            
            # Individual key terms (top 2 only)
            for term in key_terms[:2]:
                variations.append(term)
        
        return variations
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        import re
        
        stop_words = {
            "what", "is", "the", "how", "does", "are", "and", "or", 
            "but", "in", "on", "at", "to", "for", "of", "with", "by"
        }
        
        words = re.findall(r"\b[A-Za-z]+\b", query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        priority_terms = [
            "constitution", "article", "amendment", "rights", "fundamental",
            "directive", "principles", "president", "supreme", "court",
            "parliament", "state", "emergency"
        ]
        
        prioritized = [term for term in priority_terms if term in key_terms]
        
        for term in key_terms:
            if term not in prioritized:
                prioritized.append(term)
        
        return prioritized[:5]  # Limit to 5 terms
    
    def search(self, query: str, document_id: str, limit: int = 15) -> List[str]:
        """Primary Pinecone search with document filtering"""
        try:
            query_embedding = embedding_model.encode([query])[0].tolist()
            
            results = pinecone_index.query(
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
        """Add chunks to Pinecone with document_id in metadata"""
        try:
            embeddings = embedding_model.encode(chunks)
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
                            "document_id": document_id  # CRITICAL FIX
                        }
                    })
                
                pinecone_index.upsert(vectors=vectors, namespace=self.namespace)
            
            logger.info(f" Added {len(chunks)} chunks to Pinecone for document {document_id}")
        except Exception as e:
            logger.error(f" Failed to add to Pinecone fallback: {e}")

