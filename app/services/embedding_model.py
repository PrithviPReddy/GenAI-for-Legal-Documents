# app/services/embedding_model.py
import os
from openai import OpenAI

class OpenAIEmbeddingModel:
    """
    Wrapper around OpenAI embeddings API to mimic SentenceTransformer.encode()
    """
    def __init__(self, model_name="text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def encode(self, texts):
        """
        Accepts a string or a list of strings, returns list of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        return embeddings
