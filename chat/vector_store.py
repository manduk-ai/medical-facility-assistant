"""
File: vector_store.py
Author: Szymon Manduk
Date: 2024-03-05

Description:
    Class for handling the vectore store search.
    Used to search Pincone vector DB against information from the user input.

License:
    To be defined. Contact the author for more information on the terms of use for this software.
"""

from pinecone import Pinecone
import time

class VectorStore:
    """
    Class for handling the vectore store search. Used to search Pincone vector DB against information from the user input.

    """
    def __init__(self, pinecone_api_key, openai_client, index_name, embedding_model):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.embedding_model = embedding_model

        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pinecone_client.Index(self.index_name)
        
        # Initialize OpenAI client
        self.openai_client = openai_client
        
        # time.sleep(1)  # Wait for the index to be ready
        # print(f'Index {self.index_name} ready to use. Statistics:')
        # print(self.index.describe_index_stats())

    def get_embedding(self, text):
        """
        Get the embedding for a given text using OpenAI's embeddings API.
        """
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def query(self, question, condition, top_k=3):
        """
        Search the Pinecone index with the question's embedding and return top k answers matching the condition.
        """
        query_embedding = self.get_embedding(question)
        
        res = self.index.query(
            vector=query_embedding, 
            filter={"type": {"$eq": condition}},
            top_k=top_k, 
            include_metadata=True
        )
        
        answers = [r['metadata']['text'] for r in res['matches']]
        return answers
