"""
Filename: vector_store.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: Class for handling the vectore store search. Used to search Pincone vector DB against information from the user input.
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

from pinecone import Pinecone
import time

class VectorStore:
    """
    Class for handling the vector store search. Used to search Pinecone vector DB against information from the user input.

    Args:
        pinecone_api_key (str): The API key for accessing the Pinecone service.
        openai_client: The client for accessing the OpenAI service.
        index_name (str): The name of the index in the Pinecone service.
        embedding_model (str): The name of the embedding model used by the OpenAI service.

    Attributes:
        pinecone_api_key (str): The API key for accessing the Pinecone service.
        index_name (str): The name of the index in the Pinecone service.
        embedding_model (str): The name of the embedding model used by the OpenAI service.
        pinecone_client: The client for accessing the Pinecone service.
        index: The index in the Pinecone service.
        openai_client: The client for accessing the OpenAI service.
    """

    def __init__(self, pinecone_api_key, openai_client, index_name, embedding_model):
        """
        Initialize a VectorStore object.

        Args:
            pinecone_api_key (str): The API key for Pinecone.
            openai_client: The client object for OpenAI.
            index_name (str): The name of the index in Pinecone.
            embedding_model: The embedding model to be used.

        Returns:
            None
        """
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

        Args:
            text (str): The input text for which the embedding needs to be generated.

        Returns:
            numpy.ndarray: The embedding vector for the input text.
        """
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def query(self, question, condition, top_k=3):
        """
        Search the Pinecone index with the question's embedding and return top k answers matching the condition.

        Parameters:
        - question (str): The question to search for.
        - condition (str): The condition to filter the results by.
        - top_k (int): The number of top answers to return (default: 3).

        Returns:
        - answers (list): A list of top k answers matching the condition.
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
