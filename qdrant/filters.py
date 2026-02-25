from qdrant_client import QdrantClient, models
import os

client = QdrantClient(
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
)   

if __name__ == "__main__":

    collection_name = "test1_collection"


    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=4,
                distance=models.Distance.COSINE)
        )

        client.create_payload_index(
            collection_name= collection_name,
            field_name="category",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    
    points=[
    models.PointStruct(
        id=1,
        vector=[0.9, 0.1, 0.1, 0.8], # High affordability, high innovation
        payload={"name": "Budget Smartphone", "category": "electronics", "price": 299},
    ),
    models.PointStruct(
        id=2,
        vector=[0.2, 0.9, 0.8, 0.5], # High quality, high popularity
        payload={"name": "Bestselling Novel", "category": "books", "price": 19},
    ),
    models.PointStruct(
        id=3,
        vector=[0.8, 0.3, 0.2, 0.9], # High affordability, high innovation (similar to ID 1)
        payload={"name": "Smart Home Hub", "category": "electronics", "price": 89},
    ),
    # Add 2-5 more points to experiment with...
]

    client.upsert(collection_name=collection_name, points=points)

    query_vector = [0.85, 0.2, 0.1, 0.9]

    basic_results = client.query_points(collection_name, query=query_vector)
    print("Basic Search Results:")

    filtered_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value="books")
                )
            ]
        )
    )
    print(f"Filtered Search Results (category: electronics): {filtered_results}")
