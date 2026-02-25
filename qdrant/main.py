from qdrant_client import QdrantClient, models
import os


client = QdrantClient(
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
)

if __name__ == "__main__":

    collection_name = "test_collection"

    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=4,
                distance=models.Distance.COSINE
            )
        )

    collections = client.get_collections()
   # print(collections)

    points = [
        models.PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"category": "example1"}
        ),
        models.PointStruct(
            id=2,
            vector=[0.5, 0.6, 0.7, 0.8],
            payload={"category": "example"}
        )
    ]

    client.upsert(
        collection_name=collection_name,   
        points=points
        )
    

    collection_infor = client.get_collection(collection_name)
   # print(collection_infor)

    query_vector = [0.08, 0.14, 0.33, 0.28]

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=1
    )
    print(search_result)

    client.delete_collection(collection_name)
