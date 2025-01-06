from r2r import R2RClient

client = R2RClient("http://20.68.198.68:7273/")
response = client.ingest_files(
    file_paths=["data/graphs_202412310408.csv"],
    collection_ids=["4bd0c140-118a-43f1-81b7-53e8952"]
)
