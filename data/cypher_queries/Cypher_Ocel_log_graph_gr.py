from neo4j import GraphDatabase
import json

# Neo4j AuraDB connection details
URI = "neo4j+s://1bcd5ab7.databases.neo4j.io"  # Replace with your Neo4j Aura connection string
AUTH = ("neo4j", "mfk00LT6txYy1Szvv_lxpfg_UvlG5A5D921WHeWCaX0")  # Replace with your credentials

# Load JSON file
with open("process_data.json", "r") as file:
    json_data = json.load(file)

# Batch size to insert nodes efficiently
BATCH_SIZE = 10000


# Function to execute queries in batches
def execute_batch(driver, query, batch):
    if batch:
        with driver.session() as session:
            session.write_transaction(lambda tx: [tx.run(query, parameters=record) for record in batch])


# Connect to Neo4j
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # **Step 1: Drop constraints and indexes**
    constraint_queries = [
        "DROP CONSTRAINT event_id_unique IF EXISTS;",
        "DROP CONSTRAINT trade_case_unique IF EXISTS;",
        "DROP INDEX activity_name IF EXISTS;",

        # Create constraints and indexes
        "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (n:Event) REQUIRE n.ocel_id IS UNIQUE;",
        "CREATE CONSTRAINT trade_case_unique IF NOT EXISTS FOR (n:TradeCase) REQUIRE n.case_id IS UNIQUE;",
        "CREATE INDEX activity_name IF NOT EXISTS FOR (n:Activity) ON (n.name);"
    ]

    for query in constraint_queries:
        with driver.session() as session:
            session.run(query)

    print("Constraints and indexes created successfully.")

    # **Step 2: Insert Event Nodes in Batches**
    event_query = """
        MERGE (e:Event {ocel_id: $id})
        SET e.timestamp = datetime($timestamp),
            e.activity_name = $activity
    """

    event_batch = []
    for event in json_data['ocel:events']:
        event_batch.append({
            "id": event['ocel:id'],
            "timestamp": event['ocel:timestamp'],
            "activity": event['ocel:activity']
        })
        if len(event_batch) >= BATCH_SIZE:
            execute_batch(driver, event_query, event_batch)
            event_batch = []
    execute_batch(driver, event_query, event_batch)  # Insert remaining records

    print("Event nodes inserted successfully.")

    # **Step 3: Insert TradeCase Nodes in Batches**
    trade_case_query = """
        MATCH (e:Event {ocel_id: $id})
        MERGE (c:TradeCase {case_id: $case_id})
        MERGE (c)-[:CONTAINS_EVENT]->(e)
    """

    trade_case_batch = []
    for event in json_data['ocel:events']:
        if "case_id" in event["ocel:attributes"]:
            trade_case_batch.append({
                "id": event['ocel:id'],
                "case_id": event['ocel:attributes']['case_id']
            })
        if len(trade_case_batch) >= BATCH_SIZE:
            execute_batch(driver, trade_case_query, trade_case_batch)
            trade_case_batch = []
    execute_batch(driver, trade_case_query, trade_case_batch)

    print("TradeCase nodes inserted successfully.")

    # **Step 4: Insert Activity Nodes in Batches**
    activity_query = """
        MATCH (e:Event {ocel_id: $id})
        MERGE (a:Activity {name: $activity})
        MERGE (e)-[:OF_TYPE]->(a)
    """

    activity_batch = []
    for event in json_data['ocel:events']:
        activity_batch.append({
            "id": event['ocel:id'],
            "activity": event['ocel:activity']
        })
        if len(activity_batch) >= BATCH_SIZE:
            execute_batch(driver, activity_query, activity_batch)
            activity_batch = []
    execute_batch(driver, activity_query, activity_batch)

    print("Activity nodes inserted successfully.")

    # **Step 5: Insert Resource Nodes in Batches**
    resource_query = """
        MATCH (e:Event {ocel_id: $id})
        MERGE (r:Resource {name: $resource})
        MERGE (e)-[:PERFORMED_BY]->(r)
    """

    resource_batch = []
    for event in json_data['ocel:events']:
        if "resource" in event["ocel:attributes"]:
            resource_batch.append({
                "id": event['ocel:id'],
                "resource": event['ocel:attributes']['resource']
            })
        if len(resource_batch) >= BATCH_SIZE:
            execute_batch(driver, resource_query, resource_batch)
            resource_batch = []
    execute_batch(driver, resource_query, resource_batch)

    print("Resource nodes inserted successfully.")

    # **Step 6: Insert TradeObject Nodes in Batches**
    trade_object_query = """
        MATCH (e:Event {ocel_id: $id})
        MERGE (o:TradeObject {id: $obj_id, type: $obj_type})
        MERGE (e)-[:INVOLVES]->(o)
    """

    trade_object_batch = []
    for event in json_data['ocel:events']:
        for obj in event.get('ocel:objects', []):
            trade_object_batch.append({
                "id": event['ocel:id'],
                "obj_id": obj["id"],
                "obj_type": obj["type"]
            })
        if len(trade_object_batch) >= BATCH_SIZE:
            execute_batch(driver, trade_object_query, trade_object_batch)
            trade_object_batch = []
    execute_batch(driver, trade_object_query, trade_object_batch)

    print("TradeObject nodes inserted successfully.")

    # **Step 7: Add timestamp-based sequence relationships in Batches**
    sequence_query = """
        MATCH (e1:Event)
        WITH e1
        ORDER BY e1.timestamp
        WITH collect(e1) as events
        UNWIND range(0, size(events)-2) as i
        WITH events[i] as e1, events[i+1] as e2
        WHERE e1.timestamp < e2.timestamp
        MERGE (e1)-[r:NEXT_EVENT]->(e2)
        SET r.time_difference = duration.between(e1.timestamp, e2.timestamp);
    """

    with driver.session() as session:
        session.run(sequence_query)

    print("Sequence relationships created successfully.")

print("Data successfully uploaded in batches to Neo4j AuraDB!")
