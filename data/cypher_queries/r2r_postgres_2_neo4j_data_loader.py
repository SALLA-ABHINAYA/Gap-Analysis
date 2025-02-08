import os
import csv
import json
from sshtunnel import SSHTunnelForwarder
import psycopg2
import select
from neo4j import GraphDatabase
import datetime
import decimal
import uuid
import collections

def convert_to_primitive(value):
    """
    Convert non-primitive values to primitive types, ensuring all values are serializable.
    """
    if isinstance(value, list):
        # Recursively convert items in the list to primitives
        return [convert_to_primitive(i) for i in value]
    elif isinstance(value, dict):
        # Recursively convert dictionary values to primitives
        return {k: convert_to_primitive(v) for k, v in value.items()}
    elif isinstance(value, tuple):
        # Convert tuple to list (since tuples are immutable and Neo4j doesn't support them)
        return [convert_to_primitive(i) for i in value]
    elif isinstance(value, set):
        # Convert set to list (since sets are unordered and Neo4j doesn't support them)
        return [convert_to_primitive(i) for i in value]
    elif isinstance(value, frozenset):
        # Convert frozenset to list (since frozensets are immutable sets)
        return [convert_to_primitive(i) for i in value]
    elif isinstance(value, datetime.datetime):
        # Convert datetime objects to ISO 8601 string format
        return value.isoformat()
    elif isinstance(value, datetime.date):
        # Convert date objects to string format
        return value.isoformat()
    elif isinstance(value, decimal.Decimal):
        # Convert Decimal objects to float
        return float(value)
    elif isinstance(value, uuid.UUID):
        # Convert UUID objects to string format
        return str(value)
    elif isinstance(value, bool):
        # Boolean values are already primitive
        return value
    elif isinstance(value, (str, int, float)):
        # Return primitive types as is
        return value
    elif isinstance(value, collections.Counter):
        # Convert Counter objects to dict
        return dict(value)
    else:
        # For unsupported types, return a string representation
        return str(value)

def process_notification(payload, table_name="your_table"):
    """
    Process a notification payload dynamically and store it in a CSV file.
    Also upload the data to Neo4j.
    """
    try:
        # Parse the JSON payload
        data = json.loads(payload)
        
        # Get the current timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique file name based on the table name and timestamp
        file_name = f"{table_name}_{current_time}.csv"
        
        # Get the keys from the JSON data dynamically
        headers = data.keys()
        
        # Check if the file exists
        file_exists = os.path.isfile(file_name)
        
        # Open the file in append mode and store the data
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            
            # Write the header only if the file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the row
            writer.writerow(data)
        
        print(f"Data from {table_name} successfully stored in {file_name}")
        
        # Upload data to Neo4j
        upload_to_neo4j(data, table_name)
    
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON payload: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def upload_to_neo4j(data, table_name):
    """
    Upload data from the CSV to Neo4j
    """
    # Neo4j connection details
    uri ="neo4j+s://7d5da6ec.databases.neo4j.io"  # Neo4j Bolt URI
    user = "neo4j"
    password = "_W9giWFM0JKPH4p50yyg7jmXkYkYqPbf4zZPeM7CnVI"  # Replace with your Neo4j password
    
    # Establish connection to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Dynamically construct the query based on the data
        query = f"""
        MERGE (n:{table_name} {{id: $id}})
        """
        
        # Add properties dynamically
        parameters = {"id": data["id"]}  # Assuming 'id' is unique and used to match the node
        
        # Loop through the data and add properties individually
        for key, value in data.items():
            if key != "id":  # We assume 'id' is the unique identifier
                # For lists (arrays), make sure they are arrays of primitives
                if isinstance(value, list):
                    # Only accept primitive types in the list (strings, integers, booleans)
                    if all(isinstance(i, (str, int, bool)) for i in value):
                        parameters[key] = value  # Array of primitives can be passed directly
                    else:
                        print(f"Skipping non-primitive values in array for key: {key}")
                else:
                    # For primitive types (string, integer, boolean), assign directly
                    if isinstance(value, (str, int, bool)):
                        parameters[key] = value
                    else:
                        parameters[key] = convert_to_primitive(value)
        
        # Run the query with dynamic properties
        session.run(query, parameters)

        print(f"Data uploaded to Neo4j Aura for table {table_name}")

def listen_to_notifications():
    # SSH and PostgreSQL connection details
    ssh_host = "20.68.198.68"
    ssh_user = "azureuser"
    ssh_pem_file = "r2r_key.pem"
    remote_db_host = "localhost"  # Use 'localhost' if the database is on the SSH server
    db_name = "postgres"
    db_user = "postgres"
    db_port = 5432  # PostgreSQL default port

    # Establish the SSH tunnel
    with SSHTunnelForwarder(
        ssh_address_or_host=(ssh_host, 22),
        ssh_username=ssh_user,
        ssh_private_key=ssh_pem_file,
        remote_bind_address=(remote_db_host, db_port),
        local_bind_address=('127.0.0.1', 6543)  # Forward the remote port to a local port
    ) as tunnel:
        print(f"SSH Tunnel established on local port {tunnel.local_bind_port}")

        # Connect to PostgreSQL via the SSH tunnel
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            host=tunnel.local_bind_host,
            port=tunnel.local_bind_port,
            password="postgres"
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Listen to the channels for both tables
        cur.execute("LISTEN new_entry;")
        cur.execute("LISTEN new_entry_2;")
        print("Listening to notifications...")

        try:
            while True:
                if select.select([conn], [], [], 5) == ([conn], [], []):
                    conn.poll()
                    while conn.notifies:
                        notify = conn.notifies.pop(0)
                        
                        # Process notifications based on the channel
                        if notify.channel == "new_entry":
                            process_notification(notify.payload, table_name="your_table")
                        elif notify.channel == "new_entry_2":
                            process_notification(notify.payload, table_name="your_table_2")
        except KeyboardInterrupt:
            print("Stopped listening.")
        finally:
            cur.close()
            conn.close()

if __name__ == "__main__":
    listen_to_notifications()
