// First drop constraints and indexes
DROP CONSTRAINT event_id_unique IF EXISTS;
DROP CONSTRAINT trade_case_unique IF EXISTS;
DROP INDEX activity_name IF EXISTS;

// Create constraints and indexes
CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (n:Event) REQUIRE n.ocel_id IS UNIQUE;
CREATE CONSTRAINT trade_case_unique IF NOT EXISTS FOR (n:TradeCase) REQUIRE n.case_id IS UNIQUE;
CREATE INDEX activity_name IF NOT EXISTS FOR (n:Activity) ON (n.name);

// Create Event nodes
CALL apoc.load.json('process_data.json') YIELD value as value1
UNWIND value1['ocel:events'] as event1
MERGE (e:Event {
    ocel_id: event1['ocel:id'],
    timestamp: datetime(event1['ocel:timestamp']),
    activity_name: event1['ocel:activity']
});

// Create Cases and link to Events with NULL check
CALL apoc.load.json('process_data.json') YIELD value as value2
UNWIND value2['ocel:events'] as event2
MATCH (e:Event {ocel_id: event2['ocel:id']})
WITH e, event2
WHERE event2.attributes.case_id IS NOT NULL
MERGE (c:TradeCase {case_id: event2.attributes.case_id})
MERGE (c)-[:CONTAINS_EVENT]->(e);

// Create Activity nodes and relationships
CALL apoc.load.json('process_data.json') YIELD value as value3
UNWIND value3['ocel:events'] as event3
MATCH (e:Event {ocel_id: event3['ocel:id']})
MERGE (a:Activity {name: event3['ocel:activity']})
MERGE (e)-[:OF_TYPE]->(a);

// Create Resource nodes and relationships with NULL check
CALL apoc.load.json('process_data.json') YIELD value as value4
UNWIND value4['ocel:events'] as event4
MATCH (e:Event {ocel_id: event4['ocel:id']})
WITH e, event4
WHERE event4.attributes.resource IS NOT NULL
MERGE (r:Resource {name: event4.attributes.resource})
MERGE (e)-[:PERFORMED_BY]->(r);

// Create Object nodes and relationships
CALL apoc.load.json('process_data.json') YIELD value as value5
UNWIND value5['ocel:events'] as event5
UNWIND event5['ocel:objects'] as obj
MATCH (e:Event {ocel_id: event5['ocel:id']})
WITH e, obj
WHERE obj.id IS NOT NULL AND obj.type IS NOT NULL
MERGE (o:TradeObject {
    id: obj.id,
    type: obj.type
})
MERGE (e)-[:INVOLVES]->(o);

// Add timestamp-based sequence relationships between events
MATCH (e1:Event)
WITH e1
ORDER BY e1.timestamp
WITH collect(e1) as events
UNWIND range(0, size(events)-2) as i
WITH events[i] as e1, events[i+1] as e2
WHERE e1.timestamp < e2.timestamp
MERGE (e1)-[r:NEXT_EVENT]->(e2)
SET r.time_difference = duration.between(e1.timestamp, e2.timestamp);