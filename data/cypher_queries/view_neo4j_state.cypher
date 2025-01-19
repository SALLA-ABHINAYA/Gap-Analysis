// Check all node labels and counts
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
RETURN label, value.count as count
ORDER BY label;

// Check all relationship types
CALL db.relationshipTypes() YIELD relationshipType
CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {}) YIELD value
RETURN relationshipType, value.count as count
ORDER BY relationshipType;

// Examine Principle nodes structure
MATCH (p:Principle)
RETURN p.code, p.name, p.description, p.category
LIMIT 5;

// Check Activity nodes and their properties
MATCH (a:Activity)
RETURN a.name, a.completion_rate, keys(a) as properties
LIMIT 5;

// Check Event nodes
MATCH (e:Event)
RETURN e.ocel_id, e.timestamp, e.activity, keys(e) as properties
LIMIT 5;

// Look at existing relationships between key entities
MATCH p = (n)-[r]->(m)
WHERE any(label IN labels(n) WHERE label IN ['Principle', 'Activity', 'Control', 'Event'])
RETURN p LIMIT 10;