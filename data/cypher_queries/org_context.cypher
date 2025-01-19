// Drop existing constraints and indexes for organization context
DROP CONSTRAINT org_entity_id_unique IF EXISTS;
DROP CONSTRAINT system_id_unique IF EXISTS;
DROP CONSTRAINT product_id_unique IF EXISTS;
DROP CONSTRAINT client_id_unique IF EXISTS;

DROP INDEX entity_type_idx IF EXISTS;
DROP INDEX system_name_idx IF EXISTS;
DROP INDEX product_type_idx IF EXISTS;

// Create constraints
CREATE CONSTRAINT org_entity_id_unique IF NOT EXISTS
FOR (e:OrganizationalEntity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT system_id_unique IF NOT EXISTS
FOR (s:TradingSystem) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT product_id_unique IF NOT EXISTS
FOR (p:Product) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT client_id_unique IF NOT EXISTS
FOR (c:Client) REQUIRE c.id IS UNIQUE;

// Create indexes
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:OrganizationalEntity) ON (e.type);
CREATE INDEX system_name_idx IF NOT EXISTS FOR (s:TradingSystem) ON (s.name);
CREATE INDEX product_type_idx IF NOT EXISTS FOR (p:Product) ON (p.type);

// Create organizational entities and functions
UNWIND [
    {id: 'FRONT_OFF', name: 'Front Office', type: 'DEPARTMENT'},
    {id: 'MID_OFF', name: 'Middle Office', type: 'DEPARTMENT'},
    {id: 'BACK_OFF', name: 'Back Office', type: 'DEPARTMENT'},
    {id: 'RISK_MGMT', name: 'Risk Management', type: 'FUNCTION'},
    {id: 'COMPLIANCE', name: 'Compliance', type: 'FUNCTION'},
    {id: 'OPERATIONS', name: 'Operations', type: 'FUNCTION'}
] AS entity
MERGE (e:OrganizationalEntity {id: entity.id})
SET e += entity;

// Create trading systems
UNWIND [
    {id: 'BBERG', name: 'Bloomberg', type: 'ORDER_MANAGEMENT',
     description: 'Order management and execution platform'},
    {id: 'WSS', name: 'Wall Street Systems', type: 'TRADE_PROCESSING',
     description: 'Trade capture and processing system'},
    {id: 'ION', name: 'ION Trading', type: 'TRADE_PROCESSING',
     description: 'Trade processing and risk management'},
    {id: 'MUREX', name: 'Murex', type: 'RISK_MANAGEMENT',
     description: 'Risk analytics and reporting'}
] AS system
MERGE (s:TradingSystem {id: system.id})
SET s += system;

// Create FX products and lifecycle events
UNWIND [
    {id: 'FX_SPOT', name: 'FX Spot', type: 'PRODUCT',
     events: ['Quote', 'Execute', 'Settle']},
    {id: 'FX_FWD', name: 'FX Forward', type: 'PRODUCT',
     events: ['Quote', 'Execute', 'Monitor', 'Settle']},
    {id: 'FX_SWAP', name: 'FX Swap', type: 'PRODUCT',
     events: ['Quote', 'Execute', 'Monitor', 'Unwind', 'Settle']}
] AS product
MERGE (p:Product {id: product.id})
SET p += product
WITH p, product
UNWIND product.events AS event
MERGE (e:LifecycleEvent {name: event})
MERGE (p)-[:HAS_LIFECYCLE_EVENT]->(e);

// Create clients and their system configurations
UNWIND [
    {id: 'CLIENT_1', name: 'Major Bank A',
     systems: ['BBERG', 'WSS', 'MUREX']},
    {id: 'CLIENT_2', name: 'Hedge Fund B',
     systems: ['BBERG', 'ION']},
    {id: 'CLIENT_3', name: 'Corporate C',
     systems: ['BBERG']}
] AS client
MERGE (c:Client {id: client.id})
SET c.name = client.name
WITH c, client
UNWIND client.systems AS sys
MATCH (s:TradingSystem {id: sys})
MERGE (c)-[:USES_SYSTEM]->(s);

// Link systems to organizational entities
MATCH (s:TradingSystem)
MATCH (e:OrganizationalEntity)
WHERE (s.type = 'ORDER_MANAGEMENT' AND e.id = 'FRONT_OFF')
   OR (s.type = 'TRADE_PROCESSING' AND e.id = 'OPERATIONS')
   OR (s.type = 'RISK_MANAGEMENT' AND e.id = 'RISK_MGMT')
MERGE (e)-[:OPERATES]->(s);

// Connect lifecycle events to required controls
MATCH (e:LifecycleEvent)
MATCH (c:Control)
WHERE (e.name = 'Execute' AND c.id IN ['CTRL_001', 'CTRL_002'])
   OR (e.name = 'Settle' AND c.id = 'CTRL_003')
MERGE (e)-[:REQUIRES_CONTROL]->(c);

// Link to existing process activities
MATCH (e:LifecycleEvent)
MATCH (a:Activity)
WHERE (e.name = 'Execute' AND a.name = 'Trade Execution')
   OR (e.name = 'Settle' AND a.name IN ['Trade Settlement', 'Settlement Instructions'])
MERGE (e)-[:IMPLEMENTED_BY]->(a);

// Add metadata timestamps
MATCH (n)
WHERE n:OrganizationalEntity OR n:TradingSystem OR n:Product OR n:Client OR n:LifecycleEvent
SET n.created_at = CASE WHEN n.created_at IS NULL THEN datetime() ELSE n.created_at END,
    n.updated_at = datetime(),
    n.version = CASE WHEN n.version IS NULL THEN '1.0' ELSE n.version END;