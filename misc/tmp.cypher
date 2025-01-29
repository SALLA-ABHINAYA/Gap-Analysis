// Check all node labels and counts
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value
RETURN label, value.count as count
ORDER BY label;

╒══════════════════════╤═════╕
│label                 │count│
╞══════════════════════╪═════╡
│"Activity"            │35   │
├──────────────────────┼─────┤
│"Client"              │3    │
├──────────────────────┼─────┤
│"Control"             │8    │
├──────────────────────┼─────┤
│"ControlExecution"    │8    │
├──────────────────────┼─────┤
│"Event"               │82736│
├──────────────────────┼─────┤
│"Guideline"           │6    │
├──────────────────────┼─────┤
│"LifecycleEvent"      │5    │
├──────────────────────┼─────┤
│"OrganizationalEntity"│6    │
├──────────────────────┼─────┤
│"Principle"           │6    │
├──────────────────────┼─────┤
│"PrincipleCategory"   │6    │
├──────────────────────┼─────┤
│"Product"             │3    │
├──────────────────────┼─────┤
│"Requirement"         │5    │
├──────────────────────┼─────┤
│"RiskCategory"        │3    │
├──────────────────────┼─────┤
│"RiskSubCategory"     │2    │
├──────────────────────┼─────┤
│"Standard"            │1    │
├──────────────────────┼─────┤
│"TradeObject"         │12000│
├──────────────────────┼─────┤
│"TradingSystem"       │4    │


// Check all relationship types
CALL db.relationshipTypes() YIELD relationshipType
CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {}) YIELD value
RETURN relationshipType, value.count as count
ORDER BY relationshipType;

╒═════════════════════╤═════╕
│relationshipType     │count│
╞═════════════════════╪═════╡
│"ADDRESSES_RISK"     │1    │
├─────────────────────┼─────┤
│"CONTAINS_CATEGORY"  │6    │
├─────────────────────┼─────┤
│"CONTAINS_PRINCIPLE" │6    │
├─────────────────────┼─────┤
│"CONTAINS_SUB_RISK"  │2    │
├─────────────────────┼─────┤
│"EXECUTES"           │8    │
├─────────────────────┼─────┤
│"HAS_LIFECYCLE_EVENT"│12   │
├─────────────────────┼─────┤
│"HAS_REQUIREMENT"    │5    │
├─────────────────────┼─────┤
│"IMPLEMENTED_BY"     │5    │
├─────────────────────┼─────┤
│"IMPLEMENTS"         │5    │
├─────────────────────┼─────┤
│"INVOLVES"           │82736│
├─────────────────────┼─────┤
│"MONITORS"           │25   │
├─────────────────────┼─────┤
│"NEXT_EVENT"         │79873│
├─────────────────────┼─────┤
│"OF_TYPE"            │82736│
├─────────────────────┼─────┤
│"OPERATES"           │4    │
├─────────────────────┼─────┤
│"REQUIRES_CONTROL"   │3    │
├─────────────────────┼─────┤
│"USES_SYSTEM"        │6    │
└─────────────────────┴─────┘


// Examine Principle nodes structure
MATCH (p:Principle)
RETURN p.code, p.name, p.description, p.category
LIMIT 5;

╒════════╤══════════════════════════╤══════════════════════════════════════════════════════════════════════╤══════════╕
│p.code  │p.name                    │p.description                                                         │p.category│
╞════════╪══════════════════════════╪══════════════════════════════════════════════════════════════════════╪══════════╡
│"EXEC_1"│"Market Participant Roles"│"Market Participants should be clear about the capacities in which the│null      │
│        │                          │y act"                                                                │          │
├────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────┤
│"EXEC_2"│"Order Handling"          │"Market Participants should handle orders fairly and with transparency│null      │
│        │                          │"                                                                     │          │
├────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────┤
│"EXEC_3"│"Pre-Hedging"             │"Market Participants should only Pre-Hedge Client orders when acting a│null      │
│        │                          │s Principal"                                                          │          │
├────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────┤
│"EXEC_4"│"Market Disruption"       │"Market Participants should not request transactions or create orders │null      │
│        │                          │that disrupt market functioning"                                      │          │
├────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────┼──────────┤
│"RISK_1"│"Risk Framework"          │"Market Participants should have frameworks for risk management and co│null      │
│        │                          │mpliance"                                                             │          │
└────────┴──────────────────────────┴──────────────────────────────────────────────────────────────────────┴──────────┘



// Check Activity nodes and their properties
MATCH (a:Activity)
RETURN a.name, a.completion_rate, keys(a) as properties
LIMIT 5;

╒═════════════════════════════╤═════════════════╤═══════════════════════════╕
│a.name                       │a.completion_rate│properties                 │
╞═════════════════════════════╪═════════════════╪═══════════════════════════╡
│"Trade Initiated"            │0.6              │["completion_rate", "name"]│
├─────────────────────────────┼─────────────────┼───────────────────────────┤
│"Market Data Validation"     │0.6              │["completion_rate", "name"]│
├─────────────────────────────┼─────────────────┼───────────────────────────┤
│"Volatility Surface Analysis"│0.6              │["completion_rate", "name"]│
├─────────────────────────────┼─────────────────┼───────────────────────────┤
│"Quote Provided"             │0.6              │["name", "completion_rate"]│
├─────────────────────────────┼─────────────────┼───────────────────────────┤
│"Exercise Decision"          │0.6              │["completion_rate", "name"]│
└─────────────────────────────┴─────────────────┴───────────────────────────┘



// Check Event nodes
MATCH (e:Event)
RETURN e.ocel_id, e.timestamp, e.activity, keys(e) as properties
LIMIT 5;

╒════════════════════════════════════════╤════════════════════════════════╤══════════╤════════════════════════════════════════════════════════════╕
│e.ocel_id                               │e.timestamp                     │e.activity│properties                                                  │
╞════════════════════════════════════════╪════════════════════════════════╪══════════╪════════════════════════════════════════════════════════════╡
│"Case_1_Initial Margin Calculation"     │"2024-11-10T05:18:10.387330000Z"│null      │["processing_type", "timestamp", "activity_name", "ocel_id"]│
├────────────────────────────────────────┼────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────┤
│"Case_1_Trade Validation"               │"2024-11-10T05:45:10.387330000Z"│null      │["processing_type", "timestamp", "activity_name", "ocel_id"]│
├────────────────────────────────────────┼────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────┤
│"Case_1_Regulatory Reporting Generation"│"2024-11-10T06:08:10.387330000Z"│null      │["processing_type", "activity_name", "ocel_id", "timestamp"]│
├────────────────────────────────────────┼────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────┤
│"Case_1_Settlement Instructions"        │"2024-11-10T06:42:10.387330000Z"│null      │["processing_type", "timestamp", "activity_name", "ocel_id"]│
├────────────────────────────────────────┼────────────────────────────────┼──────────┼────────────────────────────────────────────────────────────┤
│"Case_1_Trade Matching"                 │"2024-11-10T07:46:10.387330000Z"│null      │["timestamp", "activity_name", "processing_type", "ocel_id"]│
└────────────────────────────────────────┴────────────────────────────────┴──────────┴────────────────────────────────────────────────────────────┘


// Look at existing relationships between key entities
MATCH p = (n)-[r]->(m)
WHERE any(label IN labels(n) WHERE label IN ['Principle', 'Activity', 'Control', 'Event'])
RETURN p LIMIT 10;

╒══════════════════════════════════════════════════════════════════════╕
│p                                                                     │
╞══════════════════════════════════════════════════════════════════════╡
│(:Event {activity_name: "Initial Margin Calculation",ocel_id: "Case_1_│
│Initial Margin Calculation",processing_type: "MANUAL",timestamp: "2024│
│-11-10T05:18:10.387330000Z"})-[:NEXT_EVENT {time_difference: "P0M0DT0.│
│015197000S"}]->(:Event {activity_name: "Initial Margin Calculation",oc│
│el_id: "Case_140_Initial Margin Calculation",processing_type: "MANUAL"│
│,timestamp: "2024-11-10T05:18:10.402527000Z"})                        │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Initial Margin Calculation",ocel_id: "Case_1_│
│Initial Margin Calculation",processing_type: "MANUAL",timestamp: "2024│
│-11-10T05:18:10.387330000Z"})-[:INVOLVES]->(:TradeObject {id: "Trade_C│
│ase_1",type: "Trade"})                                                │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Initial Margin Calculation",ocel_id: "Case_1_│
│Initial Margin Calculation",processing_type: "MANUAL",timestamp: "2024│
│-11-10T05:18:10.387330000Z"})-[:OF_TYPE]->(:Activity {completion_rate:│
│ 0.6,name: "Initial Margin Calculation"})                             │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Trade Validation",ocel_id: "Case_1_Trade Vali│
│dation",processing_type: "MANUAL",timestamp: "2024-11-10T05:45:10.3873│
│30000Z"})-[:NEXT_EVENT {time_difference: "P0M0DT0.004602000S"}]->(:Eve│
│nt {activity_name: "Trade Confirmation",ocel_id: "Case_48_Trade Confir│
│mation",processing_type: "MANUAL",timestamp: "2024-11-10T05:45:10.3919│
│32000Z"})                                                             │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Trade Validation",ocel_id: "Case_1_Trade Vali│
│dation",processing_type: "MANUAL",timestamp: "2024-11-10T05:45:10.3873│
│30000Z"})-[:INVOLVES]->(:TradeObject {id: "Trade_Case_1",type: "Trade"│
│})                                                                    │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Trade Validation",ocel_id: "Case_1_Trade Vali│
│dation",processing_type: "MANUAL",timestamp: "2024-11-10T05:45:10.3873│
│30000Z"})-[:OF_TYPE]->(:Activity {completion_rate: 0.85,name: "Trade V│
│alidation"})                                                          │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Regulatory Reporting Generation",ocel_id: "Ca│
│se_1_Regulatory Reporting Generation",processing_type: "MANUAL",timest│
│amp: "2024-11-10T06:08:10.387330000Z"})-[:NEXT_EVENT {time_difference:│
│ "P0M0DT0.033011000S"}]->(:Event {activity_name: "Trade Matching",ocel│
│_id: "Case_317_Trade Matching",processing_type: "MANUAL",timestamp: "2│
│024-11-10T06:08:10.420341000Z"})                                      │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Regulatory Reporting Generation",ocel_id: "Ca│
│se_1_Regulatory Reporting Generation",processing_type: "MANUAL",timest│
│amp: "2024-11-10T06:08:10.387330000Z"})-[:INVOLVES]->(:TradeObject {id│
│: "Trade_Case_1",type: "Trade"})                                      │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Regulatory Reporting Generation",ocel_id: "Ca│
│se_1_Regulatory Reporting Generation",processing_type: "MANUAL",timest│
│amp: "2024-11-10T06:08:10.387330000Z"})-[:OF_TYPE]->(:Activity {comple│
│tion_rate: 0.6,name: "Regulatory Reporting Generation"})              │
├──────────────────────────────────────────────────────────────────────┤
│(:Event {activity_name: "Settlement Instructions",ocel_id: "Case_1_Set│
│tlement Instructions",processing_type: "MANUAL",timestamp: "2024-11-10│
│T06:42:10.387330000Z"})-[:NEXT_EVENT {time_difference: "P0M0DT0.047611│
│000S"}]->(:Event {activity_name: "Skew Calibration",ocel_id: "Case_462│
│_Skew Calibration",processing_type: "MANUAL",timestamp: "2024-11-10T06│
│:42:10.434941000Z"})                                                  │
└──────────────────────────────────────────────────────────────────────┘