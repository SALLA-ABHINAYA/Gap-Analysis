// First drop existing constraints and indexes if they exist
DROP INDEX principle_code_idx IF EXISTS;
DROP INDEX requirement_type_idx IF EXISTS;
DROP INDEX control_name_idx IF EXISTS;
DROP INDEX risk_category_code_idx IF EXISTS;

DROP CONSTRAINT principle_code_unique IF EXISTS;
DROP CONSTRAINT standard_code_version_unique IF EXISTS;
DROP CONSTRAINT requirement_id_unique IF EXISTS;
DROP CONSTRAINT control_id_unique IF EXISTS;

// Create constraints for data integrity
CREATE CONSTRAINT principle_code_unique IF NOT EXISTS
FOR (p:Principle) REQUIRE p.code IS UNIQUE;

CREATE CONSTRAINT standard_code_version_unique IF NOT EXISTS
FOR (s:Standard) REQUIRE (s.code, s.version) IS UNIQUE;

CREATE CONSTRAINT requirement_id_unique IF NOT EXISTS
FOR (r:Requirement) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT control_id_unique IF NOT EXISTS
FOR (c:Control) REQUIRE c.id IS UNIQUE;

// Create indexes for performance
CREATE INDEX principle_code_idx IF NOT EXISTS FOR (p:Principle) ON (p.code);
CREATE INDEX requirement_type_idx IF NOT EXISTS FOR (r:Requirement) ON (r.type);
CREATE INDEX control_name_idx IF NOT EXISTS FOR (c:Control) ON (c.name);
CREATE INDEX risk_category_code_idx IF NOT EXISTS FOR (rc:RiskCategory) ON (rc.code);

// Create FX Global Code standard
MATCH (gcode:Standard {code: 'FX_GLOBAL', version: '2021'})
SET gcode.name = 'FX Global Code',
    gcode.last_updated = datetime()
WITH gcode

// Create main principle categories
UNWIND [
    {code: 'ETHICS', name: 'Ethics', description: 'Market Participants are expected to behave in an ethical and professional manner'},
    {code: 'GOV', name: 'Governance', description: 'Market Participants are expected to have a sound and effective governance framework'},
    {code: 'EXEC', name: 'Execution', description: 'Market Participants are expected to exercise care when negotiating and executing transactions'},
    {code: 'INFO', name: 'Information Sharing', description: 'Market Participants are expected to be clear and accurate in their communications'},
    {code: 'RISK', name: 'Risk Management and Compliance', description: 'Market Participants are expected to promote and maintain a robust control and compliance environment'},
    {code: 'SETTLE', name: 'Confirmation and Settlement', description: 'Market Participants are expected to put in place robust, efficient, transparent post-trade processes'}
] AS categories
MERGE (c:PrincipleCategory {code: categories.code})
SET c += categories
MERGE (gcode)-[:CONTAINS_CATEGORY]->(c);

// Create detailed principles
WITH [
    {category: 'EXEC', principles: [
        {code: 'EXEC_1', name: 'Market Participant Roles', description: 'Market Participants should be clear about the capacities in which they act'},
        {code: 'EXEC_2', name: 'Order Handling', description: 'Market Participants should handle orders fairly and with transparency'},
        {code: 'EXEC_3', name: 'Pre-Hedging', description: 'Market Participants should only Pre-Hedge Client orders when acting as Principal'},
        {code: 'EXEC_4', name: 'Market Disruption', description: 'Market Participants should not request transactions or create orders that disrupt market functioning'}
    ]},
    {category: 'RISK', principles: [
        {code: 'RISK_1', name: 'Risk Framework', description: 'Market Participants should have frameworks for risk management and compliance'},
        {code: 'RISK_2', name: 'Risk Controls', description: 'Market Participants should have processes to monitor and control risks'}
    ]}
] AS categoryPrinciples
UNWIND categoryPrinciples AS catPrinciple
MATCH (category:PrincipleCategory {code: catPrinciple.category})
UNWIND catPrinciple.principles AS principle
MERGE (p:Principle {code: principle.code})
SET p += principle
MERGE (category)-[:CONTAINS_PRINCIPLE]->(p);

// Create requirements
WITH [
    {principle: 'EXEC_2', requirements: [
        {id: 'REQ_001', description: 'Must provide clear standards for fair order handling', type: 'MANDATORY'},
        {id: 'REQ_002', description: 'Must disclose order handling policies to clients', type: 'MANDATORY'},
        {id: 'REQ_003', description: 'Should maintain audit trails of order execution', type: 'RECOMMENDED'}
    ]},
    {principle: 'RISK_1', requirements: [
        {id: 'REQ_004', description: 'Must implement risk management framework', type: 'MANDATORY'},
        {id: 'REQ_005', description: 'Should conduct regular risk assessments', type: 'RECOMMENDED'}
    ]}
] AS principleRequirements
UNWIND principleRequirements AS prinReq
MATCH (p:Principle {code: prinReq.principle})
UNWIND prinReq.requirements AS req
MERGE (r:Requirement {id: req.id})
SET r += req
MERGE (p)-[:HAS_REQUIREMENT]->(r);

// Create controls
WITH [
    {requirement: 'REQ_001', controls: [
        {id: 'CTRL_001', name: 'Order Execution Policy', description: 'Documented policy for fair order execution'},
        {id: 'CTRL_002', name: 'Time Stamping', description: 'Automated time stamping of all order events'},
        {id: 'CTRL_003', name: 'Execution Quality Monitoring', description: 'Regular monitoring of execution quality metrics'}
    ]}
] AS requirementControls
UNWIND requirementControls AS reqCtrl
MATCH (r:Requirement {id: reqCtrl.requirement})
UNWIND reqCtrl.controls AS ctrl
MERGE (c:Control {id: ctrl.id})
SET c += ctrl
MERGE (r)-[:IMPLEMENTED_BY]->(c);

// Create risk categories with hierarchical structure
MERGE (rc1:RiskCategory {code: 'MARKET_RISK', name: 'Market Risk'})
MERGE (rc2:RiskCategory {code: 'OPERATIONAL_RISK', name: 'Operational Risk'})
MERGE (rc3:RiskCategory {code: 'SETTLEMENT_RISK', name: 'Settlement Risk'})
WITH rc1, rc2, rc3

// Create sub-risk categories
MERGE (src1:RiskSubCategory {code: 'FX_SPOT_RISK', name: 'FX Spot Risk'})
MERGE (src2:RiskSubCategory {code: 'FX_SWAP_RISK', name: 'FX Swap Risk'})
MERGE (rc1)-[:CONTAINS_SUB_RISK]->(src1)
MERGE (rc1)-[:CONTAINS_SUB_RISK]->(src2)
WITH rc1, rc2, rc3

// Link controls to actual activities
MATCH (c:Control {id: 'CTRL_002'})
WITH c
MATCH (a:Activity)
WHERE a.name IN ['Trade Execution', 'Order Processing', 'Trade Confirmation']
MERGE (c)-[:MONITORS]->(a);

// Create relationships between risk categories and principles
MATCH (rc:RiskCategory {code: 'MARKET_RISK'})
MATCH (p:Principle {code: 'RISK_1'})
MERGE (p)-[:ADDRESSES_RISK]->(rc);

// Add timestamps and metadata
MATCH (n)
WHERE n:Standard OR n:Principle OR n:Requirement OR n:Control
SET n.created_at = CASE WHEN n.created_at IS NULL THEN datetime() ELSE n.created_at END,
    n.updated_at = datetime(),
    n.version = CASE WHEN n.version IS NULL THEN '1.0' ELSE n.version END;