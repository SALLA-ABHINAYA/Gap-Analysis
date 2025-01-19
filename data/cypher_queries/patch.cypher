// Add processing_type to Event nodes
MATCH (e:Event)
SET e.processing_type = CASE
    WHEN e.duration_seconds <= 60 THEN 'STP'  // Events completing within 1 minute
    ELSE 'MANUAL'
END;

// Add required_activities to Guidelines
MATCH (g:Guideline)
SET g.required_activities = CASE g.type
    WHEN 'REGULATORY' THEN ['Trade Validation', 'Risk Assessment', 'Compliance Check']
    WHEN 'OPERATIONAL' THEN ['Trade Execution', 'Trade Settlement']
    ELSE ['Trade Initiation']
END;

// Add implementation_status if missing
MATCH (g:Guideline)
WHERE g.implementation_status IS NULL
SET g.implementation_status = 'PENDING';





// Map FX Global Code Principles as Guidelines
MATCH (p:Principle)
MERGE (g:Guideline {
    id: p.code,
    name: p.name,
    type: CASE p.category
        WHEN 'RISK' THEN 'REGULATORY'
        ELSE 'OPERATIONAL'
    END,
    severity: 'High',
    description: p.description
})
WITH g, p
MATCH (p)-[:HAS_REQUIREMENT]->(r:Requirement)
WITH g, collect(r.description) as reqs
SET g.requirements = reqs,
    g.implementation_status = 'PENDING',
    g.required_activities = CASE g.type
        WHEN 'REGULATORY' THEN ['Trade Validation', 'Risk Assessment', 'Compliance Check']
        ELSE ['Trade Execution', 'Trade Settlement']
    END;

// Add processing type to Events
MATCH (e:Event)
SET e.processing_type = CASE
    WHEN duration.between(e.timestamp, e.timestamp + duration('PT1H')) <= duration('PT1M')
    THEN 'STP'
    ELSE 'MANUAL'
END;

// Map Standards to Controls
MATCH (s:Standard {code: 'FX_GLOBAL'})
MATCH (p:Principle)-[:HAS_REQUIREMENT]->(r:Requirement)
MERGE (c:Control {
    id: 'CTRL_' + r.id,
    name: r.description,
    type: 'FX_GLOBAL_CODE',
    effectiveness_score: 0.75
});

// Link Controls to Activities
MATCH (c:Control)
MATCH (a:Activity)
WHERE a.name IN ['Trade Validation', 'Risk Assessment', 'Trade Execution']
MERGE (c)-[:MONITORS]->(a);

// Set completion rates for Activities
MATCH (a:Activity)
SET a.completion_rate = CASE
    WHEN exists((a)<-[:MONITORS]-(:Control)) THEN 0.85
    ELSE 0.6
END;



// Link Controls to Requirements
MATCH (r:Requirement)
MATCH (c:Control)
WHERE c.name = r.description
MERGE (c)-[:IMPLEMENTS]->(r);

// Add relationships between Controls and Activities
MATCH (c:Control)
MATCH (a:Activity)
WHERE a.name IN ['Trade Validation', 'Risk Assessment', 'Trade Execution', 'Trade Settlement']
MERGE (c)-[:MONITORS]->(a);

// Create execution instances for tracking
MATCH (c:Control)
CREATE (ce:ControlExecution {
    id: 'CE_' + c.id + '_' + toString(datetime()),
    timestamp: datetime(),
    status: 'COMPLETED',
    result: 'PASS'
})
MERGE (ce)-[:EXECUTES]->(c);