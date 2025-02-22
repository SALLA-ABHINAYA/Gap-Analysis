"""
OCPM Gap Analysis Tool
- Connects to Neo4j Aura DB
- Processes OCEL data
- Compares against guidelines using Azure OpenAI
- Generates CSV report
"""

import os
import json
from datetime import datetime

import pandas as pd
from neo4j import GraphDatabase
import logging
from utils import get_azure_openai_client
from typing import Dict, List, Any
import traceback


def datetime_handler(obj):
    """Handle datetime serialization for JSON"""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for maximum detail
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('ocpm_gap_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OCPMAnalyzer:
    """Analyzes Object-Centric Process Mining data against guidelines"""

    def __init__(self, guideline_path: str):
        """Initialize the OCPM Analyzer"""
        logger.info(f"=== Initializing OCPMAnalyzer with guidelines: {guideline_path} ===")
        try:
            # Load guidelines first
            logger.debug("Loading guidelines...")
            self.guidelines = self._load_guidelines(guideline_path)
            logger.debug(f"Guidelines loaded, length: {len(self.guidelines)} characters")

            # Initialize Neo4j connection
            logger.debug("Retrieving Neo4j credentials...")
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                logger.warning("Environment variables not found, using default test instance")
                neo4j_uri = "neo4j+s://1bcd5ab7.databases.neo4j.io"
                neo4j_user = "neo4j"
                neo4j_password = "mfk00LT6txYy1Szvv_lxpfg_UvlG5A5D921WHeWCaX0"

            logger.debug(f"Using Neo4j URI: {neo4j_uri}")
            logger.debug(f"Using Neo4j user: {neo4j_user}")

            # Initialize Neo4j driver
            logger.debug("Initializing Neo4j driver...")
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )

            # Test connection
            logger.debug("Testing Neo4j connection...")
            self._test_connection()

            # Initialize schema and load data
            logger.debug("Initializing Neo4j schema and loading data...")
            self._initialize_neo4j_schema()

            # Initialize Azure OpenAI client
            logger.debug("Initializing Azure OpenAI client...")
            self.llm_client = get_azure_openai_client()

            logger.info("OCPMAnalyzer initialization completed successfully")

        except Exception as e:
            logger.error(f"Error initializing OCPMAnalyzer: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_neo4j_schema(self) -> None:
        """Initialize Neo4j schema and load OCEL data"""
        logger.debug("=== Initializing Neo4j Schema ===")

        try:
            # Read OCEL data from file
            ocel_path = "ocpm_output/process_data.json"
            logger.debug(f"Loading OCEL data from {ocel_path}")

            with open(ocel_path, 'r') as f:
                ocel_data = json.load(f)

            logger.debug(f"Loaded OCEL data with {len(ocel_data.get('ocel:events', []))} events")

            # Create Neo4j schema and load data
            with self.neo4j_driver.session() as session:
                # First properly clean up existing data and constraints
                logger.debug("Cleaning up existing data and constraints...")
                cleanup_queries = [
                    "MATCH (n) DETACH DELETE n",  # Delete all nodes and relationships
                    "DROP CONSTRAINT case_id IF EXISTS",  # Drop existing constraints
                    "DROP CONSTRAINT event_id IF EXISTS"
                ]
                for query in cleanup_queries:
                    session.run(query)

                # Create constraints
                logger.debug("Creating constraints...")
                constraint_queries = [
                    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE"
                ]
                for query in constraint_queries:
                    session.run(query)

                # Create Cases and Events using batch processing
                logger.debug("Creating Cases and Events...")
                batch_size = 100  # Process in smaller batches
                events = ocel_data.get('ocel:events', [])

                for i in range(0, len(events), batch_size):
                    batch = events[i:i + batch_size]
                    # Create unique cases first
                    case_query = """
                    UNWIND $cases as case_data
                    MERGE (c:Case {id: case_data.case_id})
                    ON CREATE SET c.created_at = datetime()
                    """
                    case_data = [
                        {'case_id': event.get('ocel:attributes', {}).get('case_id')}
                        for event in batch
                        if event.get('ocel:attributes', {}).get('case_id')
                    ]
                    if case_data:
                        session.run(case_query, cases=case_data)

                    # Then create events with relationships
                    event_query = """
                    UNWIND $events as event_data
                    MATCH (c:Case {id: event_data.case_id})
                    CREATE (e:Event {
                        id: event_data.event_id,
                        activity: event_data.activity,
                        timestamp: datetime(event_data.timestamp),
                        object_type: event_data.object_type
                    })
                    CREATE (c)-[:HAS_EVENT]->(e)
                    """
                    event_data = [
                        {
                            'case_id': event.get('ocel:attributes', {}).get('case_id'),
                            'event_id': event['ocel:id'],
                            'activity': event['ocel:activity'],
                            'timestamp': event['ocel:timestamp'],
                            'object_type': event.get('ocel:attributes', {}).get('object_type', 'Unknown')
                        }
                        for event in batch
                        if event.get('ocel:attributes', {}).get('case_id')
                    ]
                    if event_data:
                        session.run(event_query, events=event_data)

                    logger.debug(f"Processed batch of {len(batch)} events")

                # Verify data loading
                verification_query = """
                MATCH (c:Case)
                WITH count(c) as case_count
                MATCH (e:Event)
                WITH case_count, count(e) as event_count
                RETURN case_count, event_count
                """
                result = session.run(verification_query).single()
                logger.info(f"Loaded {result['case_count']} cases and {result['event_count']} events")

            logger.info("Neo4j schema initialized and data loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing Neo4j schema: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _test_connection(self) -> None:
        """Test Neo4j connection"""
        logger.debug("=== Testing Neo4j Connection ===")
        try:
            with self.neo4j_driver.session() as session:
                logger.debug("Executing test query...")
                result = session.run("RETURN 1 as test")
                value = result.single()
                logger.debug(f"Test query result: {value}")
            logger.info("Neo4j connection test successful")
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _determine_control_framework(self, activities: List[str]) -> str:
        """Determine appropriate control framework based on activities"""
        control_frameworks = {
            'Data Controls': ['Validation', 'Check', 'Verify', 'Audit'],
            'Process Controls': ['Execute', 'Process', 'Perform', 'Handle'],
            'System Controls': ['Calculate', 'Generate', 'Compute', 'Route'],
            'User Controls': ['Approve', 'Review', 'Authorize', 'Decide'],
            'Integration Controls': ['Connect', 'Link', 'Interface', 'Transfer'],
            'Compliance Controls': ['Monitor', 'Report', 'Track', 'Log']
        }

        activity_str = ' '.join(activities).upper()
        applied_frameworks = []

        for framework, indicators in control_frameworks.items():
            if any(ind.upper() in activity_str for ind in indicators):
                applied_frameworks.append(framework)

        if not applied_frameworks:
            return "Basic Process Controls"

        return ' & '.join(sorted(set(applied_frameworks)))

    def _check_regulatory_sla(self, timestamps: List[str], activities: List[str]) -> List[Dict[str, Any]]:
        """Analyze regulatory SLA breaches"""
        sla_breaches = []

        # Convert timestamps to datetime objects
        ts = [pd.to_datetime(t) for t in timestamps]
        activity_times = dict(zip(activities, ts))

        # Define regulatory SLAs
        regulatory_slas = {
            'Trade Execution': {
                'regulation': 'MiFID II RTS 2',
                'requirement': 'Real-time trade transparency',
                'max_delay': pd.Timedelta(minutes=1)
            },
            'Transaction Reporting': {
                'regulation': 'MiFID II Article 26',
                'requirement': 'Report by T+1',
                'max_delay': pd.Timedelta(days=1)
            },
            'Trade Confirmation': {
                'regulation': 'CFTC 1.35',
                'requirement': 'Trade confirmation timing',
                'max_delay': pd.Timedelta(minutes=15)
            },
            'Position Reconciliation': {
                'regulation': 'EMIR Article 11',
                'requirement': 'Daily reconciliation',
                'max_delay': pd.Timedelta(days=1)
            },
            'Risk Assessment': {
                'regulation': 'Dodd-Frank 165',
                'requirement': 'Risk reporting deadline',
                'max_delay': pd.Timedelta(hours=24)
            }
        }

        # Check for SLA breaches
        for activity in activities:
            base_activity = next((key for key in regulatory_slas.keys() if key in activity), None)
            if base_activity:
                sla = regulatory_slas[base_activity]

                # Find next reportable activity
                reportable_activities = {
                    'Trade Execution': ['Trade Reporting', 'Regulatory Reporting'],
                    'Transaction Reporting': ['Report Submission', 'Regulatory Filing'],
                    'Trade Confirmation': ['Confirmation Sent', 'Confirmation Matching'],
                    'Position Reconciliation': ['Reconciliation Complete', 'Position Reporting'],
                    'Risk Assessment': ['Risk Report', 'Risk Disclosure']
                }

                for report_activity in reportable_activities.get(base_activity, []):
                    if report_activity in activities:
                        idx_base = activities.index(activity)
                        idx_report = activities.index(report_activity)

                        if idx_report > idx_base:
                            time_taken = ts[idx_report] - ts[idx_base]

                            if time_taken > sla['max_delay']:
                                sla_breaches.append({
                                    'regulation': sla['regulation'],
                                    'sla_requirement': sla['requirement'],
                                    'activity': f'{base_activity} to {report_activity}',
                                    'required_time': str(sla['max_delay']),
                                    'actual_time': str(time_taken),
                                    'breach_category': 'Regulatory Reporting',
                                    'impact': f'Late {sla["requirement"].lower()}'
                                })

        return sla_breaches

    def _determine_execution_pattern(self, activities: List[str]) -> str:
        """Analyze process execution pattern"""
        start_indicators = ['Initialize', 'Start', 'Create', 'Begin']
        middle_indicators = ['Process', 'Execute', 'Handle', 'Perform']
        end_indicators = ['Complete', 'Finish', 'Close', 'End']

        activity_str = ' '.join(activities).upper()
        patterns = []

        if any(ind.upper() in activity_str for ind in start_indicators):
            patterns.append('Initialization')
        if any(ind.upper() in activity_str for ind in middle_indicators):
            patterns.append('Processing')
        if any(ind.upper() in activity_str for ind in end_indicators):
            patterns.append('Completion')

        if len(patterns) == 3:
            return "Complete Process Flow"
        elif len(patterns) == 0:
            return "Unknown Pattern"
        else:
            return ' & '.join(patterns)

    def _determine_process_type(self, activities: List[str], object_types: List[str]) -> str:
        """Determine process type based on activities and objects"""
        # Count unique object types
        unique_objects = set(object_types)

        # Analyze activity complexity
        activity_complexity = len(set(activities))

        if len(unique_objects) > 3:
            if activity_complexity > 10:
                return "Complex Multi-Object Process"
            return "Standard Multi-Object Process"
        elif len(unique_objects) > 1:
            if activity_complexity > 5:
                return "Complex Object Interaction"
            return "Simple Object Interaction"
        else:
            if activity_complexity > 3:
                return "Complex Single Object"
            return "Simple Single Object"

    def _extract_object_types(self, activities: List[str]) -> List[str]:
        """Extract potential object types from activity names"""
        object_types = set()
        for activity in activities:
            # Split activity name and look for potential object names
            words = activity.split()
            # Simple heuristic: words starting with capital letters are likely objects
            objects = [w for w in words if w[0].isupper() and len(w) > 2]
            object_types.update(objects)
        return list(object_types)

    def _get_process_context(self, activities: List[str]) -> Dict[str, str]:
        """Get comprehensive process context"""
        return {
            'control_framework': self._determine_control_framework(activities),
            'execution_pattern': self._determine_execution_pattern(activities),
            'process_type': self._determine_process_type(activities,
                                                         self._extract_object_types(activities))
        }

    def _load_guidelines(self, path: str) -> str:
        """Load process guidelines from file"""
        logger.debug(f"=== Loading guidelines from {path} ===")
        try:
            if not os.path.exists(path):
                logger.error(f"Guidelines file not found: {path}")
                raise FileNotFoundError(f"Guidelines file not found: {path}")

            # Changed to use utf-8 encoding with error handling
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                guidelines = f.read()

            if not guidelines.strip():
                logger.error("Guidelines file is empty")
                raise ValueError("Guidelines file is empty")

            logger.debug(f"Guidelines loaded successfully, size: {len(guidelines)} characters")
            return guidelines

        except Exception as e:
            logger.error(f"Error loading guidelines: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_process_flows(self) -> List[Dict[str, Any]]:
        """Retrieve process flows from Neo4j"""
        logger.debug("=== Retrieving process flows from Neo4j ===")

        query = """
        MATCH (c:Case)-[:HAS_EVENT]->(e:Event)
        WITH c, e
        ORDER BY e.timestamp
        WITH c, 
             collect(e.activity) as activities,
             collect(e.timestamp) as timestamps,
             collect(e.object_type) as object_types
        RETURN c.id as case_id, 
               activities,
               timestamps,
               object_types
        """
        logger.debug(f"Executing query: {query}")

        try:
            with self.neo4j_driver.session() as session:
                logger.debug("Starting query execution...")
                result = session.run(query)
                flows = [dict(record) for record in result]

            logger.info(f"Retrieved {len(flows)} process flows")
            if flows:
                logger.debug(f"Sample first flow: {json.dumps(flows[0], default=str)}")
            else:
                logger.warning("No process flows retrieved from Neo4j")

            return flows

        except Exception as e:
            logger.error(f"Error retrieving process flows: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _analyze_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gaps focusing on regulatory compliance and control requirements"""
        logger.debug(f"=== Analyzing case {case['case_id']} ===")

        try:
            activities = case['activities']
            timestamps = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                          for ts in case['timestamps']]
            object_types = case['object_types']

            prompt = f"""
            Analyze this FX trading process case for control and compliance gaps against guidelines.

            Process Guidelines:
            {self.guidelines}

            Case Details:
            - Case ID: {case['case_id']}
            - Activities: {activities}
            - Timestamps: {timestamps}
            - Object Types: {object_types}

            Analyze for the following specific gaps:

            1. Control Gaps (missing_steps):
            - Missing mandatory controls (e.g., limit checks, KYC validation)
            - Missing compliance validations
            - Missing regulatory reporting controls
            - Missing risk controls

            2. Unsupported Control Gaps (extra_steps):
            - Unauthorized control steps
            - Non-standard compliance checks
            - Redundant control validations
            - Unofficial control measures

            3. Control Flow Gaps (sequence_issues):
            - Pre-trade controls executed post-trade
            - Risk validations out of sequence
            - Compliance checks in wrong order
            - Control dependencies violated

            4. SLA Breaches (timing_violations):
            - Regulatory reporting deadlines missed
            - Compliance validation timeframes exceeded
            - Control execution delays
            - Risk assessment timing breaches

            5. Object Control Violations (object_violations):
            - Trade object control violations
            - Position limit breaches
            - Client object control gaps
            - Market data control issues

            6. Compliance Gaps (compliance_gaps):
            - Direct regulatory violations
            - Control framework breaches
            - Compliance rule violations
            - Regulatory reporting failures

            Return a JSON object with:
            {{
                "missing_steps": [
                    {{
                        "control": "missing control name",
                        "category": "Pre-trade/Post-trade/Risk/Compliance",
                        "regulation": "specific regulation requiring control",
                        "impact": "regulatory/compliance impact",
                        "risk_rating": "High/Medium/Low"
                    }}
                ],
                "extra_steps": [
                    {{
                        "control": "unauthorized control step",
                        "risk": "introduced compliance risk",
                        "impact": "control framework impact",
                        "mitigation_required": "True/False"
                    }}
                ],
                "sequence_issues": [
                    {{
                        "control_sequence": "affected control sequence",
                        "expected_order": ["correct control order"],
                        "actual_order": ["executed control order"],
                        "compliance_impact": "regulatory impact description"
                    }}
                ],
                "timing_violations": [
                    {{
                        "control_sla": "control or regulatory SLA",
                        "requirement": "specific timing requirement",
                        "actual_execution": "actual timing",
                        "breach_severity": "High/Medium/Low",
                        "regulatory_impact": "description of regulatory impact"
                    }}
                ],
                "object_violations": [
                    {{
                        "object_control": "violated object control",
                        "violation_type": "limit/validation/relationship",
                        "affected_objects": ["impacted objects"],
                        "control_requirement": "specific control requirement",
                        "regulatory_impact": "compliance impact"
                    }}
                ],
                "compliance_gaps": [
                    {{
                        "regulation": "specific regulation",
                        "violation": "detailed violation description",
                        "control_failure": "associated control failure",
                        "risk_level": "High/Medium/Low",
                        "remediation_priority": "1-5 (1 highest)"
                    }}
                ],
                "severity_score": "numeric score based on control and compliance impact (0-100)",
                "recommendations": [
                    {{
                        "control_enhancement": "control improvement recommendation",
                        "compliance_category": "Regulatory/Control Framework/Risk",
                        "implementation_urgency": "Immediate/Short-term/Long-term",
                        "expected_compliance_benefit": "expected regulatory compliance improvement"
                    }}
                ]
            }}
            """

            logger.debug("Calling Azure OpenAI for control and compliance analysis...")
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a financial compliance and control expert focusing on FX trading regulations and controls."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            analysis = json.loads(response.choices[0].message.content)

            # Add case metadata
            analysis['case_id'] = case['case_id']
            analysis['timestamp_range'] = {
                'start': min(case['timestamps']).isoformat(),
                'end': max(case['timestamps']).isoformat()
            }
            analysis['process_context'] = self._get_process_context(activities)

            logger.info(f"Completed control and compliance analysis for case {case['case_id']}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing case {case['case_id']}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_report(self, output_path: str = "ocpm_output/gap_analysis.csv") -> pd.DataFrame:
        """Generate gap analysis report"""
        logger.info(f"=== Generating gap analysis report to {output_path} ===")

        try:
            # Get process flows
            logger.debug("Retrieving process flows...")
            cases = self._get_process_flows()
            logger.info(f"Retrieved {len(cases)} cases for analysis")

            # Analyze each case
            results = []
            logger.debug("Starting case analysis...")
            for idx, case in enumerate(cases, 1):
                logger.debug(f"Analyzing case {idx}/{len(cases)}: {case['case_id']}")
                analysis = self._analyze_case(case)
                results.append({
                    "case_id": case["case_id"],
                    **analysis
                })
                logger.debug(f"Completed analysis for case {idx}")

            # Create DataFrame
            logger.debug("Creating DataFrame from results...")
            df = pd.DataFrame(results)
            logger.debug(f"DataFrame shape: {df.shape}")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if path has a directory component
                os.makedirs(output_dir, exist_ok=True)

            # Save to CSV
            logger.debug(f"Saving DataFrame to {output_path}...")
            df.to_csv(output_path, index=False)

            logger.info(f"Analysis report saved successfully to {output_path}")
            logger.debug(f"Report summary: {df.describe().to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def __del__(self):
        """Cleanup resources"""
        logger.debug("=== Cleaning up OCPMAnalyzer resources ===")
        try:
            if hasattr(self, 'neo4j_driver'):
                logger.debug("Closing Neo4j driver...")
                self.neo4j_driver.close()
                logger.debug("Neo4j driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    logger.info("=== Starting OCPM Gap Analysis ===")
    try:
        # Ensure output directory exists
        output_dir = "ocpm_output"
        logger.debug(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize analyzer
        guideline_path = "ocpm_data/expected_guidelines.txt"
        logger.info(f"Initializing analyzer with guidelines: {guideline_path}")
        analyzer = OCPMAnalyzer(guideline_path)

        # Generate report
        output_path = os.path.join(output_dir, "gap_analysis.csv")
        logger.info(f"Generating report to: {output_path}")
        report = analyzer.generate_report(output_path)

        logger.info(f"Analysis complete. Processed {len(report)} cases.")
        logger.info(f"Report saved to {output_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise