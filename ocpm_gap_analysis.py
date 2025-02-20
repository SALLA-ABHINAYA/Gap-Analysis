"""
OCPM Gap Analysis Tool
- Connects to Neo4j Aura DB
- Processes OCEL data
- Compares against guidelines using Azure OpenAI
- Generates CSV report
"""

import os
import json
import pandas as pd
from neo4j import GraphDatabase
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

from utils import get_azure_openai_client

# Configure detailed logging
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
                # First clear existing data
                session.run("MATCH (n) DETACH DELETE n")

                # Create constraints
                session.run("CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE")
                session.run("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")

                # Create Cases and Events
                for event in ocel_data.get('ocel:events', []):
                    case_id = event.get('ocel:attributes', {}).get('case_id')
                    if not case_id:
                        continue

                    # Create Case if not exists
                    session.run("""
                        MERGE (c:Case {id: $case_id})
                        ON CREATE SET c.created_at = datetime()
                    """, case_id=case_id)

                    # Create Event
                    session.run("""
                        MATCH (c:Case {id: $case_id})
                        CREATE (e:Event {
                            id: $event_id,
                            activity: $activity,
                            timestamp: datetime($timestamp),
                            object_type: $object_type
                        })
                        CREATE (c)-[:HAS_EVENT]->(e)
                    """,
                                case_id=case_id,
                                event_id=event['ocel:id'],
                                activity=event['ocel:activity'],
                                timestamp=event['ocel:timestamp'],
                                object_type=event.get('ocel:attributes', {}).get('object_type', 'Unknown')
                                )

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

    def _load_guidelines(self, path: str) -> str:
        """Load process guidelines from file"""
        logger.debug(f"=== Loading guidelines from {path} ===")
        try:
            if not os.path.exists(path):
                logger.error(f"Guidelines file not found: {path}")
                raise FileNotFoundError(f"Guidelines file not found: {path}")

            with open(path, 'r') as f:
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
        """Analyze gaps in individual case execution"""
        logger.debug(f"=== Analyzing case {case['case_id']} ===")
        logger.debug(f"Case details: {json.dumps(case, indent=2, default=str)}")

        try:
            # Construct the prompt
            prompt = f"""
            Analyze process execution gaps using these guidelines:
            {self.guidelines}

            Actual Execution:
            - Case ID: {case['case_id']}
            - Activities: {case['activities']}
            - Timestamps: {case['timestamps']}
            - Object Types: {case['object_types']}

            Identify:
            1. Missing required steps
            2. Unexpected activities
            3. Sequence violations
            4. Timing violations
            5. Object interaction issues
            6. Compliance gaps

            Return JSON with:
            {{
                "missing_steps": [],
                "extra_steps": [],
                "sequence_issues": [],
                "timing_violations": [],
                "object_violations": [],
                "compliance_gaps": [],
                "severity_score": 0-100,
                "recommendations": []
            }}
            """
            logger.debug(f"Generated prompt length: {len(prompt)}")
            logger.debug(f"First 200 chars of prompt: {prompt[:200]}...")

            logger.debug("Calling Azure OpenAI...")
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            logger.debug(f"Received response from Azure OpenAI")

            analysis = json.loads(response.choices[0].message.content)
            logger.debug(f"Parsed analysis result: {json.dumps(analysis, indent=2)}")

            logger.info(f"Completed analysis for case {case['case_id']}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing case {case['case_id']}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_report(self, output_path: str = "gap_analysis.csv") -> pd.DataFrame:
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
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

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