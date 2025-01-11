# process_gap_analysis.py
import streamlit as st
import pandas as pd
import json
import logging
from pathlib import Path
import traceback
from datetime import datetime
from openai import OpenAI
from neo4j import GraphDatabase
from typing import Dict, List, Any, Union
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessGapAnalyzer:
    """Process Gap Analysis using Neo4j and OpenAI"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize the analyzer with database and API credentials"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = (neo4j_user, neo4j_password)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=self.neo4j_auth)


        self.openai_client = OpenAI(
            api_key="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"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()

    def setup_database(self):
        """Create necessary Neo4j constraints and indexes"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT org_context_id IF NOT EXISTS FOR (n:OrgContext) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT guideline_id IF NOT EXISTS FOR (n:Guideline) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE",
                "CREATE INDEX event_timestamp IF NOT EXISTS FOR (n:Event) ON (n.timestamp)",
                "CREATE INDEX event_activity IF NOT EXISTS FOR (n:Event) ON (n.activity)"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.error(f"Error creating constraint: {str(e)}")
                    raise

    def load_org_context(self, file_obj: Union[str, io.BytesIO, 'StreamlitUploadedFile']):
        """Load organizational context from CSV into Neo4j"""
        try:
            # Handle different file input types
            if isinstance(file_obj, str):
                df = pd.read_csv(file_obj)
            else:
                df = pd.read_csv(file_obj)

            # Create Cypher query for org context
            query = """
            UNWIND $rows AS row
            MERGE (n:OrgContext {id: row.id})
            SET n += row
            """

            with self.driver.session() as session:
                session.run(query, rows=df.to_dict('records'))

        except Exception as e:
            logger.error(f"Error loading org context: {str(e)}")
            raise

    def load_guidelines(self, file_obj: Union[str, io.BytesIO, 'StreamlitUploadedFile']):
        """Load standard guidelines from CSV into Neo4j"""
        try:
            # Handle different file input types
            if isinstance(file_obj, str):
                df = pd.read_csv(file_obj)
            else:
                df = pd.read_csv(file_obj)

            query = """
            UNWIND $rows AS row
            MERGE (n:Guideline {id: row.id})
            SET n += row
            """

            with self.driver.session() as session:
                session.run(query, rows=df.to_dict('records'))

        except Exception as e:
            logger.error(f"Error loading guidelines: {str(e)}")
            raise

    def load_ocel_events(self, file_obj: Union[str, io.BytesIO, 'StreamlitUploadedFile']):
        """Load OCEL events from JSON into Neo4j"""
        try:
            # Handle different file input types
            if isinstance(file_obj, str):
                with open(file_obj, 'r') as f:
                    ocel_data = json.load(f)
            else:
                ocel_data = json.loads(file_obj.getvalue().decode('utf-8'))

            # Process events
            events = []
            for event in ocel_data['ocel:events']:
                event_data = {
                    'id': event['ocel:id'],
                    'timestamp': event['ocel:timestamp'],
                    'activity': event['ocel:activity'],
                    'type': event['ocel:type'],
                    'resource': event.get('ocel:attributes', {}).get('resource', 'Unknown'),
                    'case_id': event.get('ocel:attributes', {}).get('case_id', 'Unknown'),
                    'object_type': event.get('ocel:attributes', {}).get('object_type', 'Unknown'),
                    'objects': json.dumps([obj['id'] for obj in event.get('ocel:objects', [])])
                }
                events.append(event_data)

            # Create Cypher query for events
            query = """
            UNWIND $events AS event
            MERGE (e:Event {id: event.id})
            SET e += event
            """

            with self.driver.session() as session:
                session.run(query, events=events)

        except Exception as e:
            logger.error(f"Error loading OCEL events: {str(e)}")
            raise

    def analyze_process_gaps(self) -> Dict[str, Any]:
        """Analyze process gaps using Neo4j data and OpenAI"""
        try:
            with self.driver.session() as session:
                # Get organization context
                org_context_result = session.run("""
                    MATCH (n:OrgContext)
                    WITH n
                    ORDER BY n.id
                    RETURN collect({
                        id: n.id,
                        department: n.department,
                        role: n.role,
                        responsibilities: n.responsibilities
                    }) as context
                """)
                org_context = org_context_result.single()
                if org_context is None:
                    org_context = {'context': []}
                org_context = org_context['context']

                # Get guidelines
                guidelines_result = session.run("""
                    MATCH (n:Guideline)
                    WITH n
                    ORDER BY n.id
                    RETURN collect({
                        id: n.id,
                        category: n.category,
                        guideline: n.guideline,
                        priority: n.priority
                    }) as guidelines
                """)
                guidelines = guidelines_result.single()
                if guidelines is None:
                    guidelines = {'guidelines': []}
                guidelines = guidelines['guidelines']

                # Get events - Fixed query with proper ordering
                events_result = session.run("""
                    MATCH (e:Event)
                    WITH e
                    ORDER BY e.timestamp
                    RETURN collect({
                        id: e.id,
                        timestamp: e.timestamp,
                        activity: e.activity,
                        type: e.type,
                        resource: e.resource,
                        case_id: e.case_id
                    }) as events
                """)
                events = events_result.single()
                if events is None:
                    events = {'events': []}
                events = events['events']

                # Calculate additional metrics
                activity_metrics_result = session.run("""
                    MATCH (e:Event)
                    WITH e.activity as activity, count(*) as count
                    RETURN collect({activity: activity, count: count}) as metrics
                """)
                activity_metrics = activity_metrics_result.single()['metrics']

                resource_metrics_result = session.run("""
                    MATCH (e:Event)
                    WITH e.resource as resource, count(*) as count
                    RETURN collect({resource: resource, count: count}) as metrics
                """)
                resource_metrics = resource_metrics_result.single()['metrics']

                # Basic statistics
                stats = {
                    'org_context_count': len(org_context),
                    'guidelines_count': len(guidelines),
                    'events_count': len(events),
                    'activity_metrics': {m['activity']: m['count'] for m in activity_metrics},
                    'resource_metrics': {m['resource']: m['count'] for m in resource_metrics}
                }

                # If no data is found, return early with empty analysis
                if not any([org_context, guidelines, events]):
                    return {
                        'timestamp': datetime.now().isoformat(),
                        'analysis': "No data found in the database. Please ensure data is properly loaded.",
                        'data_summary': stats,
                        'status': 'no_data'
                    }

                # Prepare analysis prompt
                prompt = f"""
                Analyze the FX trading process data:

                Organization Context ({len(org_context)} items):
                {json.dumps(org_context[:5], indent=2)}

                Guidelines ({len(guidelines)} items):
                {json.dumps(guidelines[:5], indent=2)}

                Process Events ({len(events)} events):
                {json.dumps(events[:5], indent=2)}

                Process Statistics:
                - Total Events: {stats['events_count']}
                - Organizational Units: {stats['org_context_count']}
                - Guidelines: {stats['guidelines_count']}
                - Activity Distribution: {json.dumps(stats['activity_metrics'], indent=2)}
                - Resource Distribution: {json.dumps(stats['resource_metrics'], indent=2)}

                Please provide:
                1. Compliance Analysis:
                   - Check adherence to guidelines
                   - Identify potential violations
                   - Assess control effectiveness

                2. Process Patterns:
                   - Identify common sequences
                   - Highlight unusual patterns
                   - Assess process efficiency

                3. Resource Utilization:
                   - Analyze workload distribution
                   - Identify bottlenecks
                   - Assess capacity issues

                4. Recommendations:
                   - Process improvements
                   - Control enhancements
                   - Resource optimization
                """

                # Get analysis from OpenAI
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system",
                             "content": "You are a process mining and compliance expert specializing in FX trading operations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    analysis_text = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    analysis_text = f"Error generating AI analysis: {str(e)}"

                # Return complete analysis
                return {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis_text,
                    'data_summary': stats,
                    'status': 'success',
                    'details': {
                        'activity_metrics': stats['activity_metrics'],
                        'resource_metrics': stats['resource_metrics']
                    }
                }

        except Exception as e:
            logger.error(f"Error in gap analysis: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'analysis': f"Error during analysis: {str(e)}",
                'data_summary': {
                    'org_context_count': 0,
                    'guidelines_count': 0,
                    'events_count': 0
                },
                'status': 'error'
            }

def create_gap_analysis_ui():
    """Create Streamlit UI for gap analysis"""
    st.title("Process Gap Analysis")

    # Configuration inputs
    with st.expander("Database Configuration"):
        neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7689")
        neo4j_user = st.text_input("Neo4j Username", "neo4j")
        neo4j_password = st.text_input("Neo4j Password", type="password")

    # File uploads
    org_context_file = st.file_uploader("Upload Organizational Context CSV", type=['csv'])
    guidelines_file = st.file_uploader("Upload Guidelines CSV", type=['csv'])
    ocel_file = st.file_uploader("Upload OCEL JSON", type=['json'])

    if st.button("Run Analysis"):
        try:
            with ProcessGapAnalyzer(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password
            ) as analyzer:
                # Setup database
                with st.spinner("Setting up database..."):
                    analyzer.setup_database()

                # Load data
                if org_context_file is not None:
                    with st.spinner("Loading organizational context..."):
                        analyzer.load_org_context(org_context_file)

                if guidelines_file is not None:
                    with st.spinner("Loading guidelines..."):
                        analyzer.load_guidelines(guidelines_file)

                if ocel_file is not None:
                    with st.spinner("Loading OCEL events..."):
                        analyzer.load_ocel_events(ocel_file)

                # Run analysis
                with st.spinner("Analyzing process gaps..."):
                    analysis_results = analyzer.analyze_process_gaps()

                # Display results based on status
                if analysis_results['status'] == 'success':
                    st.success("Analysis completed!")
                elif analysis_results['status'] == 'no_data':
                    st.warning("No data available for analysis")
                else:
                    st.error("Analysis completed with errors")

                # Display analysis results
                st.subheader("Analysis Results")
                st.write(analysis_results['analysis'])

                # Display data summary if available
                if 'data_summary' in analysis_results:
                    st.subheader("Data Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Org Context Items",
                                analysis_results['data_summary']['org_context_count'])
                    with col2:
                        st.metric("Guidelines",
                                analysis_results['data_summary']['guidelines_count'])
                    with col3:
                        st.metric("Process Events",
                                analysis_results['data_summary']['events_count'])

        except Exception as e:
            st.error("Error during data import: " + str(e))
            st.error(f"Detailed error:\n{traceback.format_exc()}")

if __name__ == "__main__":
    st.set_page_config(page_title="Process Gap Analysis", page_icon="📊", layout="wide")
    create_gap_analysis_ui()