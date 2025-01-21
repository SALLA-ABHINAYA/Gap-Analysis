# process_gap_analysis.py
from collections import defaultdict

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
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualizationExplainer:
    """AI-powered explainer for process analytics visualizations"""

    def __init__(self, openai_client):
        self.client = openai_client
        self.logger = logging.getLogger(__name__)

    def explain_severity_distribution(self, data: Dict) -> str:
        """Explain severity distribution chart"""
        try:
            prompt = f"""
            Analyze this severity distribution of process gaps:
            High Severity: {data['high_severity']} gaps
            Medium Severity: {data['medium_severity']} gaps
            Low Severity: {data['low_severity']} gaps
            Total Gaps: {data['total_gaps']} gaps

            Provide a concise, 2-3 sentence explanation of:
            1. The distribution pattern
            2. What it indicates about process health
            3. Key areas needing attention

            Focus on business impact and actionable insights.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an expert process analyst explaining gap severity patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error explaining severity distribution: {str(e)}")
            return "Unable to generate explanation at this time."

    def explain_coverage_metrics(self, metrics: Dict) -> str:
        """Explain coverage radar chart"""
        try:
            prompt = f"""
            Analyze these process coverage metrics:
            Regulatory Coverage: {metrics['compliance']['regulatory_coverage']}%
            Process Adherence: {metrics['operational']['process_adherence']}%
            Control Coverage: {metrics['risk']['control_coverage']}%
            Risk Assessment: {metrics['risk']['risk_assessment_completion']}%

            Provide a concise, 2-3 sentence explanation of:
            1. Overall coverage effectiveness
            2. Areas of strength and weakness
            3. Immediate improvement opportunities

            Focus on practical implications and key priorities.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert process analyst explaining coverage metrics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error explaining coverage metrics: {str(e)}")
            return "Unable to generate explanation at this time."

    def explain_gap_heatmap(self, data: Dict) -> str:
        """Explain gap heatmap visualization"""
        try:
            categories = list(data.keys())
            severities = ['High', 'Medium', 'Low']
            distribution = {cat: {sev: data[cat].get(sev, 0) for sev in severities} for cat in categories}

            prompt = f"""
            Analyze this gap distribution heatmap across categories:
            {json.dumps(distribution, indent=2)}

            Provide a concise, 2-3 sentence explanation of:
            1. Key patterns and clusters
            2. Most critical gap areas
            3. Notable category-severity relationships

            Focus on meaningful patterns and their business implications.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an expert process analyst explaining gap distribution patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error explaining gap heatmap: {str(e)}")
            return "Unable to generate explanation at this time."

    def explain_timeline_view(self, recommendations: List[Dict]) -> str:
        """Explain timeline visualization"""
        try:
            priority_counts = defaultdict(int)
            timeline_ranges = defaultdict(list)

            for rec in recommendations:
                priority_counts[rec['priority']] += 1
                timeline_ranges[rec['priority']].append(rec['target_date'])

            prompt = f"""
            Analyze this recommendation timeline distribution:
            Priority Distribution: {dict(priority_counts)}
            Timeline Ranges: {dict(timeline_ranges)}

            Provide a concise, 2-3 sentence explanation of:
            1. Priority distribution pattern
            2. Timeline clustering and phases
            3. Critical implementation considerations

            Focus on execution planning and resource allocation implications.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an expert process analyst explaining implementation timelines."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error explaining timeline view: {str(e)}")
            return "Unable to generate explanation at this time."

class GapAnalysisAI:
    """AI-powered analysis for process gaps"""

    def __init__(self, client):
        """Initialize with OpenAI client"""
        self.client = client
        self.logger = logging.getLogger(__name__)

    def analyze_gap_metrics(self, metrics: Dict) -> Dict:
        """Generate AI analysis for gap metrics"""
        try:
            prompt = f"""
            Analyze these process gap metrics:

            Compliance Metrics:
            - Regulatory Coverage: {metrics['compliance']['regulatory_coverage']}%
            - Control Effectiveness: {metrics['compliance']['control_effectiveness']}%

            Operational Metrics:
            - Process Adherence: {metrics['operational']['process_adherence']}%
            - Activity Completion: {metrics['operational']['activity_completion']}%

            Risk Metrics:
            - Control Coverage: {metrics['risk']['control_coverage']}%
            - Risk Assessment Completion: {metrics['risk']['risk_assessment_completion']}%

            Provide analysis in this format:
            1. Key findings about compliance, operational efficiency, and risk management
            2. Main areas of concern
            3. Specific recommendations for improvement
            4. Priority actions to address gaps

            Format response as JSON with sections: findings, concerns, recommendations, and priorities.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a process compliance and risk management expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"Error in gap metrics analysis: {str(e)}")
            return {
                "error": f"Failed to analyze metrics: {str(e)}",
                "findings": ["Analysis currently unavailable"],
                "concerns": ["System error occurred"],
                "recommendations": ["Retry analysis"],
                "priorities": ["Check system status"]
            }

    def analyze_gap_patterns(self, gaps: List[Dict]) -> Dict:
        """Generate AI analysis for identified gap patterns"""
        try:
            # Group gaps by category
            categorized_gaps = defaultdict(list)
            for gap in gaps:
                categorized_gaps[gap['category']].append(gap)

            prompt = f"""
            Analyze these process gaps across categories:

            {json.dumps(categorized_gaps, indent=2)}

            Consider:
            1. Common patterns across categories
            2. Severity distribution
            3. Potential root causes
            4. Impact on business operations

            Format response as JSON with sections:
            - patterns: List of identified patterns
            - root_causes: List of potential causes
            - impact_analysis: Business impact assessment
            - mitigation_strategies: Recommended actions
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a process mining and compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"Error in gap pattern analysis: {str(e)}")
            return {
                "patterns": ["Analysis unavailable"],
                "root_causes": ["Error occurred"],
                "impact_analysis": ["Unable to assess"],
                "mitigation_strategies": ["System check required"]
            }

    def generate_detailed_recommendations(self, gap_report: Dict) -> List[Dict]:
        """Generate detailed, AI-powered recommendations based on gap analysis"""
        try:
            prompt = f"""
            Based on this gap analysis report:
            {json.dumps(gap_report, indent=2)}

            Generate detailed recommendations considering:
            1. Implementation complexity
            2. Resource requirements
            3. Expected impact
            4. Timeline for implementation
            5. Dependencies and prerequisites

            Format each recommendation as a JSON object with:
            - description: Detailed description
            - complexity: High/Medium/Low
            - resources: Required resources
            - impact: Expected benefits
            - timeline: Implementation timeline
            - dependencies: List of prerequisites
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a process improvement and implementation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            recommendations = json.loads(response.choices[0].message.content)

            # Add unique IDs and timestamps
            for i, rec in enumerate(recommendations):
                rec['id'] = f"REC_{i + 1:03d}"
                rec['timestamp'] = datetime.now().isoformat()

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return [{"error": f"Failed to generate recommendations: {str(e)}"}]

    def summarize_gap_report(self, report: Dict) -> Dict:
        """Generate executive summary of gap analysis report"""
        try:
            prompt = f"""
            Summarize this gap analysis report:
            {json.dumps(report, indent=2)}

            Provide:
            1. Executive summary (2-3 sentences)
            2. Key metrics summary
            3. Critical findings
            4. Urgent actions required

            Format as JSON with these sections.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a process analytics expert presenting to executives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {
                "executive_summary": f"Error generating summary: {str(e)}",
                "key_metrics": [],
                "critical_findings": [],
                "urgent_actions": []
            }


def display_ai_insights(report: Dict):
    """Display AI-powered insights in the gap analysis dashboard"""
    if 'executive_summary' in report:
        st.subheader("🤖 AI-Powered Executive Summary")
        with st.expander("View Executive Summary", expanded=True):
            st.write(report['executive_summary']['executive_summary'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Key Metrics")
                for metric in report['executive_summary']['key_metrics']:
                    st.write(f"• {metric}")

            with col2:
                st.markdown("### Critical Findings")
                for finding in report['executive_summary']['critical_findings']:
                    st.write(f"• {finding}")

            st.markdown("### Urgent Actions")
            for action in report['executive_summary']['urgent_actions']:
                st.error(f"🚨 {action}")

    if 'ai_metrics_analysis' in report:
        st.subheader("📊 Metrics Analysis")
        with st.expander("View Detailed Metrics Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Key Findings")
                for finding in report['ai_metrics_analysis']['findings']:
                    st.write(f"• {finding}")

                st.markdown("### Areas of Concern")
                for concern in report['ai_metrics_analysis']['concerns']:
                    st.warning(f"⚠️ {concern}")

            with col2:
                st.markdown("### Recommendations")
                for rec in report['ai_metrics_analysis']['recommendations']:
                    st.write(f"• {rec}")

                st.markdown("### Priority Actions")
                for action in report['ai_metrics_analysis']['priorities']:
                    st.error(f"🚨 {action}")

    if 'ai_pattern_analysis' in report:
        st.subheader("🔍 Pattern Analysis")
        with st.expander("View Pattern Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Identified Patterns")
                for pattern in report['ai_pattern_analysis']['patterns']:
                    st.write(f"• {pattern}")

                st.markdown("### Root Causes")
                for cause in report['ai_pattern_analysis']['root_causes']:
                    st.write(f"• {cause}")

            with col2:
                st.markdown("### Business Impact")
                for impact in report['ai_pattern_analysis']['impact_analysis']:
                    st.warning(f"📈 {impact}")

                st.markdown("### Mitigation Strategies")
                for strategy in report['ai_pattern_analysis']['mitigation_strategies']:
                    st.success(f"🎯 {strategy}")

    if 'ai_recommendations' in report:
        st.subheader("💡 Detailed Recommendations")
        with st.expander("View Detailed Recommendations"):
            for rec in report['ai_recommendations']:
                with st.container():
                    st.markdown(f"### {rec['description']}")
                    cols = st.columns(4)

                    with cols[0]:
                        st.metric("Complexity", rec['complexity'])
                    with cols[1]:
                        st.metric("Timeline", rec['timeline'])
                    with cols[2]:
                        st.metric("Impact", rec['impact'])
                    with cols[3]:
                        st.metric("Resources", len(rec['resources']))

                    if rec['dependencies']:
                        st.markdown("#### Dependencies")
                        for dep in rec['dependencies']:
                            st.write(f"• {dep}")

                    st.markdown("---")

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