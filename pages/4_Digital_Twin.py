import streamlit as st
from neo4j import GraphDatabase
import traceback
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import pandas as pd
import os
import networkx as nx
import json
import pandas as pd
from typing import Dict, List, Tuple
import logging
import logging

from pygments.lexers import go

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from IRMAI import FXTradingGapAnalyzer, GapAnalysisVisualizer

st.set_page_config(
    page_title="Digital Twin",
    page_icon="👋",
    layout="wide"
)


from json import JSONEncoder
from datetime import datetime
from neo4j.time import DateTime

class Neo4jJsonEncoder(JSONEncoder):
    """Custom JSON encoder for Neo4j types"""
    def default(self, obj):
        if isinstance(obj, DateTime):
            # Convert Neo4j DateTime to ISO format string
            return obj.to_native().isoformat()
        if isinstance(obj, datetime):
            # Convert Python datetime to ISO format string
            return obj.isoformat()
        return super().default(obj)


class GraphExtractorAnalyzer:
    """Extracts graph data from Neo4j and performs gap analysis"""

    def __init__(self, driver):
        self.driver = driver
        self.logger = logging.getLogger(__name__)

    def extract_graph_to_file(self) -> str:
        """
        Extract graph data from Neo4j and save to file
        Returns: Path to saved file
        """
        try:
            with self.driver.session() as session:
                # Extract nodes with timestamp handling
                nodes_query = """
                MATCH (n)
                RETURN DISTINCT
                    id(n) as id,
                    labels(n) as labels,
                    properties(n) as properties
                """
                nodes_result = session.run(nodes_query)

                # Process nodes, ensuring datetime conversion
                nodes = []
                for record in nodes_result:
                    node_dict = dict(record)
                    # Ensure properties is a dict
                    if 'properties' not in node_dict:
                        node_dict['properties'] = {}
                    nodes.append(node_dict)

                # Extract relationships
                rels_query = """
                MATCH (s)-[r]->(t)
                RETURN DISTINCT
                    id(s) as source_id,
                    id(t) as target_id,
                    type(r) as type,
                    properties(r) as properties
                """
                rels_result = session.run(rels_query)

                # Process relationships, ensuring datetime conversion
                relationships = []
                for record in rels_result:
                    rel_dict = dict(record)
                    # Ensure properties is a dict
                    if 'properties' not in rel_dict:
                        rel_dict['properties'] = {}
                    relationships.append(rel_dict)

                # Create graph structure
                graph_data = {
                    'nodes': nodes,
                    'relationships': relationships
                }

                # Save to file using custom encoder
                output_path = 'ocpm_output/fx_trade_graph.json'
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2, cls=Neo4jJsonEncoder)

                return output_path

        except Exception as e:
            self.logger.error(f"Error extracting graph: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def load_graph_from_file(self, file_path: str) -> nx.DiGraph:
        """Load graph from saved JSON file into NetworkX"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        G = nx.DiGraph()

        # Add nodes
        for node in data['nodes']:
            # Convert properties to ensure they're all native Python types
            properties = node.get('properties', {})
            # Parse any ISO format dates back to datetime
            for key, value in properties.items():
                if isinstance(value, str) and 'T' in value:
                    try:
                        properties[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        pass  # Keep as string if not a valid datetime

            G.add_node(
                node['id'],
                labels=node['labels'],
                **properties
            )

        # Add edges
        for rel in data['relationships']:
            # Convert relationship properties
            properties = rel.get('properties', {})
            for key, value in properties.items():
                if isinstance(value, str) and 'T' in value:
                    try:
                        properties[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        pass

            G.add_edge(
                rel['source_id'],
                rel['target_id'],
                type=rel['type'],
                **properties
            )

        return G

    def analyze_graph_gaps(self, G: nx.DiGraph, guidelines_path: str) -> Dict:
        """
        Analyze gaps between actual graph and guidelines
        Args:
            G: NetworkX graph of actual process
            guidelines_path: Path to guidelines JSON
        Returns:
            Dictionary of gap analysis results
        """
        try:
            self.logger.info("Starting graph gap analysis")
            with open(guidelines_path, 'r') as f:
                guidelines = json.load(f)

            # Initialize gap tracking
            gaps = {
                'missing_activities': [],
                'missing_controls': [],
                'timing_violations': [],
                'sequence_violations': [],
                'relationship_violations': []
            }

            # Get nodes by type
            activity_nodes = [(n, d) for n, d in G.nodes(data=True)
                              if 'Activity' in d.get('labels', [])]
            control_nodes = [(n, d) for n, d in G.nodes(data=True)
                             if 'Control' in d.get('labels', [])]

            # Track existing activities
            actual_activities = {
                d.get('properties', {}).get('name'): n
                for n, d in activity_nodes
                if d.get('properties', {}).get('name')  # Only include if name exists
            }

            # Check for missing activities from each object type
            for obj_type, obj_data in guidelines.items():
                required_activities = set(obj_data.get('activities', []))
                existing = set(actual_activities.keys())
                missing = required_activities - existing

                if missing:
                    gaps['missing_activities'].append({
                        'object_type': obj_type,
                        'missing': list(missing)
                    })

            # Check control coverage - Fixed version
            control_to_activities = {}
            for n, d in control_nodes:
                control_name = d.get('properties', {}).get('name')
                if control_name:
                    # Get monitored activities for this control from edges
                    monitored = []
                    for _, target in G.edges(n):
                        target_data = G.nodes[target].get('properties', {})
                        target_name = target_data.get('name')
                        if target_name:
                            monitored.append(target_name)
                    control_to_activities[control_name] = monitored

            # Check each activity for control coverage
            for activity_name in actual_activities:
                is_controlled = False
                for control_activities in control_to_activities.values():
                    if activity_name in control_activities:
                        is_controlled = True
                        break

                if not is_controlled:
                    gaps['missing_controls'].append(activity_name)

            # Check timing thresholds
            with open('ocpm_output/output_ocel_threshold.json', 'r') as f:
                thresholds = json.load(f)

            for activity_name, activity_node in actual_activities.items():
                activity_data = G.nodes[activity_node].get('properties', {})
                duration = activity_data.get('duration_hours')

                if duration is not None:  # Only check if we have duration data
                    for obj_type, obj_data in thresholds.items():
                        activity_thresholds = obj_data.get('activity_thresholds', {})
                        if activity_name in activity_thresholds:
                            max_duration = activity_thresholds[activity_name].get('max_duration_hours')
                            if max_duration and duration > max_duration:
                                gaps['timing_violations'].append({
                                    'activity': activity_name,
                                    'actual_duration': duration,
                                    'threshold': max_duration,
                                    'object_type': obj_type
                                })

            # Check sequence violations
            for obj_type, obj_data in guidelines.items():
                expected_sequence = obj_data.get('activities', [])
                if not expected_sequence:
                    continue

                # Get actual sequence from graph paths
                for start_activity in actual_activities:
                    if start_activity == expected_sequence[0]:
                        path = []
                        current = actual_activities[start_activity]
                        visited = set()  # Prevent cycles

                        while current is not None and current not in visited:
                            visited.add(current)
                            activity_name = G.nodes[current].get('properties', {}).get('name')
                            if activity_name:
                                path.append(activity_name)

                            # Get next activity in path
                            successors = [s for s in G.successors(current)
                                          if s not in visited and
                                          G.nodes[s].get('properties', {}).get('name')]
                            current = successors[0] if successors else None

                        if path and path != expected_sequence:
                            gaps['sequence_violations'].append({
                                'object_type': obj_type,
                                'expected': expected_sequence,
                                'actual': path
                            })

            # Check relationship violations
            for obj_type, obj_data in guidelines.items():
                required_relationships = set(obj_data.get('relationships', []))
                actual_relationships = set()

                # Get nodes of this object type
                obj_nodes = [n for n, d in G.nodes(data=True)
                             if d.get('properties', {}).get('name') == obj_type]

                # Get actual relationships from graph
                for obj_node in obj_nodes:
                    for _, neighbor in G.edges(obj_node):
                        neighbor_type = G.nodes[neighbor].get('properties', {}).get('name')
                        if neighbor_type:
                            actual_relationships.add(neighbor_type)

                missing_relationships = required_relationships - actual_relationships
                if missing_relationships:
                    gaps['relationship_violations'].append({
                        'object_type': obj_type,
                        'missing_relationships': list(missing_relationships)
                    })

            self.logger.info("Completed graph gap analysis")
            return gaps

        except Exception as e:
            self.logger.error(f"Error in graph gap analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def generate_graph_metrics(self, G: nx.DiGraph) -> Dict:
        """
        Calculate comprehensive graph metrics
        Args:
            G: NetworkX graph
        Returns:
            Dictionary of graph metrics
        """
        try:
            self.logger.info("Generating graph metrics")
            metrics = {
                'basic': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
                },
                'complexity': {
                    'cyclomatic': G.number_of_edges() - G.number_of_nodes() + 2,
                    'connectivity': nx.node_connectivity(G) if G.number_of_nodes() > 1 else 0,
                    'avg_clustering': nx.average_clustering(G.to_undirected()),
                    'diameter': nx.diameter(G) if nx.is_strongly_connected(G) else None
                },
                'centrality': {
                    'degree': self._get_top_nodes(nx.degree_centrality(G)),
                    'betweenness': self._get_top_nodes(nx.betweenness_centrality(G)),
                    'closeness': self._get_top_nodes(nx.closeness_centrality(G))
                },
                'structural': {
                    'strongly_connected_components': len(list(nx.strongly_connected_components(G))),
                    'weakly_connected_components': len(list(nx.weakly_connected_components(G))),
                    'avg_path_length': nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else None
                },
                'flow': {
                    'bottlenecks': self._identify_bottlenecks(G),
                    'parallel_paths': self._count_parallel_paths(G)
                }
            }

            self.logger.info("Completed generating graph metrics")
            return metrics

        except Exception as e:
            self.logger.error(f"Error generating graph metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _get_top_nodes(self, centrality_dict: Dict, top_n: int = 5) -> List[Dict]:
        """Get top N nodes by centrality measure"""
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        return [{'node': n, 'score': s} for n, s in sorted_nodes[:top_n]]

    def _identify_bottlenecks(self, G: nx.DiGraph) -> List[Dict]:
        """Identify potential bottlenecks in the graph"""
        betweenness = nx.betweenness_centrality(G)
        # Nodes with high betweenness and low out-degree are potential bottlenecks
        bottlenecks = []
        for node, score in betweenness.items():
            if score > 0.5 and G.out_degree(node) < 2:
                bottlenecks.append({
                    'node': node,
                    'betweenness': score,
                    'out_degree': G.out_degree(node)
                })
        return bottlenecks

    def _count_parallel_paths(self, G: nx.DiGraph) -> Dict:
        """Count parallel paths between activities"""
        parallel_paths = {}
        nodes = list(G.nodes())
        for i, source in enumerate(nodes):
            for target in nodes[i + 1:]:
                try:
                    paths = list(nx.all_simple_paths(G, source, target))
                    if len(paths) > 1:
                        parallel_paths[f"{source}->{target}"] = len(paths)
                except:
                    continue
        return parallel_paths

def handle_data_import():
    """Handle data import operations"""
    st.subheader("Data Import")

    # Database Configuration
    with st.expander("Neo4j Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7689")
        with col2:
            neo4j_user = st.text_input("Username", "neo4j")
        with col3:
            neo4j_password = st.text_input("Password", type="password")

        if st.button("Test Connection"):
            try:
                driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                with driver.session() as session:
                    session.run("RETURN 1").single()
                st.success("Connected to Neo4j successfully!")
                st.session_state.neo4j_connected = True
                st.session_state.neo4j_credentials = {
                    'uri': neo4j_uri,
                    'user': neo4j_user,
                    'password': neo4j_password
                }
                driver.close()
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                st.session_state.neo4j_connected = False

    # File Upload Section
    st.subheader("Process Data Files")

    org_context = st.file_uploader(
        "Upload Organizational Context (CSV)",
        type=['csv'],
        key="org_context"
    )

    guidelines = st.file_uploader(
        "Upload Guidelines (CSV)",
        type=['csv'],
        key="guidelines"
    )

    ocel_file = st.file_uploader(
        "Upload OCEL JSON",
        type=['json'],
        key="ocel"
    )

    if st.button("Import to Neo4j"):
        if not (org_context and guidelines and ocel_file):
            st.warning("Please upload all required files")
            return

        try:
            import_to_neo4j(
                st.session_state.neo4j_credentials,
                org_context,
                guidelines,
                ocel_file
            )
            st.success("Data imported successfully!")
        except Exception as e:
            st.error(f"Import failed: {str(e)}")
            st.error(traceback.format_exc())


def handle_graph_analytics():
    """Handle graph analytics operations with mock ML analytics"""
    logger.info("Starting analytics dashboard")

    st.subheader("FX Trading Analytics Dashboard")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="ML Model Accuracy",
            value="94.5%",
            delta="1.2%"
        )

    with col2:
        st.metric(
            label="Anomalies Detected",
            value="27",
            delta="-5",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Pattern Confidence",
            value="87.2%",
            delta="0.5%"
        )

    with col4:
        st.metric(
            label="Trading Volume",
            value="$1.2B",
            delta="$0.1B"
        )

    # Mock data
    mock_volume = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=6, freq='M'),
        'EUR': [4000, 3000, 2000, 2780, 1890, 2390],
        'USD': [2400, 1398, 9800, 3908, 4800, 3800],
        'GBP': [2400, 2210, 2290, 2000, 2181, 2500]
    })

    mock_patterns = pd.DataFrame({
        'time': ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00'],
        'volume': [30, 45, 35, 50, 25, 40],
        'volatility': [65, 55, 85, 45, 60, 75]
    })

    mock_anomalies = pd.DataFrame({
        'interval': range(1, 7),
        'normal': [40, 30, 20, 27, 18, 23],
        'anomaly': [24, 13, 38, 15, 42, 19]
    })

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Trading Volume", "ML Patterns", "Anomalies"])

    with tab1:
        st.subheader("Currency Trading Volume Analysis")
        # Using plotly express for volume analysis
        fig_volume = px.line(mock_volume,
                             x='date',
                             y=['EUR', 'USD', 'GBP'],
                             title="Trading Volume by Currency")
        fig_volume.update_layout(
            height=400,
            template='plotly_dark',
            yaxis_title="Volume",
            xaxis_title="Date"
        )
        st.plotly_chart(fig_volume, use_container_width=True)

    with tab2:
        st.subheader("ML-Detected Trading Patterns")
        # Using plotly express for patterns
        fig_patterns = px.area(mock_patterns,
                               x='time',
                               y=['volume', 'volatility'],
                               title="Trading Patterns Over Time")
        fig_patterns.update_layout(
            height=400,
            template='plotly_dark',
            yaxis_title="Value",
            xaxis_title="Time"
        )
        st.plotly_chart(fig_patterns, use_container_width=True)

    with tab3:
        st.subheader("Anomaly Detection")
        # Using plotly express for anomalies
        fig_anomalies = px.bar(mock_anomalies,
                               x='interval',
                               y=['normal', 'anomaly'],
                               title="Trade Anomaly Distribution",
                               barmode='group')
        fig_anomalies.update_layout(
            height=400,
            template='plotly_dark',
            yaxis_title="Count",
            xaxis_title="Time Interval"
        )
        st.plotly_chart(fig_anomalies, use_container_width=True)

    # Add explanatory text
    st.markdown("""
    ### Key ML Insights
    - **Pattern Recognition**: Machine learning models have identified recurring patterns in currency pair movements
    - **Anomaly Detection**: AI algorithms flagged 27 potential unusual trading patterns in the last 24 hours
    - **Volume Analysis**: Peak trading activity correlates with European market hours
    - **Risk Assessment**: ML models provide real-time risk scoring for each trade
    """)

    # Add confidence scores
    with st.expander("🎯 Model Performance Metrics"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            - Model Accuracy: 94.5%
            - False Positive Rate: 2.3%
            - Pattern Recognition: 87.2%
            """)

        with col2:
            st.markdown("""
            - Processing Latency: <100ms
            - Data Coverage: 99.9%
            - Model Confidence: High
            """)

    st.info(
        "Note: This is a demonstration using simulated data to showcase ML/AI capabilities in FX trading analytics.")

def import_to_neo4j(credentials, org_file, guidelines_file, ocel_file):
    """Import data to Neo4j database"""
    driver = GraphDatabase.driver(
        credentials['uri'],
        auth=(credentials['user'], credentials['password'])
    )

    try:
        # Import organizational context
        org_data = pd.read_csv(org_file)
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (o:Organization {id: row.id})
                SET o += row
            """, rows=org_data.to_dict('records'))

        # Import guidelines
        guide_data = pd.read_csv(guidelines_file)
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (g:Guideline {id: row.id})
                SET g += row
            """, rows=guide_data.to_dict('records'))

        # Import OCEL data
        ocel_data = json.load(ocel_file)
        events = [
            {
                'id': e['ocel:id'],
                'timestamp': e['ocel:timestamp'],
                'activity': e['ocel:activity'],
                'resource': e.get('ocel:attributes', {}).get('resource', 'Unknown'),
                'case_id': e.get('ocel:attributes', {}).get('case_id', 'Unknown')
            }
            for e in ocel_data['ocel:events']
        ]

        with driver.session() as session:
            session.run("""
                UNWIND $events AS event
                MERGE (e:Event {id: event.id})
                SET e += event
            """, events=events)

    finally:
        driver.close()

def run_neo4j_analysis(credentials):
    """Run analysis queries in Neo4j"""
    driver = GraphDatabase.driver(
        credentials['uri'],
        auth=(credentials['user'], credentials['password'])
    )

    try:
        results = {}
        with driver.session() as session:
            # Process flow analysis
            process_flow = session.run("""
                MATCH (e1:Event)-[r:NEXT]->(e2:Event)
                RETURN e1.activity as source, 
                       e2.activity as target, 
                       count(*) as frequency
                ORDER BY frequency DESC
            """)
            results['process_flow'] = process_flow.data()

            # Resource analysis
            resource_analysis = session.run("""
                MATCH (e:Event)
                WITH e.resource as resource, count(*) as events
                RETURN resource, events
                ORDER BY events DESC
            """)
            results['resources'] = resource_analysis.data()

        return results

    finally:
        driver.close()

def display_analysis_results(results):
    """Display analysis results"""
    if not results:
        st.warning("No analysis results available")
        return

    tabs = st.tabs(["Process Flow", "Resource Analysis"])

    with tabs[0]:
        st.subheader("Process Flow Analysis")
        if results.get('process_flow'):
            st.dataframe(pd.DataFrame(results['process_flow']))

    with tabs[1]:
        st.subheader("Resource Analysis")
        if results.get('resources'):
            st.dataframe(pd.DataFrame(results['resources']))

# Main page content
st.title("Digital Twin")
tab1, tab2 = st.tabs(["Import Data", "Gap Analysis"])

with tab1:
    handle_data_import()
with tab2:
    handle_graph_analytics()