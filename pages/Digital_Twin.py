import streamlit as st
from neo4j import GraphDatabase
import traceback
import json
import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from IRMAI import FXTradingGapAnalyzer, GapAnalysisVisualizer

st.set_page_config(
    page_title="Digital Twin",
    page_icon="👋",
    layout="wide"
)

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
    """Handle graph analytics operations with detailed logging and AI explanations"""
    logger.info("Starting graph analytics handler")

    st.subheader("Gap Analysis")

    if not hasattr(st.session_state, 'neo4j_connected') or not st.session_state.neo4j_connected:
        logger.warning("Neo4j not connected")
        st.warning("Please connect to Neo4j and import data first")
        return

    credentials = st.session_state.neo4j_credentials
    logger.info("Found Neo4j credentials")

    if st.button("Run Comprehensive Analysis"):
        try:
            logger.info("Starting comprehensive analysis")
            with st.spinner("Running comprehensive gap analysis..."):
                analyzer = FXTradingGapAnalyzer(
                    uri=credentials['uri'],
                    user=credentials['user'],
                    password=credentials['password']
                )

                report = analyzer.generate_gap_report()
                visualizer = GapAnalysisVisualizer(report)
                dashboard = visualizer.generate_dashboard()

                # Display Overview Metrics with AI Explanations
                st.subheader("Overview Analysis")
                metrics_col1, metrics_col2 = st.columns([2, 1])

                with metrics_col1:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Gaps", report['summary']['total_gaps'])
                    with col2:
                        st.metric("High Severity", report['summary']['high_severity'])
                    with col3:
                        st.metric("Process Coverage",
                                  f"{report['metrics']['operational']['process_adherence']:.1f}%")
                    with col4:
                        st.metric("Control Effectiveness",
                                  f"{report['metrics']['risk']['control_coverage']:.1f}%")

                with metrics_col2:
                    st.markdown("### 🤖 AI Insights")
                    st.markdown(dashboard['explanations'].get('overview', 'No overview analysis available'))

                # Display Visualizations with AI Explanations
                st.subheader("Detailed Analysis")

                # Severity Distribution
                viz_col1, viz_col2 = st.columns([2, 1])
                with viz_col1:
                    severity_fig = dashboard['figures'].get('severity_distribution')
                    if severity_fig:
                        st.plotly_chart(severity_fig, use_container_width=True)
                with viz_col2:
                    st.markdown("### 🤖 Severity Analysis")
                    st.markdown(dashboard['explanations'].get('severity_distribution',
                                                              'No severity analysis available'))

                # Coverage Analysis
                viz_col3, viz_col4 = st.columns([2, 1])
                with viz_col3:
                    coverage_fig = dashboard['figures'].get('coverage_radar')
                    if coverage_fig:
                        st.plotly_chart(coverage_fig, use_container_width=True)
                with viz_col4:
                    st.markdown("### 🤖 Coverage Analysis")
                    st.markdown(dashboard['explanations'].get('coverage_radar',
                                                              'No coverage analysis available'))

                # Gap Heatmap
                viz_col5, viz_col6 = st.columns([2, 1])
                with viz_col5:
                    heatmap_fig = dashboard['figures'].get('gap_heatmap')
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                with viz_col6:
                    st.markdown("### 🤖 Pattern Analysis")
                    st.markdown(dashboard['explanations'].get('gap_heatmap',
                                                              'No pattern analysis available'))

                # Timeline View
                viz_col7, viz_col8 = st.columns([2, 1])
                with viz_col7:
                    timeline_fig = dashboard['figures'].get('timeline_view')
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                with viz_col8:
                    st.markdown("### 🤖 Timeline Analysis")
                    st.markdown(dashboard['explanations'].get('timeline_view',
                                                              'No timeline analysis available'))

                # Recommendations Section
                st.subheader("AI-Powered Recommendations")
                if 'recommendations' in report:
                    for idx, rec in enumerate(report['recommendations'], 1):
                        with st.expander(f"Recommendation {idx}: {rec.get('description', 'No description')}"):
                            st.markdown(f"**Priority:** {rec.get('priority', 'N/A')}")
                            st.markdown(f"**Impact:** {rec.get('impact', 'N/A')}")
                            st.markdown(f"**Timeline:** {rec.get('target_date', 'N/A')}")

                logger.info("Analysis display completed")
                analyzer.close()

        except Exception as e:
            logger.error(f"Analysis failed with error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(f"Analysis failed: {str(e)}")
            st.error(traceback.format_exc())

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