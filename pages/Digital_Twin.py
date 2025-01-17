import streamlit as st
from neo4j import GraphDatabase
import traceback
import json
import pandas as pd
import os

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
    """Handle graph analytics operations"""
    st.subheader("Graph Analytics")

    if not hasattr(st.session_state, 'neo4j_connected') or not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j and import data first")
        return

    analysis_tabs = st.tabs(["Process Flow", "Comprehensive Gap Analysis"])

    with analysis_tabs[1]:
        st.subheader("Comprehensive Gap Analysis")

        credentials = st.session_state.neo4j_credentials

        analysis_type = st.multiselect(
            "Select Analysis Types",
            ["Regulatory Compliance", "Process Execution", "Control Effectiveness"],
            default=["Regulatory Compliance"]
        )

        if st.button("Run Comprehensive Analysis"):
            try:
                with st.spinner("Running comprehensive gap analysis..."):
                    analyzer = FXTradingGapAnalyzer(
                        uri=credentials['uri'],
                        user=credentials['user'],
                        password=credentials['password']
                    )

                    # Generate report
                    report = analyzer.generate_gap_report()

                    # Create visualizer
                    visualizer = GapAnalysisVisualizer(report)
                    dashboard = visualizer.generate_interactive_dashboard()

                    # Display metrics
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

                    # Display AI Insights if available
                    if 'ai_findings' in report:
                        st.subheader("🤖 AI-Powered Insights")
                        for finding in report['ai_findings']:
                            with st.expander(f"{finding['category']} - {finding['severity']}"):
                                st.write(f"**Description:** {finding['description']}")
                                st.write(f"**Impact:** {finding['impact']}")
                                if finding.get('related_controls'):
                                    st.write("**Related Controls:**")
                                    for control in finding['related_controls']:
                                        st.write(f"- {control}")

                    # Display visualizations in two columns
                    st.subheader("Analysis Visualizations")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(dashboard['severity_distribution'],
                                        use_container_width=True)
                        st.plotly_chart(dashboard['gap_heatmap'],
                                        use_container_width=True)

                    with col2:
                        st.plotly_chart(dashboard['coverage_radar'],
                                        use_container_width=True)
                        st.plotly_chart(dashboard['timeline_view'],
                                        use_container_width=True)

                    # Display detailed recommendations
                    st.subheader("Detailed Recommendations")
                    visualizer.display_recommendations_table()

                    analyzer.close()

            except Exception as e:
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
tab1, tab2 = st.tabs(["Import Data", "Graph Analytics"])

with tab1:
    handle_data_import()
with tab2:
    handle_graph_analytics()