import streamlit as st
from neo4j import GraphDatabase
import traceback
import json
import pandas as pd
import os

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

    if st.button("Run Analysis"):
        try:
            results = run_neo4j_analysis(st.session_state.neo4j_credentials)
            display_analysis_results(results)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

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