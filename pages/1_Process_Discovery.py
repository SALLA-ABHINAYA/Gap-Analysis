# app.py
import shutil

import streamlit as st
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
import os, json
import plotly.graph_objects as go
from process_mining_enhanced import FXProcessMining
from risk_analysis import ProcessRiskAnalyzer, EnhancedFMEA


# Set up Graphviz path based on environment
azure_file_path = os.getenv("AZURE_FILE_PATH")
if azure_file_path:
    # We're in Azure
    os.environ["PATH"] = os.environ["PATH"] + ":" + azure_file_path
else:
    # We're in Windows local environment
    graphviz_path = "C:\\samadhi\\technology\\Graphviz\\bin"
    os.environ["PATH"] = os.environ["PATH"] + ";" + graphviz_path


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'output', 'staging']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Create directories
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)


def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    create_directories()
    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def analyze_risks(event_log, bpmn_graph):
    """Perform risk analysis on process model"""
    try:
        # Initialize ProcessRiskAnalyzer
        risk_analyzer = ProcessRiskAnalyzer(event_log, bpmn_graph)
        risk_analyzer.analyze_bpmn_graph()

        # Initialize EnhancedFMEA
        fmea = EnhancedFMEA(
            failure_modes=risk_analyzer.failure_modes,
            activity_stats=risk_analyzer.activity_stats
        )

        # Get risk assessment results
        risk_assessment = fmea.assess_risk()

        # Calculate process metrics
        process_metrics = {
            'total_activities': len(risk_analyzer.activity_stats),
            'high_risk_activities': len([r for r in risk_assessment if r['rpn'] > 200]),
            'medium_risk_activities': len([r for r in risk_assessment if 100 < r['rpn'] <= 200]),
            'low_risk_activities': len([r for r in risk_assessment if r['rpn'] <= 100])
        }

        return risk_assessment, process_metrics

    except Exception as e:
        st.error(f"Error in risk assessment: {str(e)}")
        raise


def visualize_risk_distribution(risk_assessment_results):
    """Create visualization of risk distribution"""
    activities = [r['failure_mode'] for r in risk_assessment_results]
    rpn_values = [r['rpn'] for r in risk_assessment_results]
    severities = [r['severity'] for r in risk_assessment_results]
    likelihoods = [r['likelihood'] for r in risk_assessment_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=severities,
        y=likelihoods,
        mode='markers',
        marker=dict(
            size=[r['rpn'] * 5 for r in risk_assessment_results],
            color=rpn_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="RPN")
        ),
        text=activities,
        hovertemplate="<b>Activity:</b> %{text}<br>" +
                      "<b>Severity:</b> %{x:.2f}<br>" +
                      "<b>Likelihood:</b> %{y:.2f}<br>" +
                      "<b>RPN:</b> %{marker.color:.2f}<br>"
    ))

    fig.update_layout(
        title="Risk Distribution Matrix",
        xaxis_title="Severity",
        yaxis_title="Likelihood",
        showlegend=False
    )

    return fig


# In 1_Process_Discovery.py

def main():
    st.set_page_config(page_title="IRMAI Process Discovery", page_icon="📊", layout="wide")

    # Header and Instructions
    st.title("📊 Process Discovery")
    st.info("Upload your Event Log to perform Process Discovery.")

    # File Upload
    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            with st.spinner('Processing data...'):
                # Save and analyze file
                file_path = save_uploaded_file(uploaded_file)
                bpmn_graph, event_log = process_mining_analysis(file_path)

            st.success('Analysis completed successfully!')

            # Create tabs for different analysis sections
            tabs = st.tabs(["Process Discovery"])

            # Process Mining Tab
            with tabs[0]:
                st.subheader("Process Discovery")

                # Display process visualizations
                st.subheader("Process Visualizations")
                viz_tabs = st.tabs(["BPMN", "Petri Net", "Process Tree"])
                with viz_tabs[0]:
                    st.image("output/fx_trade_bpmn.png")
                with viz_tabs[1]:
                    st.image("output/fx_trade_petri_net.png")
                with viz_tabs[2]:
                    st.image("output/fx_trade_process_tree.png")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure:")
            st.write("1. Your CSV file has the required columns: case_id, activity, timestamp")
            st.write("2. The data format is correct")
            st.write("3. Graphviz is installed and in your system PATH")


def process_mining_analysis(csv_path):
    """Perform process mining analysis and save CSV for later use"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs("ocpm_output", exist_ok=True)

        # Copy uploaded file to output directory for use by Outlier Analysis
        output_csv_path = os.path.join("ocpm_output", "event_log.csv")
        shutil.copy2(csv_path, output_csv_path)

        # Initialize FX Process Mining
        fx_miner = FXProcessMining(csv_path)
        fx_miner.preprocess_data()
        fx_miner.discover_process()

        # Get process model components
        process_tree = fx_miner.process_tree
        petri_net = fx_miner.process_model
        initial_marking = fx_miner.initial_marking
        final_marking = fx_miner.final_marking
        event_log = fx_miner.event_log

        # Generate visualizations
        pn_gviz = pn_visualizer.apply(petri_net, initial_marking, final_marking)
        pn_visualizer.save(pn_gviz, "output/fx_trade_petri_net.png")

        pt_gviz = pt_visualizer.apply(process_tree)
        pt_visualizer.save(pt_gviz, "output/fx_trade_process_tree.png")

        bpmn_graph = pm4py.convert_to_bpmn(process_tree)
        bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
        bpmn_visualizer.save(bpmn_gviz, "output/fx_trade_bpmn.png")

        return bpmn_graph, event_log

    except Exception as e:
        st.error(f"Error in process analytics analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        raise

main()
