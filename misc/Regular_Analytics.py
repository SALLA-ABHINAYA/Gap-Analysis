# app.py
import streamlit as st
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
import os


import plotly.graph_objects as go
from process_mining_enhanced import FXProcessMining
from risk_analysis import ProcessRiskAnalyzer, EnhancedFMEA

# # Set up Graphviz path
azure_file_path = os.getenv("AZURE_FILE_PATH")
os.environ["PATH"] += azure_file_path


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'output', 'staging']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    create_directories()
    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_mining_analysis(csv_path):
    """Perform comprehensive process mining analysis"""
    try:
        # Initialize FX Process Mining
        fx_miner = FXProcessMining(csv_path)
        fx_miner.preprocess_data()
        fx_miner.discover_process()

        # Get process mining results
        mining_results = fx_miner.generate_report()

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

        # Convert to BPMN and visualize
        bpmn_graph = pm4py.convert_to_bpmn(process_tree)
        bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
        bpmn_visualizer.save(bpmn_gviz, "output/fx_trade_bpmn.png")

        # Get process statistics with error handling
        process_stats = mining_results.get('process_statistics', {})
        start_activities = [
            {'activity': act, 'count': count}
            for act, count in list(process_stats.get('start_activities', {}).items())[:5]
        ]
        end_activities = [
            {'activity': act, 'count': count}
            for act, count in list(process_stats.get('end_activities', {}).items())[-5:]
        ]

        # Get metrics with default values for error cases
        conformance = {
            'fitness': mining_results.get('conformance', {}).get('fitness', 0) * 100,  # Convert to percentage
            'completed_traces': mining_results.get('conformance', {}).get('completed_traces', 0),
            'total_traces': mining_results.get('conformance', {}).get('total_traces', 0)
        }

        performance = {
            'avg_case_duration': mining_results.get('performance', {}).get('avg_case_duration', 0),
            'median_case_duration': mining_results.get('performance', {}).get('median_case_duration', 0),
            'bottlenecks': mining_results.get('performance', {}).get('bottlenecks', [])
        }

        return (conformance, performance, start_activities, end_activities,
                bpmn_graph, event_log, mining_results)

    except Exception as e:
        st.error(f"Error in process analytics analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        raise


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


def main():
    st.set_page_config(page_title="IRMAI Process Analytics", page_icon="📊", layout="wide")

    # Header and Instructions
    st.title("📊 IRMAI Process Analytics")
    st.info("Upload your Transaction Log to run process analytics and risk analysis")

    # File Upload
    uploaded_file = st.file_uploader("Upload Transaction Log (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            with st.spinner('Processing data...'):
                # Save and analyze file
                file_path = save_uploaded_file(uploaded_file)
                results = process_mining_analysis(file_path)

                if not results:
                    st.error("Error processing file")
                    return

                (conformance, performance, start_activities, end_activities,
                 bpmn_graph, event_log, mining_results) = results

            st.success('Analysis completed successfully!')

            # Create tabs for different analysis sections
            tabs = st.tabs(["Process Analytics", "Risk Assessment", "Performance", "Improvements"])

            # Process Mining Tab
            with tabs[0]:
                st.subheader("Process Analytics Results")

                # Display conformance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Process Fitness", f"{conformance['fitness']:.2%}")
                with col2:
                    st.metric("Completed Traces", conformance['completed_traces'])
                with col3:
                    st.metric("Total Traces", conformance['total_traces'])

                # Display process visualizations
                st.subheader("Process Visualizations")
                viz_tabs = st.tabs(["BPMN", "Petri Net", "Process Tree"])
                with viz_tabs[0]:
                    st.image("output/fx_trade_bpmn.png")
                with viz_tabs[1]:
                    st.image("output/fx_trade_petri_net.png")
                with viz_tabs[2]:
                    st.image("output/fx_trade_process_tree.png")

            # Risk Assessment Tab
            with tabs[1]:
                st.subheader("Risk Analysis")
                risk_results, process_metrics = analyze_risks(event_log, bpmn_graph)

                # Display risk metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Risks Identified", len(risk_results))
                with col2:
                    avg_rpn = sum(r['rpn'] for r in risk_results) / len(risk_results)
                    st.metric("Average Risk Priority Number", f"{avg_rpn:.2f}")

                # Display risk distribution
                st.plotly_chart(visualize_risk_distribution(risk_results))

                # Display detailed risk analysis
                for risk in risk_results:
                    with st.expander(f"Risk: {risk['failure_mode']} (RPN: {risk['rpn']})"):
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Severity", f"{risk['severity']:.2f}")
                        with cols[1]:
                            st.metric("Likelihood", f"{risk['likelihood']:.2f}")
                        with cols[2]:
                            st.metric("Detectability", f"{risk['detectability']:.2f}")

            # Performance Tab
            with tabs[2]:
                st.subheader("Performance Analysis")

                # Display performance metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Case Duration",
                              f"{performance['avg_case_duration']:.2f} seconds")
                with col2:
                    st.metric("Median Case Duration",
                              f"{performance['median_case_duration']:.2f} seconds")

                # Display bottlenecks
                st.subheader("Process Bottlenecks")
                for bottleneck in performance['bottlenecks'][:5]:
                    st.warning(
                        f"**{bottleneck['source']} → {bottleneck['target']}**\n\n"
                        f"Average Duration: {bottleneck['avg_duration']:.2f} seconds\n\n"
                        f"Frequency: {bottleneck['frequency']} occurrences"
                    )

            # Improvements Tab
            with tabs[3]:
                st.subheader("Suggested Improvements")
                for improvement in mining_results['improvements']:
                    with st.expander(f"{improvement['type'].title()} Improvement"):
                        st.write(f"**Issue:** {improvement['issue']}")
                        st.write(f"**Recommendation:** {improvement['recommendation']}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure:")
            st.write("1. Your CSV file has the required columns: case_id, activity, timestamp")
            st.write("2. The data format is correct")
            st.write("3. Graphviz is installed and in your system PATH")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>© 2024 IRMAI. All rights reserved.</p>
            <p>Process Analytics & Risk Analysis Solutions</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
