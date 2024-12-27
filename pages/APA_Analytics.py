 #pages/APA_Analytics.py

import streamlit as st
from ocpm_analysis import create_ocpm_ui
import os
from Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer
from pathlib import Path


def setup_ocpm_page():
    """Set up the OCPM analysis page with integrated unfair analysis."""

    st.title("📊 IRMAI APA Analytics")

    st.info("""APA provides a detailed view of process interactions by considering multiple object types and their relationships.""")

    # Create directories if they don't exist
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)

    # Create tabs for different analysis
    main_tabs = st.tabs(["Process Analysis", "Unfairness Analysis"])

    with main_tabs[0]:
        # Add OCPM-specific UI
        create_ocpm_ui()

        # Save OCEL path in session state after successful processing
        if 'ocpm_df' in st.session_state:
            ocel_path = Path("ocpm_output/process_data.json")
            st.session_state['ocel_path'] = str(ocel_path)

        with st.expander("Understanding Object Types in FX Trading"):
            st.write("""
            The APA analysis considers four main object types:
            1. **Trade Objects**: Core trading activities
            2. **Market Objects**: Market-related activities
            3. **Risk Objects**: Risk-related activities
            4. **Settlement Objects**: Settlement activities
            """)

    with main_tabs[1]:
        run_unfairness_analysis()


def find_ocel_file():
    """Find the OCEL file in the expected locations."""
    possible_paths = [
        "ocpm_output/process_data.json",
        "ocpm_data/process_data.json"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def run_unfairness_analysis():
    """Run unfairness analysis using the enhanced UnfairOCELAnalyzer."""

    st.subheader("Unfairness Analysis")

    # First check session state for OCEL path
    ocel_path = st.session_state.get('ocel_path')

    # If not in session state, try to find it
    if not ocel_path:
        ocel_path = find_ocel_file()

    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        analyzer = UnfairOCELAnalyzer(ocel_path)
        plots, metrics = analyzer.get_analysis_plots()

        # Show the analysis results with traceability
        tabs = st.tabs([
            "Resource Discrimination",
            "Time Bias",
            "Case Priority",
            "Handover Patterns"
        ])

        with tabs[0]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['resource_discrimination'])
            with col2:
                st.markdown("### Resource Bias Findings")
                for resource, data in metrics['resource'].items():
                    if isinstance(data, dict) and data.get('bias_score', 0) > 0.2:
                        st.warning(f"⚠️ {resource}: {data['bias_score']:.2f} bias score")
                        # Add trace exploration
                        if st.button(f"Show traces for {resource}"):
                            trace_report = analyzer.generate_trace_report('resource', resource)
                            st.text(trace_report)

        with tabs[1]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['time_bias'])
            with col2:
                st.markdown("### Processing Time Statistics")
                for resource, data in metrics['time'].items():
                    if isinstance(data, dict):
                        with st.expander(f"{resource} Details"):
                            st.write(f"Mean time: {data.get('mean_time', 0):.2f} hours")
                            st.write(f"Std Dev: {data.get('std_dev', 0):.2f} hours")
                            # Add trace exploration
                            if st.button(f"Show traces for {resource} times"):
                                trace_report = analyzer.generate_trace_report('resource', resource)
                                st.text(trace_report)

        with tabs[2]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['case_priority'])
            with col2:
                st.markdown("### Case Priority Analysis")
                for case_type, data in metrics['case'].items():
                    if isinstance(data, dict) and abs(data.get('bias_score', 0)) > 0.2:
                        status = "👎" if data.get('bias_score', 0) > 0 else "👍"
                        st.info(f"{case_type}: {status}")
                        # Add trace exploration
                        if st.button(f"Show traces for {case_type}"):
                            trace_report = analyzer.generate_trace_report('case', case_type)
                            st.text(trace_report)

        with tabs[3]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['handover'])
            with col2:
                st.markdown("### Significant Handovers")
                for pattern, data in metrics['handover'].items():
                    if isinstance(data, dict) and data.get('percentage', 0) > 10:
                        st.write(f"🔄 {pattern}: {data['percentage']:.1f}%")
                        # Add trace exploration
                        if st.button(f"Show traces for {pattern}"):
                            trace_report = analyzer.generate_trace_report('handover', pattern)
                            st.text(trace_report)

    except Exception as e:
        st.error(f"Error in unfairness analysis: {str(e)}")
        import traceback
        st.error(f"Detailed error:\n{traceback.format_exc()}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="IRMAI APA Analytics",
        page_icon="📊",
        layout="wide"
    )
    setup_ocpm_page()