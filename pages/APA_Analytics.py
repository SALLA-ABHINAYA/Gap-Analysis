# pages/APA_Analytics.py

import streamlit as st
from ocpm_analysis import create_ocpm_ui, OCPMAnalyzer
import os
from typing import Optional
import pandas as pd
from pages.Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer
import json
from pathlib import Path


def setup_ocpm_page():
    """Set up the OCPM analysis page with integrated unfair analysis."""
    st.title("📊 IRMAI APA Analytics")
    st.info("""APA provides a detailed view of process interactions by considering multiple object types and their relationships.""")

    # Create directories if they don't exist
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)

    # Create tabs for different analyses
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
    """Run unfairness analysis using the existing UnfairOCELAnalyzer."""
    st.subheader("Unfairness Analysis")

    # First check session state for OCEL path
    ocel_path = st.session_state.get('ocel_path')

    # If not in session state, try to find it
    if not ocel_path:
        ocel_path = find_ocel_file()

    # Debug information if needed
    with st.expander("Debug Information"):
        st.write(f"OCEL Path: {ocel_path}")
        st.write(f"Session State: {st.session_state}")
        if ocel_path:
            st.write(f"File exists: {os.path.exists(ocel_path)}")

    if not ocel_path or not os.path.exists(ocel_path):
        st.warning("⚠️ Please process data in the Process Analysis tab first.")
        return

    try:
        # Initialize the existing UnfairOCELAnalyzer
        analyzer = UnfairOCELAnalyzer(ocel_path)
        plots, metrics = analyzer.get_analysis_plots()

        # Show the analysis results using the existing tabs from UnfairOCELAnalyzer
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

        with tabs[3]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(plots['handover'])
            with col2:
                st.markdown("### Significant Handovers")
                for pattern, data in metrics['handover'].items():
                    if isinstance(data, dict) and data.get('percentage', 0) > 10:
                        st.write(f"🔄 {pattern}: {data['percentage']:.1f}%")

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