"""
IRMAI Process Analytics with multipage structure
"""
# IRMAI.py (main app)
import streamlit as st

from pages.Unfair_Advanced_Process_Logs_Analytics import UnfairOCELAnalyzer

st.set_page_config(
    page_title="IRMAI Process Analytics",
    page_icon="👋",
    layout="wide"
)

st.write("# Welcome to IRMAI Process Analytics! 👋")

st.sidebar.success("Select an analytics option above.")

st.markdown(
    """
    ### Available Analytics Options:

    1. **APA (Advanced Process Analytics)**
       - Object-centric process analysis
       - Detailed interaction visualization
       - Performance metrics

    2. **Unfairness Detection**
       - Resource discrimination analysis
       - Time bias detection
       - Case priority analysis
       - Handover pattern analysis

    Select an option from the sidebar to begin your analysis.

    ### Getting Started:
    1. Start with APA Analytics to process your event log
    2. Review unfairness analysis for deeper insights
    3. Download detailed reports for offline analysis
    """
)

# pages/1_APA_Analytics.py
import streamlit as st
from ocpm_analysis import create_ocpm_ui, OCPMAnalyzer
import os
from typing import Optional
import pandas as pd
from pathlib import Path




def setup_ocpm_page():
    """Set up the OCPM analysis page."""
    st.title("📊 IRMAI APA Analytics")
    st.info(
        """APA provides a detailed view of process interactions by considering multiple object types and their relationships.""")

    # Create directories if they don't exist
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)

    # Add OCPM-specific UI
    create_ocpm_ui()

    # Add object types explanation
    with st.expander("Understanding Object Types in FX Trading"):
        st.write("""
        The APA analysis considers four main object types:

        1. **Trade Objects**: Core trading activities
        2. **Market Objects**: Market-related activities
        3. **Risk Objects**: Risk-related activities
        4. **Settlement Objects**: Settlement activities
        """)


if __name__ == "__main__":
    setup_ocpm_page()

# pages/2_Unfairness_Analytics.py
import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt



def setup_unfairness_page():
    """Set up the unfairness analysis page."""
    st.title("🔍 Process Unfairness Analysis")
    st.info("""Analyze process fairness across resources, time, and case handling.""")

    ocel_path = "ocpm_output/process_data.json"

    if not Path(ocel_path).exists():
        st.warning("⚠️ Please run APA Analytics first to generate process data.")
        return

    try:
        with st.spinner('Analyzing process fairness...'):
            analyzer = UnfairOCELAnalyzer(ocel_path)
            plots, metrics = analyzer.get_analysis_plots()

            # Create tabs for different analyses
            analysis_tabs = st.tabs([
                "Resource Discrimination",
                "Time Bias",
                "Case Priority",
                "Handover Patterns"
            ])

            with analysis_tabs[0]:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.pyplot(plots['resource_discrimination'])
                with col2:
                    show_resource_metrics(metrics['resource'])

            with analysis_tabs[1]:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.pyplot(plots['time_bias'])
                with col2:
                    show_time_metrics(metrics['time'])

            with analysis_tabs[2]:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.pyplot(plots['case_priority'])
                with col2:
                    show_case_metrics(metrics['case'])

            with analysis_tabs[3]:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.pyplot(plots['handover'])
                with col2:
                    show_handover_metrics(metrics['handover'])

            # Add download button
            if st.button("📊 Generate Detailed Report"):
                report = generate_unfairness_report(metrics)
                st.download_button(
                    "📥 Download Report",
                    report,
                    "unfairness_analysis_report.txt",
                    "text/plain"
                )

    except Exception as e:
        st.error(f"Error in unfairness analysis: {str(e)}")


def show_resource_metrics(metrics):
    st.markdown("### Resource Bias Findings")
    for resource, data in metrics.items():
        if data['bias_score'] > 0.2:
            st.warning(f"⚠️ {resource}: {data['bias_score']:.2f} bias score")
        elif data['bias_score'] < -0.2:
            st.info(f"ℹ️ {resource}: {data['bias_score']:.2f} bias score")


def show_time_metrics(metrics):
    st.markdown("### Processing Time Statistics")
    for resource, data in metrics.items():
        with st.expander(f"{resource} Details"):
            st.metric("Mean Processing Time", f"{data['mean_time']:.2f} hrs")
            st.metric("Standard Deviation", f"{data['std_dev']:.2f} hrs")


def show_case_metrics(metrics):
    st.markdown("### Case Priority Analysis")
    for case_type, data in metrics.items():
        if abs(data.get('bias_score', 0)) > 0.2:
            status = "👎" if data.get('bias_score', 0) > 0 else "👍"
            st.info(f"{status} {case_type}")


def show_handover_metrics(metrics):
    st.markdown("### Significant Handovers")
    for pattern, data in metrics.items():
        if isinstance(data, dict) and data.get('percentage', 0) > 10:
            st.write(f"🔄 {pattern}: {data['percentage']:.1f}%")


def generate_unfairness_report(metrics):
    """Generate detailed unfairness analysis report."""
    report_lines = [
        "IRMAI Process Unfairness Analysis Report",
        "======================================"
    ]

    for analysis_type, data in metrics.items():
        report_lines.extend([
            f"\n{analysis_type.upper()} ANALYSIS",
            "-" * len(f"{analysis_type.upper()} ANALYSIS")
        ])

        for key, values in data.items():
            report_lines.append(f"\n{key}:")
            if isinstance(values, dict):
                for metric, value in values.items():
                    report_lines.append(f"  {metric}: {value}")
            else:
                report_lines.append(f"  Value: {values}")

    return "\n".join(report_lines)


if __name__ == "__main__":
    setup_unfairness_page()