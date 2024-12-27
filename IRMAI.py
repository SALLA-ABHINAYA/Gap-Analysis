"""
IRMAI Process Analytics with multi-page structure
"""
# IRMAI.py (main app)
import streamlit as st

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

