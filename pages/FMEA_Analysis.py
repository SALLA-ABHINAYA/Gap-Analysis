import streamlit as st
import pandas as pd
import os
import traceback
from fmea_analyzer import OCELFMEAAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_fmea_analysis():
    """Display FMEA analysis using existing OCEL data"""
    try:
        # Use existing OCEL path
        ocel_path = "ocpm_output/process_data.json"

        if not os.path.exists(ocel_path):
            st.error("OCEL data not found. Please run process analysis first.")
            return

        analyzer = OCELFMEAAnalyzer(ocel_path)

        with st.spinner("Analyzing failure modes..."):
            analyzer.identify_failure_modes()
            report = analyzer.generate_report()

            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Failure Modes", report['summary']['total_failure_modes'])
            with col2:
                st.metric("High Risk Issues", report['summary']['high_risk_count'])
            with col3:
                st.metric("Medium Risk Issues", report['summary']['medium_risk_count'])

            # Display failure modes analysis
            st.subheader("Failure Modes Analysis")
            failure_modes_df = pd.DataFrame(report['failure_modes'])
            st.dataframe(failure_modes_df)

            # Display recommendations
            st.subheader("Recommendations")
            recommendations_df = pd.DataFrame(report['recommendations'])
            st.dataframe(recommendations_df)

    except Exception as e:
        logger.error(f"Error performing FMEA analysis: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error performing FMEA analysis: {str(e)}")

# Main page content
st.title("FMEA Analysis")
show_fmea_analysis()