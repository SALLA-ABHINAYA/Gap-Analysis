# app_ocpm.py

import streamlit as st
from ocpm_analysis import create_ocpm_ui
import os
from typing import Optional
import pandas as pd

# Set page config at the very beginning
st.set_page_config(
    page_title="IRMAI Process Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("")

def setup_ocpm_page():
    """Set up the OCPM analysis page"""
    st.title("📊 IRMAI OCPM Analytics")
    st.info("""Object-Centric Process Mining (OCPM) provides a more detailed view of process interactions 
            by considering multiple object types and their relationships.""")

    # Create directories if they don't exist
    os.makedirs("ocpm_data", exist_ok=True)
    os.makedirs("ocpm_output", exist_ok=True)

    # Add OCPM-specific UI
    create_ocpm_ui()

    # Add explanation of object types
    with st.expander("Understanding Object Types in FX Trading"):
        st.write("""
        The OCPM analysis considers four main object types:

        1. **Trade Objects**: Represent the core trading activities
           - Trade execution
           - Trade allocation
           - Trade confirmation

        2. **Market Objects**: Handle market-related activities
           - Market data validation
           - Quote management
           - Volatility analysis

        3. **Risk Objects**: Manage risk-related activities
           - Risk assessment
           - Strategy validation
           - Greeks calculations

        4. **Settlement Objects**: Handle settlement activities
           - Premium settlement
           - Position reconciliation
           - Collateral management
        """)

    # Add documentation
    with st.expander("How to Use OCPM Analysis"):
        st.write("""
        1. Upload your event log CSV file
        2. View object interactions through the heatmap
        3. Analyze object metrics in the dashboard
        4. Explore object lifecycles for specific object types

        The analysis will show:
        - How different objects interact
        - Distribution of activities across object types
        - Object lifecycle patterns
        - Performance metrics per object type
        """)


def main():
    # Create sidebar for navigation
    # Run OCPM analysis UI
    setup_ocpm_page()


if __name__ == "__main__":
    main()