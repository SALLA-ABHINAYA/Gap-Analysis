# IRMAI.py - Static Landing Page
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
    ## Intelligent Resource Management AI (IRMAI)

    IRMAI is a comprehensive process analytics platform that combines traditional process mining with 
    advanced AI capabilities for deeper insights into your business processes.

    ### Core Features:

    1. **Advanced Process Analytics (APA)**
       - Object-centric process mining
       - Multi-dimensional process analysis
       - Interactive visualizations
       - Real-time process metrics

    2. **AI-Powered Insights**
       - Natural language process queries
       - Automated pattern detection
       - Intelligent anomaly detection
       - Predictive analytics

    3. **Unfairness Detection**
       - Resource allocation analysis
       - Workload distribution metrics
       - Bias detection algorithms
       - Fair process recommendations

    ### Getting Started:

    1. Select **APA Analytics** from the sidebar to begin your analysis
    2. Upload your process data in the Process Analysis tab
    3. Explore insights across multiple dimensions
    4. Use AI capabilities to deep dive into your processes

    ### Data Requirements:

    - OCEL (Object-Centric Event Log) format
    - Event timestamps
    - Activity labels
    - Resource information
    - Case identifiers

    For detailed documentation and support, visit our [documentation](#) or contact the support team.
    """
)

# Bottom section with metrics/stats
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Supported Object Types",
        value="4+",
        help="Trade, Market, Risk, Settlement, and more"
    )

with col2:
    st.metric(
        label="Analysis Capabilities",
        value="10+",
        help="Including process mining, AI analysis, and unfairness detection"
    )

with col3:
    st.metric(
        label="Visualization Types",
        value="15+",
        help="Interactive charts, process maps, and statistical visualizations"
    )

# Additional resources section
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Resources
- [Documentation](#)
- [User Guide](#)
- [API Reference](#)
""")