import streamlit as st
from integrated_ocpm import create_integrated_ocpm_ui

st.set_page_config(
    page_title="IRMAI Process Analytics",
    page_icon="👋",
    layout="wide"
)


def main():
    st.write("# Welcome to IRMAI Process Analytics! 👋")

    # Create sidebar navigation
    st.sidebar.success("Select an analytics option.")

    # Add analytics options in sidebar
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type",
        ["Regular Process Mining", "Integrated OCPM Analysis"]
    )

    if analysis_type == "Regular Process Mining":
        st.write("## Regular Process Mining")
        # Your existing process mining code/functionality here
        st.write("Upload a process log to begin analysis")

    elif analysis_type == "Integrated OCPM Analysis":
        st.write("## Integrated Object-Centric Process Mining")
        # Call the integrated OCPM UI
        create_integrated_ocpm_ui()


if __name__ == "__main__":
    main()