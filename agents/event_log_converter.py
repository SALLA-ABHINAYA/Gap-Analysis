import streamlit as st
import pandas as pd
import sys
import os

from simulator import simulate_process


def analyze_event_log(df):
    """Generate process description from event log"""
    activities = df['activity'].unique()
    variants = df.groupby('case_id')['activity'].agg(list).value_counts()
    resources = df['resource'].unique() if 'resource' in df.columns else []

    log_summary = f"""
    Process contains {len(activities)} activities:
    {', '.join(activities)}

    Top 5 process variants:
    {variants.head().to_string()}

    Resources involved: {len(resources)}
    Additional attributes: {[col for col in df.columns if col not in ['case_id', 'activity', 'timestamp', 'resource']]}

    Please analyze this process and:
    1. Identify potential object types (e.g., orders, items, resources)
    2. Describe object relationships and lifecycle
    3. Consider batching/splitting patterns
    4. Note synchronization points
    """
    return log_summary


# Modified Streamlit interface
st.title("Event Log to OCEL Converter")

with st.form("converter_form"):
    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=['csv'])
    api_key = st.text_input("API Key:", type="password")
    desc_model = st.text_input("Description Model:", value="gpt-4")
    sim_model = st.text_input("Simulation Model:", value="gpt-4")
    submitted = st.form_submit_button("Convert")

if submitted and uploaded_file and api_key:
    # Read and analyze event log
    df = pd.read_csv(uploaded_file, sep=';')
    process_desc = analyze_event_log(df)

    # Save description for simulator
    with open("agents/target_process.txt", "w") as f:
        f.write(process_desc)

    # Use existing simulator with analyzed process
    try:
        simulate_process(
            target_process=process_desc,
            api_key=api_key,
            description_generation_model=desc_model,
            simulation_generation_model=sim_model,
            output_file="agents/ocel_output.xml",
            simulation_script="agents/conversion_script.py"
        )

        # Provide download link
        if os.path.exists("agents/ocel_output.xml"):
            with open("agents/ocel_output.xml", "rb") as file:
                st.download_button(
                    label="Download OCEL File",
                    data=file,
                    file_name="converted_log.xml",
                    mime="application/octet-stream"
                )
    except Exception as e:
        st.error(f"Conversion failed: {str(e)}")