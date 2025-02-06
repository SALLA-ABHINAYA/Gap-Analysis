import streamlit as st 
from r2r import R2RClient
from PyPDF2 import PdfReader
import os
import time

def graph_creation(context):
    client = R2RClient("http://20.68.198.68:7272")
    ingest_response = client.documents.create(
                    file_path=context,
                    id=None)
    print(ingest_response)
    document_id = ingest_response['results']['document_id']
    print(document_id)
    
    time.sleep(10)
    extract_response = client.documents.extract(document_id)
    
    entities = client.documents.list_entities(document_id)
    relationships = client.documents.list_relationships(document_id)

    

def main():
    st.set_page_config(layout="wide")

    st.title("Context Upload")


    uploaded_file = st.file_uploader("Please upload a PDF or TXT file", type=['pdf','txt'])

    if uploaded_file is not None:

        try:
            with st.spinner("Processing your file..."):
                path = os.path.join("temp", uploaded_file.name)  # Create temp directory
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                graph_creation(path)
                
            st.success("Successfully Ingested")

        except Exception as e:
            st.error(f"Error analyzing the file: {str(e)}")
            st.info("Please ensure your file is either a PDF or txt file.")
        
        


if __name__ == "__main__":
    main()