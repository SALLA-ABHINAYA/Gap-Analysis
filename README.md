# pm4py-poc-1
pm4py-poc-1

## On going Runs
---
cd C:\samadhi\personal\side_hustle\IRMAI\workspace

pm4py-poc-1\Scripts\activate

## step 1 : Generate Synthetic FX Trade Event Log
----
python synthetic_data/synthetic_data_gr.py

## step 2 : Run Process Map & Risk Analysis
----
streamlit run app.py

streamlit run app_ocpm.py

streamlit run unfair_ocpm.py

streamlit run IRMAI.py

## step 3 : Gap Analysis
----
python gap_analysis.py




## 1 time setup
---
cd C:\samadhi\personal\side_hustle\IRMAI\workspace
python -m venv pm4py-poc-1
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg