FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN apt install graphviz -y

RUN mkdir -p /code/temp

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN <<EOF
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
EOF

ENV AZURE_FILE_PATH=/mnt/azure/graphviz-12.2.0.zip

COPY . .

CMD ["streamlit","run","IRMAI.py","--server.port","8501"]


