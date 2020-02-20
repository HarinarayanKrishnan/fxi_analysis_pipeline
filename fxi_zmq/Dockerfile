FROM continuumio/miniconda3
COPY document_to_zmq_replay.py requirements.txt /app/

RUN conda install -y numcodecs python-snappy && pip install -r /app/requirements.txt

CMD python /app/document_to_zmq_replay.py

