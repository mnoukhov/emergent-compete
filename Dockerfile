FROM pytorch/pytorch:latest

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
