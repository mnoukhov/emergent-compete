FROM pytorch/pytorch:latest

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD notebook_requirements.txt notebook_requirements.txt
RUN pip install -r notebook_requirements.txt
