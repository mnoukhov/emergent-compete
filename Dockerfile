FROM pytorch/pytorch:1.3-cuda10.0-cudnn7-devel

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install -e .

WORKDIR /src
