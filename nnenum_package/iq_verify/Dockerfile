FROM docker.io/library/python:3.8-slim@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f

WORKDIR /home

RUN python -m pip install \
    swiglpk==5.0.8 \
    gurobipy==11.0.2 \
    Mosek==10.1.12 \
    scipy==1.10.1 \
    matplotlib==3.6.2 \
    cvxpy==1.3.2 \
    cvxopt==1.3.2 \
    onnxruntime==1.13.1 \
    termcolor==2.1.1 \
    numpy==1.23.5

RUN apt-get update && apt-get install -y \
    vim \
    less \
    nano \
    emacs

COPY ./README.md /home/README.md

COPY ./setup.py /home/setup.py

COPY ./examples /home/examples

COPY ./iq_verify /home/iq_verify

COPY ./rep /home/rep

COPY ./mosek.lic /root/mosek/mosek.lic

RUN python -m pip install .

RUN echo "benchmark,instance,result,time,comment" >> /home/results.csv

RUN mkdir images/

CMD /bin/bash

## sudo docker build . -t wehbedoug/iqverify:20251013-07

## sudo docker push wehbedoug/iqverify:20251013-07

## sudo docker run -it --rm -v ./images:/home/images wehbedoug/iqverify:20251013-07
