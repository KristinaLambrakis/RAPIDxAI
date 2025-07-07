FROM ubuntu:20.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

ENV SERVICE_VERSION v5
ENV USERNAME rapidxai
ENV HOME /home/$USERNAME

RUN groupadd --gid 5000 $USERNAME \
    && useradd --home-dir $HOME --create-home --uid 5000 \
        --gid 5000 --shell /bin/sh --skel /dev/null $USERNAME

USER $USERNAME

ENV WORKSPACE $HOME/workspace
RUN mkdir -p $WORKSPACE

COPY requirements_$SERVICE_VERSION.txt $WORKSPACE/
RUN pip3 install -r $WORKSPACE/requirements_$SERVICE_VERSION.txt
RUN chmod -R 755 ~/.local

COPY path_utils.py $WORKSPACE/
COPY aiml $WORKSPACE/aiml
COPY service $WORKSPACE/service
COPY test $WORKSPACE/test
COPY config $WORKSPACE/config
COPY run_tester.sh $WORKSPACE
COPY troponin $WORKSPACE/troponin
COPY models $WORKSPACE/models

WORKDIR $WORKSPACE/service/$SERVICE_VERSION



