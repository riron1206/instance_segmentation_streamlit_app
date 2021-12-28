FROM sinpcw/pytorch:1.9.0

USER root

RUN pip install -U pip

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pyyaml
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

RUN pip install streamlit

WORKDIR ./