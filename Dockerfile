FROM pandas/pandas:pip-all

RUN pip3 install catboost==1.1.1
RUN pip3 install lightgbm==3.3.5
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
WORKDIR /
