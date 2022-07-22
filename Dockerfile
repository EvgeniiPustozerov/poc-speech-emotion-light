FROM python:3.10-slim
WORKDIR /app
COPY . .

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN apt-get -y install gcc mono-mcs
RUN apt-get -y install g++

# Install every package one after another to track time
RUN python -m pip install --upgrade pip
# RUN pip install -r requirements.txt
RUN pip install aiogram>=2.20, librosa>=0.9.1, matplotlib>=3.5.1, nemo>=4.1.1, nemo_toolkit>=1.9.0, numpy==1.22.0
RUN pip install omegaconf>=2.2.2, pandas>=1.3.5, pydub>=0.25.1, scikit_learn>=1.1.1, scipy>=1.8.1, wget>=3.2
RUN pip install xgboost>=1.5.2

# Packages not included into requirements
RUN pip install soundfile
RUN pip install hydra-core
RUN pip install pytorch_lightning
RUN pip install braceexpand
RUN pip install webdataset
RUN pip install inflect
RUN pip install transformers
RUN pip install sentencepiece
RUN pip install Cython
RUN pip install youtokentome
RUN pip install pyannote.audio
RUN pip install IPython
RUN pip install editdistance
RUN pip install h5py
RUN pip install sacremoses
RUN pip install sacrebleu
RUN pip install einops
RUN pip install jieba
RUN pip install opencc
RUN pip install pangu
RUN pip install ipadic
RUN pip install mecab-python3
RUN pip install praat-parselmouth
RUN pip install pyctcdecode
RUN pip install psutil
RUN pip install ijson
RUN pip install spacy
RUN pip install torchaudio
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

CMD ["python", "./TelegramBot.py"]
# Nect commands are: docker build -t pustozerov/poc-speech-recognition:1.0
