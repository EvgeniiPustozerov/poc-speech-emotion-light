from copy import deepcopy
import pytorch_lightning as pl
import torchaudio
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.tasks import Segmentation
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
from pyannote.audio.utils.signal import binarize
from pyannote.database import get_protocol
from pytorch_lightning import loggers as pl_loggers
import logging

torchaudio.set_audio_backend("soundfile")
(device,) = get_devices(needs=1)


def test(model, protocol, subset="test"):
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())
    inference = Inference(model, device=device)
    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)
    return abs(metric)


def main():
    diar_protocol = get_protocol('MyDatabase.SpeakerDiarization.MyProtocol')
    pretrained_model = Model.from_pretrained("pyannote/segmentation")
    seg_task = Segmentation(diar_protocol, max_num_speakers=2, num_workers=0, vad_loss='bce')
    der_pretrained = test(model=pretrained_model, protocol=diar_protocol, subset="train")
    print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
    model_diarization = deepcopy(pretrained_model)
    model_diarization.task = seg_task

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="data/models/diarization_checkpoints/")
    trainer = pl.Trainer(max_epochs=10, logger=tb_logger)
    trainer.fit(model_diarization)
    results = test(model=model_diarization, protocol=diar_protocol, subset="train")
    print(f"Local DER (train) = {results * 100:.1f}%")
    results = test(model=model_diarization, protocol=diar_protocol, subset="development")
    print(f"Local DER (development) = {results * 100:.1f}%")
    results = test(model=model_diarization, protocol=diar_protocol, subset="test")
    print(f"Local DER (test) = {results * 100:.1f}%")


if __name__ == '__main__':
    main()
