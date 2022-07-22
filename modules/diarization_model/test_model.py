from pyannote.audio import Model, Inference
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.tasks import Segmentation
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
from pyannote.audio.utils.signal import binarize
from pyannote.database import get_protocol

from modules.diarization_model.visualization import make_inference_plot, make_annotated_inference_plot


def make_inference(model, sound_file):
    inference = Inference(model, device=device)
    hypothesis = binarize(inference(sound_file))
    print(hypothesis)
    make_inference_plot(inference, sound_file)


def test(model, protocol, subset="test"):
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())
    inference = Inference(model, device=device)
    make_annotated_inference_plot(inference, files[0])
    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        metric(reference, hypothesis, uem=uem)
    return abs(metric)


(device,) = get_devices(needs=1)
diar_protocol = get_protocol('MyDatabase.SpeakerDiarization.MyProtocol')
pretrained_model = Model.from_pretrained("pyannote/segmentation")
diarization_model = pretrained_model.load_from_checkpoint(checkpoint_path="data/models/diarization_model.ckpt")
seg_task = Segmentation(diar_protocol, max_num_speakers=2, num_workers=0)
# der_pretrained = test(model=pretrained_model, protocol=diar_protocol, subset="test")
# print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
def_trained = test(model=diarization_model, protocol=diar_protocol, subset="test")
print(f"Local DER (trained) = {def_trained * 100:.1f}%")
