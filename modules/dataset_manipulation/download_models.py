import gzip
import shutil

import wget


def gunzip(file_path, output_path):
    with gzip.open(file_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()


def download_arpa():
    data_dir = "../../data/models"
    ARPA_URL = 'https://kaldi-asr.org/models/5/4gram_big.arpa.gz'
    f = wget.download(ARPA_URL, data_dir)
    gunzip(f, f.replace(".gz", ""))
