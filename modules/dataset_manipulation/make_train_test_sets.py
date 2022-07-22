from sklearn.model_selection import train_test_split

tags = ('train', 'development', 'test')
proportions = {'train': 60, 'development': 20, 'test': 20}

FOLDER_MODULE = "modules/diarization_model/"
FOLDER_CONFIGS = "info/configs/diarization/"
FILE_LIST = "lists/dataset_diarization_reference.lst"
FILE_RTTM = "rttms/dataset_diarization_reference.rttm"
FILE_UEM = "uems/dataset_diarization_reference.uem"
with open(FOLDER_CONFIGS + FILE_LIST) as f:
    lines = f.readlines()
with open(FOLDER_CONFIGS + FILE_RTTM) as f:
    rttms = f.readlines()
with open(FOLDER_CONFIGS + FILE_UEM) as f:
    uems = f.readlines()

lines_train, lines_test = train_test_split(lines, test_size=0.4, shuffle=False)
lines_development, lines_test = train_test_split(lines_test, test_size=0.5, shuffle=False)
subsets = {"train": lines_train, "development": lines_development, "test": lines_test}

for subset, subset_lines in subsets.items():
    with open(FOLDER_CONFIGS + "lists/dataset_diarization_" + subset + ".lst", "w") as f:
        f.write("".join(subset_lines))
    subset_rttms = [k for k in rttms if k.split(" ")[1] in [elem.strip() for elem in subset_lines]]
    with open(FOLDER_CONFIGS + "rttms/dataset_diarization_" + subset + ".rttm", "w") as f:
        f.write("".join(subset_rttms))
    subset_uems = [k for k in uems if k.split(" ")[0] in [elem.strip() for elem in subset_lines]]
    with open(FOLDER_CONFIGS + "uems/dataset_diarization_" + subset + ".uem", "w") as f:
        f.write("".join(subset_uems))
