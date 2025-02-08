import urllib.request
import pickle
import os

UNK_TOKEN = "<UNK>"
UNK_IDX = 0
MIN_ACCURANCES = 20
WORD_TO_IDX_FILE = "word_to_idx"

def get_text8_data():
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not os.path.exists('text8.zip'):
        urllib.request.urlretrieve(url, 'text8.zip')

    with open('text8.zip', 'rb') as f:
        from zipfile import ZipFile
        with ZipFile(f) as archive:
            data = archive.read('text8').decode('utf-8')
    return data


if not os.path.isfile(WORD_TO_IDX_FILE):
    text8_data = get_text8_data().split()
    statinfo = os.stat("./text8.zip")
    # expected: 31344016
    print(f"File size:{statinfo.st_size}")
    print(f"{len(text8_data)=}")
    print(f"{text8_data[:50]=}")

    word_to_idx: dict[str, int] = {UNK_TOKEN: UNK_IDX}
    word_counts = {}

    for word in text8_data:
        if word_counts.get(word, None) is None:
            word_counts[word] = 0
        word_counts[word] += 1

    current_index = UNK_IDX + 1
    for i in range(0, len(text8_data)):
        if word_counts[text8_data[i]] < MIN_ACCURANCES:
            # Replacing rare each word with UNK token
            text8_data[i] = UNK_TOKEN
        else:
            if text8_data[i] not in word_to_idx:
                word_to_idx[text8_data[i]] = current_index
                current_index += 1

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    pickle.dump(word_to_idx, open(WORD_TO_IDX_FILE, 'wb'))
    print(f"Saved word_to_idx in {WORD_TO_IDX_FILE}")
    print(f"{len(word_counts)=}, {len(word_to_idx)=}, {len(text8_data)=}")
else:
    print(f"{WORD_TO_IDX_FILE} file already exists. Skipping")

