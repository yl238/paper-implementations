import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from config import config

if __name__ == "__main__":
    name = config["name"]
    save_name = config["local_path"]

    save_path = os.path.join("data", save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(save_path, name)
    if not os.path.exists(csv_path):
        zipurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(save_path)
