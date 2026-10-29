from os.path import exists
from pickle import dump, load
from glob import glob
from json import loads
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
import umap
import numpy as np
np.random.seed(42)

OUT_PATH = "FigurePieces/Figure2-"

BASE_DIR = "ihc/data/classification/"
SAMPLE_MATCH = "*.*/"
METADATA_FILE = "metadata.json"
EMBEDDING_FILE = "embedding.json"
CMAP = ListedColormap(['red', 'green', 'blue'])
CMAP_s = ListedColormap(['red', 'blue'])
S = 128

model_map = {
        "image": "CNN",
        "image_vt": "ViT",
        }

category_map = {
        "tn": "Triple Negative",
        "stage_basic": "Stage",
        "grade": "Grade",
        "hilostage": "Stage",
        "hilograde": "Grade",
        }

glob_str = BASE_DIR + SAMPLE_MATCH + EMBEDDING_FILE

model_embeddings = dict()
if not exists("embeddings.pkl"):
    for filename in glob(glob_str):
        print(filename)
        base = filename.replace(EMBEDDING_FILE, "")
        meta_filename = base + METADATA_FILE
        embd_data = dict()
        meta_data = dict()
        with open(filename, "r") as fhandle:
            embd_data = loads(fhandle.read())
        with open(meta_filename, "r") as fhandle:
            meta_data = loads(fhandle.read())
        for rkey in embd_data.keys():
            key = rkey.replace("Embedding-", "")
            if key not in model_embeddings.keys():
                model_embeddings[key] = list()
            category = key.split(".")[0].split("2")[1] + "_i"
            category_v = key.split(".")[0].split("2")[1] + "_i"
            if category_v == "hilostage_i":
                category_v = "stage_basic_i"
            elif category_v == "hilograde_i":
                category_v = "grade_i"
            if category_v not in meta_data.keys():
                continue
            embedding = embd_data[rkey]
            label = meta_data[category_v]
            if "hilo" in category:
                print(label, end=': ')
                label = int(label > 1)
                print(label)
            if "hilostage" in category:
                print(rkey, filename, category, category_v, embedding, label)
            model_embeddings[key].append((embedding, label))
    with open("embeddings.pkl", "wb+") as fhandle:
        dump(model_embeddings, fhandle)
else:
    with open("embeddings.pkl", "rb") as fhandle:
        model_embeddings = load(fhandle)

plt.rc('axes', linewidth=4)
for model in model_embeddings.keys():
    data = model_embeddings[model]
    if len(data) == 0:
        continue
    print(model)
    embeddings, labels = zip(*data)
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    assert embeddings.shape[0] == labels.shape[0]
    fitter = umap.UMAP()
    umap_embedding = fitter.fit_transform(embeddings)
    name_raw = model.split(".")[0]
    mid = model.split(".")[1]
    mname = name_raw.split("-")[0]
    category = name_raw.split("2")[1]
    _nbins = 2
    _CMAP = CMAP_s
    if category in ["stage_basic", "grade"]:
        _nbins = 3
        _CMAP = CMAP
        continue
    lmodel = model_map[mname]
    try:
        lcategory = category_map[category]
    except:
        continue
    name = name_raw.split("-")[0] + "-" + name_raw.split("2")[1] + "-" + mid

    plt.clf()
    plt.scatter(umap_embedding[:,0], umap_embedding[:,1], c=labels, s=S, cmap=_CMAP)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.tight_layout()
    plt.savefig(OUT_PATH + name + ".svg")
    plt.savefig(OUT_PATH + name + ".pdf")
