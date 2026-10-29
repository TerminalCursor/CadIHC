import os
from json import loads
import pickle
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

print(torch.cuda.is_available())

N_BATCH = 100
N_TIME = 10

def random_crop(im, w=256, h=256):
    iw,ih = im.size
    x = np.random.randint(iw-w)
    y = np.random.randint(ih-h)
    return im.crop((x,y,x+w,y+h))

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

img_name = "TMA00.A01"
try:
    img_name = os.environ["DATA_NAME"]
except:
    img_name = ""
img_path = f"ihc/data/classification/{img_name}/image.png"
jsn_path = f"ihc/data/classification/{img_name}/metadata.json"
#img_name = img_path.split("/")[-2]
img = Image.open(img_path).convert("RGB")

with open(jsn_path, "r") as fhandle:
    data = fhandle.read()
jdata = loads(data)
try:
    assignment = jdata["assignment"]
except:
    assignment = "unknown"

embeddings = []
tile_encoder.eval()
with torch.no_grad():
    for _ in range(N_TIME):
        simgs = [random_crop(img) for _ in range(N_BATCH)]
        sinputs = [transform(simg).unsqueeze(0) for simg in simgs]
        sinputs = torch.concat(sinputs)
        print(sinputs.shape)
        output = tile_encoder(sinputs).squeeze()

        #nimg = random_crop(img)
        #sample_input = transform(nimg).unsqueeze(0)
        #output = tile_encoder(sample_input).squeeze()

        embedding = output.detach().cpu().numpy()
        embeddings.append(embedding)

embeddings = np.stack(embeddings, axis=0)
with open(f"{img_name}.pickle", "wb") as fhandle:
    ddict = {
        "embed": embeddings,
        "assignment": assignment
        }
    pickle.dump(ddict, fhandle)
    print(ddict["assignment"])
