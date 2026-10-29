import os
from json import loads
import pickle
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

print(torch.cuda.is_available())

N_BATCH = 100
N_TIME = 10

def random_crop(im, w=256, h=256):
    iw,ih = im.size
    x = np.random.randint(iw-w)
    y = np.random.randint(ih-h)
    return im.crop((x,y,x+w,y+h))

model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

try:
    img_name = os.environ["DATA_NAME"]
except:
    img_name = "TMA00.A01"
img_path = f"ihc/data/classification/{img_name}/image.png"
jsn_path = f"ihc/data/classification/{img_name}/metadata.json"
image = Image.open(img_path).convert("RGB")

with open(jsn_path, "r") as fhandle:
    data = fhandle.read()
jdata = loads(data)
try:
    assignment = jdata["assignment"]
except:
    assignment = "unknown"

embeddings = []
for _ in range(N_TIME):
    simgs = [random_crop(image, w=224, h=224) for _ in range(N_BATCH)]
    sinputs = [transforms(s).unsqueeze(0) for s in simgs]
    sinputs = torch.concat(sinputs)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(sinputs)
    class_token = output[:,0]
    patch_tokens = output[:, 5:]
    embed = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    embed = embed.detach().cpu().numpy()
    embeddings.append(embed)
    #print(embed.shape)
    #img = random_crop(image, w=224, h=224)
    #print(img.mode)
    #img = transforms(img).unsqueeze(0)
    #print(img.shape)
    #with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    #    output = model(img)
    #class_token = output[:,0]
    #patch_tokens = output[:, 5:]
    #embed = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    #print(embed.shape)
embeddings = np.concat(embeddings)
print(embeddings.shape)
with open(f"{img_name}.pickle", "wb") as fhandle:
    ddict = {
        "embed": embeddings,
        "assignment": assignment
        }
    pickle.dump(ddict, fhandle)
    print(ddict["assignment"])
