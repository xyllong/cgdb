from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# image2 = image
# image = Image.open("3.png")
# image2 = Image.open("1.png")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to("cuda")

@torch.no_grad()
def dino_style_similarity(image1, image2):
    inputs = processor(images=image1, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
    last_hidden_states = outputs.pooler_output

    inputs2 = processor(images=image2, return_tensors="pt")
    inputs2 = inputs2.to("cuda")
    outputs2 = model(**inputs2)
    last_hidden_states2 = outputs2.pooler_output

    return torch.nn.functional.cosine_similarity(last_hidden_states, last_hidden_states2).item()

@torch.no_grad()
def get_dino_features(images):
    inputs = processor(images=images, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
    last_hidden_states = outputs.pooler_output
    return last_hidden_states

@torch.no_grad()
def dino_style_similarities(image1, image2):
    last_hidden_states1 = get_dino_features(image1)
    last_hidden_states2 = get_dino_features(image2)

    res = []
    for hidden_states1 in last_hidden_states1:
        resi = []
        for hidden_states2 in last_hidden_states2:
            resi.append(torch.nn.functional.cosine_similarity(hidden_states1[None, ...], hidden_states2[None, ...]).item())
        res.append(resi)
        
    return res


if __name__ == "__main__":
    image = Image.open("/home/ldy/ldy/jtcomp/32323.png")
    image2 = Image.open("/home/ldy/ldy/jtcomp/reff.png")
    print(dino_style_similarity(image, image2))
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.pooler_output

# inputs2 = processor(images=image2, return_tensors="pt")
# outputs2 = model(**inputs2)
# last_hidden_states2 = outputs2.pooler_output

# def calc_cosine_similarity(a, b):
#     return a@(b.T) / (a.norm()*b.norm())
# print(calc_cosine_similarity(last_hidden_states, last_hidden_states2))