import torch
import skimage.io as io
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from transformers import BlipProcessor, BlipModel


def main(clip_model_type: str):
    device = torch.device('cuda:0')

    # Load BLIP ViT-Large
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large").vision_model.to(device)

    out_path = f"../data/dataBLIP/{clip_model_type}_val.pkl"

    with open('../MSCOCO2017/annotations/captions_val2017.json', 'r') as f:
        data = json.load(f)

    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []

    for i in tqdm(range(len(data['annotations']))):
        d = data['annotations'][i]
        img_id = d["image_id"]
        filename = f"../MSCOCO2017/val2017/{int(img_id):012d}.jpg"
        
        # Load image and preprocess for BLIP
        image = io.imread(filename)
        image = processor(images=Image.fromarray(image), return_tensors="pt").pixel_values.to(device)

        # Extract ViT-Large features
        with torch.no_grad():
            prefix = model(image).last_hidden_state[:, 0, :].cpu()  # Extract CLS token representation

        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)

        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-Large", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'ViT-Large'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))