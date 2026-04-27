import os, json
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@torch.no_grad()
def embed_batch(model, processor, paths):
    images = []
    for p in paths:
        # Try both .jpg and .png if file doesn't exist
        if not os.path.exists(p):
            alt_path = p.replace('.jpg', '.png') if p.endswith('.jpg') else p.replace('.png', '.jpg')
            if os.path.exists(alt_path):
                p = alt_path
        images.append(Image.open(p).convert("RGB"))
    
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    vision_outputs = model.vision_model(**inputs)
    image_embeds = vision_outputs.pooler_output
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds.cpu().numpy()



def main():
    print("Loading items...")
    with open("data/items.json", "r") as f:
        items = json.load(f)
    
    print("Loading CLIP model...")
    model, processor = load_model()
    
    print(f"Embedding {len(items)} images...")
    all_vecs = []
    all_ids = []
    batch_size = 16
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_ids = [it["item_id"] for it in batch]
        batch_paths = [it["path"] for it in batch]
        
        v = embed_batch(model, processor, batch_paths)
        all_vecs.append(v)
        all_ids.extend(batch_ids)
        print(f"  {i+len(batch)}/{len(items)}")
    
    all_vecs = np.vstack(all_vecs).astype("float32")
    
    print("Building FAISS index...")
    d = all_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_vecs)
    
    os.makedirs("artifacts", exist_ok=True)
    faiss.write_index(index, "artifacts/faiss.index")
    with open("artifacts/item_ids.json", "w") as f:
        json.dump(all_ids, f)
    np.save("artifacts/item_vectors.npy", all_vecs)
    
    print(f"✅ Done! Created index with {len(all_ids)} items, dim={d}")

if __name__ == "__main__":
    main()
