# Install required libraries
!pip install transformers torchvision torchaudio imageio[ffmpeg] --quiet

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import imageio
import torch
import os

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()

# Captioning function
def extract_and_caption(video_path, frame_num=5):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    try:
        frame = reader.get_data(frame_num)
    except IndexError:
        frame = reader.get_data(0)  # fallback to first frame
    image = Image.fromarray(frame).convert('RGB')

    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Upload videos
from google.colab import files
uploaded = files.upload()  # Upload of all 10 videos 

# Caption 
captions = []
for filename in uploaded.keys():
    print(f"Processing {filename}...")
    cap = extract_and_caption(filename)
    print(f">> {cap}")
    captions.append(f"{filename} - {cap}")

# Save to file
with open("generated_captions.txt", "w") as f:
    for line in captions:
        f.write(line + "\n")

print("\nAll captions saved to generated_captions.txt")
