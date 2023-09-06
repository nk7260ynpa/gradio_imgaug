import numpy as np
from PIL import Image
import torch
import torchvision as trv
import gradio as gr

def Preprocess(img):
    img = np.array(img)
    tensor = trv.transforms.Compose([trv.transforms.ToTensor()])(img)
    tensor = trv.transforms.Resize((256, 256), antialias=True)(tensor)
    return tensor

def imgaug():
    transforms = trv.transforms.Compose([trv.transforms.RandomRotation(30),
                                         trv.transforms.RandomGrayscale(p=0.5)])
    return transforms

def ProProcess(tensor):
    tensor = torch.clip(tensor, min=0.0, max=1.0)
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return img

def Process(img):
    tensor = Preprocess(img)
    transforms = imgaug()
    tensor = transforms(tensor)
    img = ProProcess(tensor)
    return img

demo = gr.Interface(fn=Process,
             inputs=gr.Image(type="pil"),
             outputs=["image"],
             examples=["parachute.jpeg", "bus.jpg", "sheep.jpg"])
    
demo.launch()  