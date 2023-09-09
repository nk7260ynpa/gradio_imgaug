import numpy as np
from PIL import Image
import torch
import torchvision as trv
import gradio as gr

def Preprocess(img):
    img = np.array(img)
    tensor = trv.transforms.Compose([trv.transforms.ToTensor()])(img)
    return tensor

def Tensor_To_img(tensor):
    tensor = torch.clip(tensor, min=0.0, max=1.0)
    img = tensor.permute(1, 2, 0).numpy()
    return img


def TransFunction(img, RotateAngle):
    tensor = Preprocess(img)
    tensor = trv.transforms.functional.rotate(tensor, RotateAngle)
    img = Tensor_To_img(tensor)
    return img
    

demo = gr.Interface(fn=TransFunction,
                    inputs=[gr.Image(type="pil"), gr.Slider(-180, 180, value=0)],
                    outputs=["image"],
                    examples=[["parachute.jpeg", None],["bus.jpg", None], ["sheep.jpg", None]])
    
demo.launch()  