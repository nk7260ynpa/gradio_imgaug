import numpy as np
from PIL import Image
import torch
import torchvision as trv
import gradio as gr

def Preprocess(img):
    #img = np.array(img)
    return img

def Tensor_To_img(tensor):
    tensor = torch.clip(tensor, min=0.0, max=1.0)
    img = tensor.permute(1, 2, 0).numpy()
    return img


def TransFunction(img, RotateAngle, Horizon_Flip, Vertical_Flip, 
                  Equalize, Brightness, Contrast, Saturation, Sharpness):
    img = Preprocess(img)
    tensor = trv.transforms.functional.pil_to_tensor(img)
    if Equalize:
        tensor = trv.transforms.functional.equalize(tensor)
    tensor = tensor / 255.
    tensor = trv.transforms.functional.rotate(tensor, RotateAngle)
    if Horizon_Flip:
        tensor = trv.transforms.functional.hflip(tensor)
    if Vertical_Flip:
        tensor = trv.transforms.functional.vflip(tensor)
    
    tensor = trv.transforms.functional.adjust_brightness(tensor, Brightness)
    tensor = trv.transforms.functional.adjust_contrast(tensor, Contrast)
    tensor = trv.transforms.functional.adjust_saturation(tensor, Saturation)
    tensor = trv.transforms.functional.adjust_sharpness(tensor, Sharpness)
    
    
    img = Tensor_To_img(tensor)
    return img
    

demo = gr.Interface(fn=TransFunction,
                    inputs=[gr.Image(type="pil"), 
                            gr.Slider(-180, 180, value=0), 
                            gr.Checkbox(value=False),
                            gr.Checkbox(value=False),
                            gr.Checkbox(value=False),
                            gr.Slider(0, 2, value=1),
                            gr.Slider(0, 2, value=1),
                            gr.Slider(0, 2, value=1),
                            gr.Slider(0, 2, value=1)],
                    outputs=["image"],
                    examples=[["parachute.jpeg"],
                              ["bus.jpg"], 
                              ["sheep.jpg"]])
    
demo.launch()  