import numpy as np
from PIL import Image
import torch
import torchvision as trv
import gradio as gr

def Preprocess(img):
    #img = np.array(img)
    return img

def Tensor_To_img(img):
    img = torch.clip(img, min=0.0, max=1.0)
    img = img.permute(1, 2, 0).numpy()
    return img


def AUGFunction(img, Equalize, Grayscale, VerticalFlip, HorizontalFlip, Autocontrast, Invert, brightness,
                contrast, saturation,
                Erasing, degrees, Crop):
    AUG = trv.transforms.Compose([trv.transforms.PILToTensor(),
                                  trv.transforms.Resize((256, 256), antialias=True),
                                  trv.transforms.RandomEqualize(p=Equalize),
                                  trv.transforms.ConvertImageDtype(dtype=torch.float32),
                                  trv.transforms.RandomGrayscale(p=Grayscale),
                                  trv.transforms.RandomAutocontrast(p=Autocontrast),
                                  trv.transforms.RandomVerticalFlip(p=VerticalFlip),
                                  trv.transforms.RandomHorizontalFlip(p=HorizontalFlip),
                                  trv.transforms.RandomInvert(p=Invert),
                                  trv.transforms.ColorJitter(brightness=brightness,
                                                             contrast=contrast,
                                                             saturation=saturation),
                                  trv.transforms.RandomErasing(p=Erasing),
                                  trv.transforms.RandomAffine(degrees=degrees),
                                  trv.transforms.RandomCrop((Crop, Crop))])
    img = Preprocess(img)
    img = AUG(img)
    img = Tensor_To_img(img)
    return img
    

demo = gr.Interface(fn=AUGFunction,
                    inputs=[gr.Image(value="sheep.jpg", type="pil"),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 1, value=0),
                            gr.Slider(0, 180, value=0),
                            gr.Slider(0, 256, value=256)],
                    outputs=["image"],
                    examples=[["sheep.jpg"],
                              ["parachute.jpeg"],
                              ["bus.jpg"]])
    
demo.launch()  