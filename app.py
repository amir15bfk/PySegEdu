import streamlit as st
import pdf2image
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms
from models import fcbformer,fcn2,unet,doubleunet
import os


# Load pretrained UNet 
model = fcbformer.FCBFormer()
option = st.selectbox(
    "select the model ?",
    ["FCN","Unet","DoubleUnet","FCBFormer"])
if option == "FCBFormer":
    model = fcbformer.FCBFormer()
    size = (352,352)
elif option == "Unet":
    model = unet.Unet()
    size = (128,128)
elif option == "DoubleUnet":
    model = doubleunet.build_doubleunet()
    size = (352,352)
elif option == "FCN":
    model = fcn2.FCN8s()
    size = (352,352)
    
models = os.listdir("./Trained_models")
print(models)
selected = st.selectbox(
    "select the source ?",
    [i for i in models if i.startswith(option)])

model.load_state_dict(torch.load("Trained_models/"+selected)["model_state_dict"])
model.eval()
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(size),
                            transforms.Normalize(mean, std)])
transform2 = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(size)])
st.write("""
# Segmentation Model Demo
Upload an image and select a model to perform segmentation.
""")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Preprocess  
    input_tensor = transform(image).float()
    image_tensor = transform2(image).float()

    # Segment image
    with torch.no_grad():
        output_tensor = model(input_tensor.unsqueeze(0))
        analog_image = torch.sigmoid(output_tensor)
        pred_masks = (analog_image > 0.5).float()
    
    pred_mask = pred_masks[0][0]
    # Convert NumPy array to PIL image
    original_image = image
    img_p = image_tensor
    for i in range(img_p.shape[1]):
        for j in range(img_p.shape[2]):
            if pred_mask[i, j]:
                img_p[1, i, j] = pred_mask[i, j]*0.9
                img_p[0, i, j] = img_p[0, i, j] * 0.9
                img_p[2, i, j] = img_p[2, i, j] * 0.9
    segmented_image = transforms.ToPILImage()(pred_masks.squeeze())
    analog_image = transforms.ToPILImage()(analog_image.squeeze())
    img_p = transforms.ToPILImage()(img_p.squeeze())
    image_tensor = transforms.ToPILImage()(image_tensor.squeeze())


    st.image([img_p], width=500)