import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms


class cnn(nn.Module):
    def __init__(self,n_channel):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(n_channel,16,kernel_size=3,padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(16,32,kernel_size=3,padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32,64,kernel_size=3,padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,256),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64,19)
        )
            
        
    def forward(self,x):
        x= self.features(x)
        x=self.classifier(x)

        return x
model = cnn(1)
model.load_state_dict(torch.load("hand_written_digits_calculator/digit_model.pth", map_location=torch.device("cpu")))
model.eval()

st.set_page_config(layout="wide")
st.title("Handwritten Digit Calculator")

st.write("Draw the **first number**, an **operator**, and the **second number** below:")

# Layout 3 columns
col1, col2, col3 = st.columns(3)

canvas_settings = {
    "fill_color": "black",
    "stroke_width": 10,
    "stroke_color": "white",
    "background_color": "black",
    "height": 200,
    "width": 200,
    "drawing_mode": "freedraw"
}

# First digit canvas
with col1:
    st.subheader("Digit 1")
    canvas1 = st_canvas(key="canvas1", **canvas_settings)

# Operator canvas
with col2:
    st.subheader("Operator")
    canvas2 = st_canvas(key="canvas2", **canvas_settings)

# Second digit canvas
with col3:
    st.subheader("Digit 2")
    canvas3 = st_canvas(key="canvas3", **canvas_settings)

# Utility function to preprocess canvas image

def preprocess_canvas_image(canvas_result):
    img_array = canvas_result.image_data[:, :, :3].astype(np.uint8)  # Drop alpha channel
    pil_image = Image.fromarray(img_array)
    transform = transforms.Compose([
        transforms.Grayscale(),         
        transforms.Resize((56,56)),       
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(pil_image).unsqueeze(0)  # add batch dim: [1, 1, 28, 28]
    return tensor


# Predict and calculate

idx_to_class={
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '+', 11: '-', 12: '/', 13: '=', 14: '*', 15: '-', 16: '*', 17: '4', 18: 'z'
    }

if st.button("Calculate"):
    img1 = preprocess_canvas_image(canvas1)
    img2 = preprocess_canvas_image(canvas2)
    img3 = preprocess_canvas_image(canvas3)

    with torch.no_grad():
        pred1 = model(img1)
        pred2 = model(img3)
        pred_op = model(img2)

    if img1 is not None and img2 is not None and img3 is not None:
       digit1 = torch.argmax(pred1, dim=1).item()
       digit2 = torch.argmax(pred2, dim=1).item()
       op_idx  = torch.argmax(pred_op, dim=1).item()
       operator = idx_to_class[op_idx] 
       try:
            expression = f"{digit1} {operator} {digit2}"
            result = eval(expression)
            st.success(f"{expression} = {result}")
       except Exception as e:
            st.error(f"Error evaluating expression: {e}")
    else:
        st.warning("Please draw all three: digit, operator, and digit.")
