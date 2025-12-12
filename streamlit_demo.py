# Load Models
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import seaborn as sns
from customcnn import CustomCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

torch.serialization.add_safe_globals([CustomCNN])


torch.serialization.add_safe_globals([transforms.Compose])

# load exported models
res_net = torch.load("./ResNet_model.pth", map_location=torch.device('cpu'),  weights_only=False)
viggl_net = torch.load("./VGG_model.pth", map_location=torch.device('cpu'),  weights_only=False)
mobil_net = torch.load("./MobileNetV2_model.pth", map_location=torch.device('cpu'),  weights_only=False)
custom_net = torch.load("./CustomCNN_model.pth", map_location=torch.device('cpu'),  weights_only=False)
transform = torch.load("./transform.pth", map_location=torch.device('cpu'),  weights_only=False ) # must be false to load full objects


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sreamlit UI
st.title("Fatigue Detection")
st.write("Upload an image to detect fatigue")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image",width=200)

    # Process image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Select Model
    model_name = st.selectbox("Select a model", ["ResNet50", "VGG16", "MobileNetV2", "CustomCNN"])
    if(model_name == "ResNet50"):
        model = res_net
    elif(model_name == "VGG16"):
        model = viggl_net
    elif(model_name == "MobileNetV2"):
        model = mobil_net
    elif(model_name == "CustomCNN"):
        model = custom_net
    else:
        model_name = "ResNet50"
        model = res_net
        

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs.max(1)[0]
        if predicted.item() == 0:
            st.write("Fatigue")
        else:
            st.write("Non-Fatigue")
        st.write(f"Confidence: {confidence.item() * 100:.2f}%")


# Introspection Using Saliency Maps
if uploaded_file is not None:
    st.write("Saliency Map Introspection")

    # Make sure input requires grad
    image_tensor.requires_grad = True

    # Forward pass
    outputs = model(image_tensor)
    score, predicted = torch.max(outputs, 1)

    # Backward pass w.r.t predicted class
    model.zero_grad()
    score.backward()

    # Get gradients w.r.t input
    saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=1)  # [1,H,W]
    saliency = saliency.squeeze().cpu().numpy()
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

    # Convert input tensor to displayable image
    img_display = image_tensor.cpu().squeeze(0).permute(1,2,0).detach().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = np.clip(std * img_display + mean, 0,1).astype(np.float32)

    # Overlay saliency map
    plt.figure(figsize=(6,6))
    plt.imshow(img_display)
    plt.imshow(saliency, cmap='hot', alpha=0.5)
    plt.axis('off')

    # Show in Streamlit
    st.pyplot(plt.gcf())

  