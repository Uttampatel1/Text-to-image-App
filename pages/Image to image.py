import streamlit as st
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

st.title("DreamBooth Image-to-Image App")

st.markdown("This app allows you to transform images using the DreamBooth model.")

# Load the DreamBooth model
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("cpu")

# Define the function to generate images
def generate_image(init_image, prompt, num_inference_steps, strength, guidance_scale):
    image = pipe(prompt, image=init_image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale).images[0]
    return image

# Allow the user to upload an image
st.header("Select an image to transform:")
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Allow the user to input a prompt
st.header("Enter a prompt:")
prompt = st.text_input("Prompt")

# Set the default values for the other parameters
num_inference_steps = 2
strength = 0.5
guidance_scale = 0.0

# Generate the image
if uploaded_image is not None and prompt is not "":
    init_image = load_image(uploaded_image).resize((512, 512))
    image = generate_image(init_image, prompt, num_inference_steps, strength, guidance_scale)

    # Display the generated image
    st.image(image, caption="Generated image")
    
# Add a slider to control the strength of the transformation
st.header("Strength:")
strength = st.slider("Strength", 0.0, 1.0, value=0.5, step=0.1)

# Add a slider to control the number of inference steps
st.header("Number of inference steps:")
num_inference_steps = st.slider("Number of inference steps", 1, 5, value=2, step=1)

# Add a slider to control the guidance scale
st.header("Guidance scale:")
guidance_scale = st.slider("Guidance scale", 0.0, 1.0, value=0.0, step=0.1)

# Generate the image
if uploaded_image is not None and prompt is not "":
    init_image = load_image(uploaded_image).resize((512, 512))
    image = generate_image(init_image, prompt, num_inference_steps, strength, guidance_scale)

    # Display the generated image
    st.image(image, caption="Generated image")
