import streamlit as st
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
import os

st.title("DreamBooth Image-to-Image App 💭🌈")

# st.markdown("This app allows you to transform images using the DreamBooth model. ✨")

# Load the DreamBooth model
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo")
# pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cpu")
# pipe.to("cuda")

# Define the function to generate images
def generate_image(init_image, prompt, num_inference_steps, strength, guidance_scale):
    image = pipe(prompt, image=init_image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale).images[0]
    return image

# Save the uploaded image to a temporary directory
def save_uploaded_image(uploaded_image):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.join(temp_dir, "uploaded_image.png")
    with open(file_name, "wb") as f:
        f.write(uploaded_image.getbuffer())
    return file_name

# Allow the user to upload an image
st.sidebar.header("Options")
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

# Allow the user to input a prompt
st.sidebar.header("Prompt")
prompt = st.text_input("Enter a prompt")
# prompt = st.sidebar.text_input("Enter a prompt")

# Set the default values for the other parameters
num_inference_steps = st.sidebar.slider("Number of Inference Steps", 1, 10, 2)
strength = st.sidebar.slider("Strength", 0.0, 1.0, 0.5)
guidance_scale = st.sidebar.slider("Guidance Scale", 0.0, 1.0, 0.0)

# Set default values
if not st.sidebar.button("Reset to Default"):
    num_inference_steps = 2
    strength = 0.5
    guidance_scale = 0.0

# Generate the image
if uploaded_image is not None and prompt:
    # Save the uploaded image to a temporary directory
    image_path = save_uploaded_image(uploaded_image)

    # Load the saved image
    init_image = load_image(image_path)

    # Resize the image to the desired size
    re_image = init_image.resize((512, 512))

    # Generate the image
    image = generate_image(re_image, prompt, num_inference_steps, strength, guidance_scale)

    # Resize the generated image to the original size
    image = image.resize(init_image.size)

    # Display the generated image
    st.image(image, caption="Generated image", use_column_width=True)

        
# # Add a slider to control the strength of the transformation
# st.header("Strength:")
# strength = st.slider("Strength", 0.0, 1.0, value=0.5, step=0.1)

# # Add a slider to control the number of inference steps
# st.header("Number of inference steps:")
# num_inference_steps = st.slider("Number of inference steps", 1, 5, value=2, step=1)

# # Add a slider to control the guidance scale
# st.header("Guidance scale:")
# guidance_scale = st.slider("Guidance scale", 0.0, 1.0, value=0.0, step=0.1)

# # Generate the image
# if uploaded_image is not None and prompt is not "":
#     init_image = load_image(uploaded_image).resize((512, 512))
#     image = generate_image(init_image, prompt, num_inference_steps, strength, guidance_scale)

#     # Display the generated image
#     st.image(image, caption="Generated image")
