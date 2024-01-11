import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

# Load the pre-trained model and set it up
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo" , torch_dtype=torch.float16)
pipe.to("cpu")

# Streamlit app
def main():
    st.title("🌈Text to Image Generation🎨")

    # User input for the prompt
    prompt = st.text_area("✏️ Enter a prompt:", "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe.")

    # Generate image on button click
    if st.button("Generate Image 🚀"):
        # Show a spinner while generating the image
        with st.spinner("Generating Image..."):
            # Perform inference with the provided prompt
            image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        # Display the generated image with style
        st.image(image, caption="Generated Image 🖼️", use_column_width=True, output_format="JPEG")

        # Add a download button for the generated image
        download_btn = st.button("Download Image 💾")
        if download_btn:
            # Convert the PIL Image to bytes
            image_bytes = Image.fromarray(image.numpy())
            
            # Offer the image for download
            st.download_button(
                label="Download Image",
                key="download_image",
                on_click=lambda: save_image(image_bytes),
                help="Download the generated image.",
            )

def save_image(image_bytes):
    # Save the image to the user's local machine
    with open("generated_image.jpg", "wb") as f:
        f.write(image_bytes.tobytes())

if __name__ == "__main__":
    main()
