import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

# Load the pre-trained model and set it up
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo" )
# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo" , torch_dtype=torch.float16)
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

        # Add a button to download the image

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        btn = st.download_button(label="📥 Download Image",data=buf,file_name="generated_image.jpg",mime="image/jpeg")

if __name__ == "__main__":
    main()
