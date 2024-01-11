import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch

# Load the pre-trained model and set it up
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

# Streamlit app
def main():
    st.title("Text to Image Generation App")

    # User input for the prompt
    prompt = st.text_area("Enter a prompt:", "A cinematic shot of a baby raccoon wearing an intricate Italian priest robe.")

    # Generate image on button click
    if st.button("Generate Image"):
        # Perform inference with the provided prompt
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
