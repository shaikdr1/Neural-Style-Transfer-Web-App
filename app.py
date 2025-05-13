import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
import io
from model import compute_loss, create_model, preprocess_image, generate_noise_image
import cv2

# Load model
model = create_model()

# Function to load and display the image
def load_image(uploaded_image):
    img = Image.open(uploaded_image)
    return np.array(img)


# Function to generate the stylized image
def generate_stylized_image(content_image, style_image):
    # Preprocess images for the model
    content_image = preprocess_image(content_image)
    style_image = preprocess_image(style_image)

    # Initialize the generated image as the content image
    generated_image = tf.Variable(generate_noise_image(content_image))

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=3.0)

    # Perform gradient descent to minimize the total loss
    for i in range(400):
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            loss = compute_loss(model, content_image, style_image, generated_image)

        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])

        if i % 10 == 0:
            st.write(f"Iteration {i}: Loss = {loss.numpy()}")

    c = generated_image.numpy()
    c[..., 0] += 103.939  # B
    c[..., 1] += 116.779  # G
    c[..., 2] += 123.68   # R

    c= c[..., ::-1]

    # Clip to [0, 255] and convert to uint8 for image display
    c = np.clip(c, 0, 255).astype('uint8') 
    c = c.squeeze(0)

    # Post-process the generated image to be in the valid range
    #generated_image = generated_image.numpy().squeeze()

    # Normalize the pixel values to be in the range [0, 255] for display
    #generated_image = np.clip(generated_image, 0, 255).astype('uint8')

    return c

# Streamlit UI
def main():
    st.title("Neural Style Transfer Web App")
    st.write("Upload a content image and a style image to apply the neural style transfer.")

    content_image = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_image = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_image is not None and style_image is not None:
        # Load the images
        content_img = load_image(content_image)
        style_img = load_image(style_image)

        st.image(content_img, caption="Content Image", use_column_width=True)
        st.image(style_img, caption="Style Image", use_column_width=True)

        # Generate the stylized image
        st.write("Generating stylized image...")
        generated_img = generate_stylized_image(content_img, style_img)
        
        # Display the generated image
        st.image(generated_img, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
