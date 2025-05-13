
# You can leave the rest of your code (content_loss, style_loss, total_variation_loss) as it i
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import backend as K
import numpy as np




# Load VGG19 model pre-trained on ImageNet data
def load_vgg19_model():
   model = applications.VGG19(weights='imagenet', include_top=False)
   model.trainable = False
   return model


# Preprocessing for images
def preprocess_image(image):
   image = tf.image.resize(image, (224, 224))  # Resize image to 224x224 (VGG input size)
   image = image[None, ...]  # Add batch dimension
   image = applications.vgg19.preprocess_input(image)
   return image


# Calculate the content loss
def content_loss(content, generated):
   m, n_H, n_W, n_C = content.get_shape().as_list()

   unrolled_content = tf.reshape(content, shape=[m, n_H * n_W, n_C]) # Or tf.reshape(a_C, shape=[m, -1 , n_C])
   unrolled_generated = tf.reshape(generated, shape=[m, n_H * n_W, n_C]) # Or tf.reshape(a_G, shape=[m, -1 , n_C])

   J_content = (1 / (4.0 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(unrolled_content - unrolled_generated))

   return J_content


# Calculate the style loss
def gram_matrix(A):
  GA = tf.matmul(A, tf.transpose(A))
  return GA


def style_loss(style, generated):
 
    m, n_H, n_W, n_C = style.get_shape().as_list()
    
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    olk = tf.reshape(style, [n_H * n_W, n_C])
    style_transposed = tf.transpose(olk)
    klg = tf.reshape(generated, [n_H * n_W, n_C])
    generated_transposed = tf.transpose(klg)

    # Computing gram_matrices for both images S and G (≈2 lines)
    style_gram = gram_matrix(style_transposed)
    generated_gram = gram_matrix(generated_transposed)

    # Computing the loss (≈1 line)
    fact = (.5 / (n_H * n_W * n_C)) ** 2
    J_style_layer = fact * tf.reduce_sum(tf.square(style_gram - generated_gram, 2))
    
    ### END CODE HERE ###
    
    return J_style_layer



def generate_noise_image(content_image, noise_ratio = 0.6):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, 224, 224, 3)).astype('float32')
    
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image




# Define the full neural style transfer model
def compute_loss(model, content_image, style_image, generated_image):
   # Extract features from the content and style images
   content_layers = ['block3_conv1']  # Choose a layer for content (block5_conv2 is a good choice)
   style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # Style layers


   # Get the feature outputs for each layer
   outputs = [model.get_layer(layer).output for layer in content_layers + style_layers]
   feature_extractor = tf.keras.Model(inputs=model.input, outputs=outputs)
  
   # Get the feature representations
   content_features = feature_extractor(content_image)
   style_features = feature_extractor(style_image)
   generated_features = feature_extractor(generated_image)
  
   # Compute the losses
   c_loss = content_loss(content_features[0], generated_features[0])
   s_loss = 0
   for sf, gf in zip(style_features[1:], generated_features[1:]):
       s_loss += 0.2 * style_loss(sf, gf)
  


   # Weights for each loss term
   total_loss = 10 * c_loss + 40 * s_loss
   return total_loss


# Create the model
def create_model():
   model = load_vgg19_model()
   return model




   


