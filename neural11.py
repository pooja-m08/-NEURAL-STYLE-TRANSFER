import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Allow TensorFlow to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load and preprocess an image
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Visualization function
def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [img.shape[2] if img.ndim == 4 else img.shape[1] for img in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)

    for i in range(n):
        plt.subplot(gs[i])
        img = images[i][0] if images[i].ndim == 4 else images[i]
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.title(titles[i] if i < len(titles) else '')
    plt.show()

# Paths to your images
content_path = r"C:\Users\kakhi\Desktop\content_img.jpeg"
style_path = r"C:\Users\kakhi\Desktop\style_img.jpeg"

# Load content and style images
content_image = load_img(content_path)
style_image = load_img(style_path)

# Load style transfer model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Display the original content, style, and stylized images
show_n(
    [content_image, style_image, stylized_image],
    titles=['Original content image', 'Style image', 'Generated image']
)
