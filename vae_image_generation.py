import argparse
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import cv2
from scipy.ndimage import zoom

class VAEGenerator:
    def __init__(self, args):
        self.args = args
        self.desired_height = args.height
        self.desired_width = args.width
        self.latent_dim = args.latent_dim
        self.channels = args.channels
        self.decoder = self.build_decoder()  # Create the decoder instance

    def build_encoder(self):
        # Define the encoder architecture
        encoder_input = tf.keras.Input(shape=(self.desired_height, self.desired_width, self.channels))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_input)
        # Add more encoder layers as needed
        flatten = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim)(flatten)
        z_log_var = layers.Dense(self.latent_dim)(flatten)

        # Define the sampling function for reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch_size = tf.shape(z_mean)[0]
            latent_dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])
        encoder = Model(encoder_input, [z_mean, z_log_var, z])
        return encoder

    def build_decoder(self):
        # Define the decoder architecture
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense((self.desired_height // 4) * (self.desired_width // 4) * 64, activation="relu")(decoder_input)
        x = layers.Reshape((self.desired_height // 4, self.desired_width // 4, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        # Add more decoder layers as needed
        decoded_output = layers.Conv2DTranspose(self.channels, 3, activation="sigmoid", padding="same")(x)

        decoder = Model(decoder_input, decoded_output)
        return decoder

    def generate_images(self, num_samples):
        # Generate images from random latent points
        latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        generated_images = self.decoder.predict(latent_points)  # Use self.decoder
        return generated_images

    def apply_effects(self, images):
        blurred_images = []
        for img in images:
            # Resize the generated image to match the desired size
            img = zoom(img, (self.desired_height / img.shape[0], self.desired_width / img.shape[1], 1), order=1)

            # Create a gradient background of the desired size
            gradient_background = np.linspace(0, 1, self.desired_height)[:, np.newaxis]
            gradient_background = np.repeat(gradient_background, self.desired_width, axis=1)
            gradient_background = np.stack([gradient_background] * 3, axis=-1)  # Make it RGB

            # Overlay the generated image onto the gradient background
            img_with_background = img * gradient_background

            # Randomize colors
            random_color = np.random.rand(3)
            img_with_random_color = img_with_background * random_color

            # Apply Gaussian blur
            blurred_img = cv2.GaussianBlur(img_with_random_color, (15, 15), 0)

            blurred_images.append(blurred_img)

        return blurred_images

    def save_generated_images(self, generated_images, output_dir):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through generated images and save them
        for i, img in enumerate(generated_images):
            # Construct the filename (you can use a numbering scheme or other unique identifiers)
            filename = os.path.join(output_dir, f"generated_image_{i}.png")

            # Ensure the image is in the correct format (0-255 integer values)
            img = (img * 255).astype(np.uint8)

            # Save the image using OpenCV
            cv2.imwrite(filename, img)

    def run(self):
        # Generate images and apply effects
        generated_images = self.generate_images(self.args.num_samples)
        images_with_effects = self.apply_effects(generated_images)

        # Save generated images
        self.save_generated_images(images_with_effects, self.args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Variational Autoencoder (VAE) Image Generation")
    parser.add_argument("--height", type=int, default=900, help="Desired image height")
    parser.add_argument("--width", type=int, default=900, help="Desired image width")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels (e.g., 3 for RGB)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory for generated images")

    args = parser.parse_args()

    generator = VAEGenerator(args)
    generator.run()

if __name__ == "__main__":
    main()
