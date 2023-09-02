# VAE Image Generation

This program generates images using a Variational Autoencoder (VAE) model and applies effects to them.

## How It Works

The program consists of a Python script that uses TensorFlow and Keras to create and train a VAE model. Here's how it works:

1. The VAE model is built with an encoder and a decoder.
2. The encoder takes input images and encodes them into a lower-dimensional latent space.
3. The decoder takes random points in the latent space and decodes them into images.
4. The generated images are then processed with effects, including resizing, adding gradient backgrounds, randomizing colors, and applying Gaussian blur.
5. The processed images are saved to an output directory.

## Installation

Follow these steps to set up and run the program:

1. **Clone the Repository**: Clone this repository to your local machine and navigate to the cloned directory.

   ```bash
   git clone https://github.com/itzreqle/vae-image-generation.git
   cd vae-image-generation
   ```
   
2. Install Dependencies: Install the required Python dependencies.

  ```bash
  pip install tensorflow opencv-python-headless numpy scipy
  ```

Make sure you have Python and pip installed.

3. Run the Program: Execute the vae_image_generation.py script with the desired command-line arguments.
Example:

  ```bash
  python vae_image_generation.py --height 900 --width 900 --latent_dim 100 --channels 3 --num_samples 10 --output_dir samples
  ```

You can customize the program's behavior by adjusting the command-line arguments. Refer to the How It Works section for more details on the available options.

4. View Generated Images: The generated images with effects will be saved in the specified output directory (samples in the example above). You can open and view them using an image viewer.

## Command-Line Arguments

Here are the available command-line arguments and their descriptions:

- `--height`: Desired image height (default: 900)
- `--width`: Desired image width (default: 900)
- `--latent_dim`: Latent dimension (default: 100)
- `--channels`: Number of image channels (default: 3, e.g., for RGB)
- `--num_samples`: Number of images to generate (default: 10)
- `--output_dir`: Output directory for generated images (default: "samples")

Feel free to adjust these arguments to control the image generation process.

## License

This project is licensed under the MIT License. See the [LICENSE](https://chat.openai.com/LICENSE) file for details.
