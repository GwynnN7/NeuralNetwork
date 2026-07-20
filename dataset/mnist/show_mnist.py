import sys
import struct
import numpy as np
import matplotlib.pyplot as plt

def show_mnist_image(file_path, index):
    try:
        with open(file_path, 'rb') as f:

            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

            if index < 0 or index >= num_images:
                print(f"Error: Index must be between 0 and {num_images - 1}")
                return

            image_size = rows * cols
            offset = 16 + (index * image_size)

            f.seek(offset)
            image_data = f.read(image_size)

            image = np.frombuffer(image_data, dtype=np.uint8).reshape((rows, cols))

            plt.imshow(image, cmap='gray')
            plt.title(f"MNIST Test Set\nIndex: {index}")
            plt.axis('off')
            plt.show()
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_mnist.py <index>")
        sys.exit(1)

    try:
        target_index = int(sys.argv[1])
    except ValueError:
        print("Error: Please provide a valid integer index.")
        sys.exit(1)

    dataset_path = "t10k-images-idx3-ubyte"
    show_mnist_image(dataset_path, target_index)