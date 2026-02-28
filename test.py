
# import torch, ultralytics, torchreid
# print("Torch:", torch.__version__)
# print("CUDA:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0))
# print("All good")
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

file_path = 'global_person_gallery.pkl' # Replace with your file path
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data)

# Assuming 'data' is a list/array of images, save the first one as an image file
if isinstance(data, list) or isinstance(data, np.ndarray):
    if len(data) > 0:
        plt.imshow(data[0]) # Display the first image in the collection
        plt.title("Image 1")
        plt.savefig('image.png')  # Save the image to a file
        print("Image saved as 'image.png'")

    # To view multiple images, you can use a loop or subplots
    # Example for a few images:
    # for i in range(min(5, len(data))):
    #     plt.figure()
    #     plt.imshow(data[i])
    #     plt.title(f"Image {i+1}")
    #     plt.show()
