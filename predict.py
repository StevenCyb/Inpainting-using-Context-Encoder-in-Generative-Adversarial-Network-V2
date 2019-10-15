import os
import cv2
import argparse
import numpy as np
from gan_ce.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image.")
ap.add_argument("-m", "--mask", required=True, help="Path to the mask.")
ap.add_argument("-o", "--output", required=True, help="Path to save prediction.")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path to the weights (default='./weights/weights.ckpt').")
args = vars(ap.parse_args())

# Verify the passed parameters
if not os.path.isdir(os.path.dirname(args["weights"])):
    raise Exception("Path to weights is invalid.")
if not os.path.isfile(args["image"]):
    raise Exception("Path to image is invalid.")
if not os.path.isfile(args["mask"]):
    raise Exception("Path to mask is invalid.")

# Load the image to inpaint
image = cv2.imread(args["image"], 3)
# Load the mask to inpaint
mask = cv2.imread(args["mask"], 3)

# Initalize the GAN (Context Encoder(Generator) and Discriminator) 
network = Network()
# Load the weights
network.load_weights_generator(weights_path=args["weights"])
# Start prediction and save the results
prediction = network.predict(image, mask)
cv2.imwrite(args["output"], prediction)