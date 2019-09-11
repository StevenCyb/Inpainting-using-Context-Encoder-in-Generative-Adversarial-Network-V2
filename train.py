import os
import cv2
import argparse
from gan_ce.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", required=False, default='./training', help="Path to the training dataset (default='./training').")
ap.add_argument("-it", "--iterations", required=False, type=int, default=50000, help="No. of training iterations (default=10000).")
ap.add_argument("-bs", "--batch_size", required=False, type=int, default=1, help="The batch size (default=1).")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path where to store the weights (default='./weights/weights.ckpt').")
ap.add_argument("-sit", "--saving_iterations", required=False, type=int, default=1000, help="In which steps should the weights be stored (default=1000).")
ap.add_argument("-c", "--checkpoint", required=False, default='', help="Continue with checkpoint from [...].")
ap.add_argument("-mi", "--mse_interrupt", required=False, type=int, default=9999999, help="MSE for generator prediction to interrupt (default=not_used).")
ap.add_argument("-mirr", "--min_rectangle_ratio", required=False, type=float, default=0.1, help="Min. mask-rectangle size default (default=0.1).")
ap.add_argument("-marr", "--max_rectangle_ratio", required=False, type=float, default=0.5, help="Max. mask-rectangle size default (default=0.1).")
ap.add_argument("-br", "--border_ratio", required=False, type=float, default=0.5, help="Ratio to expand the roi (default=0.5).")
args = vars(ap.parse_args())

# Verify the passed parameters
if not os.path.isdir(args["dataset_path"]):
    raise Exception("Path to training dataset is invalid.")
if not isinstance(args["iterations"], int) or args["iterations"] < 1:
    raise Exception("iterations has an invalid value.")
if not isinstance(args["batch_size"], int) or args["batch_size"] < 1:
    raise Exception("Batch size has an invalid value.")
if not os.path.isdir(os.path.dirname(args["weights"])):
    raise Exception("Path to store weights is invalid.")
if not isinstance(args["saving_iterations"], int) or args["saving_iterations"] < 1:
    raise Exception("Saving iterations has an invalid value.")
if not isinstance(args["mse_interrupt"], int):
    raise Exception("MSE-Interrupt has an invalid value.")
if not isinstance(args["min_rectangle_ratio"], float):
    raise Exception("Min. mask-rectangle has an invalid value. Must be a float number smaller than max_rectangle_ratio.")
if not isinstance(args["max_rectangle_ratio"], float) or args["max_rectangle_ratio"] < args["min_rectangle_ratio"]:
    raise Exception("Max. mask-rectangle has an invalid value. Must be a float number and bigger than min_rectangle_ratio and together with border_ratio smaller than 1.0.")
if not isinstance(args["border_ratio"], float) or (args["border_ratio"] + args["max_rectangle_ratio"]) > 1.0:
    raise Exception("Border-Ratio has an invalid value. Must be a float number and together with max_rectangle_ratio smaller than 1.0.")

# Load the training images with has the extension .jpg and .png.
# Convert them into RGB and store in an array 
training_images = []
for image_path in os.listdir(args["dataset_path"]):
    if image_path.endswith(".jpg") or image_path.endswith(".png"):
        training_images.append(cv2.imread(args["dataset_path"] + "/" + image_path, 3))

# Check if at least one image to train exists
if len(training_images) == 0:
    raise Exception("The specified training dataset directory is empty.")

# Initalize the GAN (Context Encoder(Generator) and Discriminator) 
network = Network()

# Load checkpoint if is setted
if args["checkpoint"] != '':
	network.load_weights(weights_path=args["weights"])
# Start training
network.train(images=training_images, iterations=args["iterations"], batch_size=args["batch_size"], weights_path=args["weights"], saving_iterations=args["saving_iterations"], mse_interrupt=args["mse_interrupt"], min_rectangle_ratio=args["min_rectangle_ratio"], max_rectangle_ratio=args["max_rectangle_ratio"], border_ratio=args["border_ratio"])