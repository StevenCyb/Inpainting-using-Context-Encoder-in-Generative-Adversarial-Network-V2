import os
import cv2
import numpy as np
from copy import deepcopy
from random import randint
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
import gan_ce.network_utils as nu

class Network:
    def __init__(self, epsilon=1e-10, net_input_size=(256, 256), net_output_size=(128, 128)):
        # Network size
        self.net_input_size = net_input_size
        self.net_output_size = net_output_size
        
        # To store avg. mse
        self.avg_mse = None

        # Reset old session stuff because of the recovery bug (see https://github.com/tflearn/tflearn/issues/527)
        tf.reset_default_graph()

        # Create a flag to set is is training phase or not
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # Discriminator input for ground truth
        self.x_in = tf.placeholder(tf.float32, [None, net_output_size[0], net_output_size[1], 3])
        # Generator input for masked images 
        self.masked_x_in = tf.placeholder(tf.float32, [None, net_input_size[0], net_input_size[1], 3])
        
        # Creat the Generator
        self.generator = self.create_generator(self.masked_x_in)
        # Create the adversial discriminator with input for ground truth
        self.real_discriminator = self.create_discriminator(self.x_in, reuse=tf.AUTO_REUSE)
        # Reuse the created adversial discriminator and set the Generator output as Input (for the fake images)
        self.fake_discriminator = self.create_discriminator(self.generator, reuse=tf.AUTO_REUSE)

        # Define the discriminator loss by calculating the adverssarial los (like in the paper)
        self.discriminator_loss = -tf.reduce_mean(tf.log(self.real_discriminator + epsilon) + tf.log(1 - self.fake_discriminator + epsilon))
        # Define the generator loss by calculating the joint loss of GAN loss and L2 reconstruction loss
        self.generator_loss = -tf.reduce_mean(tf.log(self.fake_discriminator + epsilon)) + 100*tf.reduce_mean(tf.reduce_sum(tf.square(self.x_in - self.generator), [1, 2, 3]))

        # Define a Adam Optimizer to train the disciminator with the respective loss
        self.discriminator_optimizer = tf.train.AdamOptimizer(2e-4).minimize(self.discriminator_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator"))
        # Define a Adam Optimizer to train the generator with the respective loss
        self.generator_optimizer = tf.train.AdamOptimizer(2e-4).minimize(self.generator_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))

        # Create tensorflow session and 
        self.sess = tf.Session()
        # Initialize instanced variables
        self.sess.run(tf.global_variables_initializer())

    def load_weights(self, weights_path='./weights/weights.ckpt'):
        if not os.path.isdir(os.path.dirname(weights_path)):
            raise Exception("Cannot finde weights on path '" + weights_path + "'")
        # Load the weights for the generator and discriminator
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_path)
        print('Weights loaded.')
		
    def load_weights_generator(self, weights_path='./weights/weights.ckpt'):
        if not os.path.isdir(os.path.dirname(weights_path)):
            raise Exception("Cannot finde weights on path '" + weights_path + "'")
        # Load the weights for the generator only
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
        saver.restore(self.sess, weights_path)

    def create_generator(self, input):
        # Create a generator model as described in the paper
        with tf.variable_scope("generator", reuse=None):
            ## Encoder
            x = nu._leaky_relu(nu._conv2d(input, "conv1", filters=64, kernel_size=4,  strides=2, padding="same"))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv2", filters=64, kernel_size=4, strides=2, padding="same"), "conv2", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv3", filters=128, kernel_size=4, strides=2, padding="same"), "conv3", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv4", filters=256, kernel_size=4, strides=2, padding="same"), "conv4", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv5", filters=512, kernel_size=4, strides=2, padding="same"), "conv5", is_training=self.is_training))
            ## Bottleneck
            x = nu._fully_connected_2d(x, "fc1")
            ## Decoder
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans1", filters=512, kernel_size=4, strides=2, padding="same"), "conv_trans1", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans2", filters=256, kernel_size=4, strides=2, padding="same"), "conv_trans2", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans3", filters=128, kernel_size=4, strides=2, padding="same"), "conv_trans3", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d_transpose(x, "conv_trans4", filters=64, kernel_size=4, strides=2, padding="same"), "conv_trans4", is_training=self.is_training))
            ## Output
            x = tf.nn.tanh(nu._conv2d(x, "conv_out", filters=3, kernel_size=4, strides=1, padding="same"))
            return x

    def create_discriminator(self, input,  reuse=None):
        # Create a discriminator model as described in the paper
        with tf.variable_scope("discriminator", reuse=reuse):
            x = nu._leaky_relu(nu._conv2d(input, "conv1", filters=64, kernel_size=4,  strides=2, padding="same"))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv2", filters=64, kernel_size=4, strides=2, padding="same"), "conv2", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv3", filters=128, kernel_size=4, strides=2, padding="same"), "conv3", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv4", filters=256, kernel_size=4, strides=2, padding="same"), "conv4", is_training=self.is_training))
            x = nu._leaky_relu(nu._batch_norm(nu._conv2d(x, "conv5", filters=512, kernel_size=4, strides=2, padding="same"), "conv5", is_training=self.is_training))
            x = nu._fully_connected(x, "fc2")
            x = tf.sigmoid(x)
            return x

    def calculate_mse(self, batch_size, a, b):
        mse = 0.0
        for i in range(batch_size):
            err = np.sum((((a[i, :, :, :] + 1) * 127.5) - ((b[i, :, :, :] + 1) * 127.5)) ** 2)
            err /= float(self.net_output_size[0] * self.net_output_size[1])
            mse += err
        return mse / batch_size

    def do_preprocessing(self, input_image=None, gt_image=None, mask=None, border_ratio=1.0, net_input_size=(256, 256), net_output_size=(128, 128)):
        # Clean values
        mask[np.where((mask<=[125,125,125]).all(axis=2))] = 0
        mask[np.where((mask>[125,125,125]).all(axis=2))] = 255

        # Detect contours (region of interest)
        contours, _ = cv2.findContours(image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
        
        border_imge = deepcopy(input_image)
        # Mask the areas that should be inpainted by setting them to [255, 255, 255]
        border_imge[np.where((mask==[255, 255, 255]).all(axis=2))] = [255, 255, 255]
        # Create image with white border
        border = int(input_image.shape[1] * border_ratio)
        border_imge = cv2.copyMakeBorder(border_imge, top=border, bottom=border, left=border, right=border, borderType= cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        inputs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if gt_image is not None:
                gt_image = (cv2.resize(gt_image[y: y + h, x: x + w], net_output_size) / 127.5) - 1.0
            inputs.append(
                [
                    [ # Inner boundary
                        [x, y], # Top-left corner
                        [x + w, y + h] # Bottom-right corner
                    ],
                    [gt_image], # Ground through
                    [(cv2.resize(border_imge[y - int(h * border_ratio) + border: y + h + int(h * border_ratio) + border, x - int(w * border_ratio) + border: x + w + int(w * border_ratio) + border], net_input_size) / 127.5) - 1.0] # Network input
                ]
            )
        return inputs

    def undo_preprocessing(self, image, prediction_image, roi):
        prediction_image = cv2.resize(prediction_image, (roi[1][0] - roi[0][0], roi[1][1] - roi[0][1]))
        image[roi[0][1]: roi[1][1], roi[0][0]: roi[1][0]] = ((prediction_image + 1) * 127.5).astype(np.uint8)
        return image

    def train(self, images=[], iterations=50000, batch_size=1, weights_path='./weights/weights.ckpt.index', saving_iterations=1000, mse_interrupt=9999999, min_rectangle_ratio=0.1, max_rectangle_ratio=0.3, border_ratio=1.0):
        saver = tf.train.Saver()
        iteration = 0
        while iteration <= iterations or mse_interrupt <= self.avg_mse:
            # Check if MSE is None and set it to 0.0
            if self.avg_mse is None:
                self.avg_mse = 0.0
            # Create two batches for masked images (input for Generator) and a batch with the ground truth (for the "real Disciminator")
            masked_batch = np.zeros([batch_size, self.net_input_size[0], self.net_input_size[1], 3])
            ground_truth_batch = np.zeros([batch_size, self.net_output_size[0], self.net_output_size[1], 3])

            # Now we fill the batches with n random tiles
            batch_idx = 0
            for index in np.random.randint(0, len(images), [batch_size]):
                # Create a random mask with one rectangle
                random_mask = np.zeros([images[index].shape[0], images[index].shape[1]])
                random_mask = np.dstack((random_mask, random_mask, random_mask)).astype(np.uint8)
                w, h = np.random.randint(int(images[index].shape[1] * min_rectangle_ratio), int(images[index].shape[1] * max_rectangle_ratio)), np.random.randint(int(images[index].shape[0] * min_rectangle_ratio), int(images[index].shape[0] * max_rectangle_ratio))
                x, y = np.random.randint(w * border_ratio, int(images[index].shape[1] - (w * border_ratio * 2))), np.random.randint(h * border_ratio, int(images[index].shape[0] - (h * border_ratio * 2)))
                cv2.rectangle(random_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
                # Preprocess input
                processed_data = self.do_preprocessing(input_image=deepcopy(images[index]), gt_image=deepcopy(images[index]), mask=random_mask, border_ratio=border_ratio, net_input_size=self.net_input_size, net_output_size=self.net_output_size)
                # Add to batch
                masked_batch[batch_idx] = processed_data[0][2][0]
                ground_truth_batch[batch_idx] = processed_data[0][1][0]
                batch_idx += 1
            
            # Train the Disciminator
            self.sess.run(self.discriminator_optimizer, feed_dict={self.masked_x_in: masked_batch, self.x_in: ground_truth_batch, self.is_training:True})
            # Train the Generator
            self.sess.run(self.generator_optimizer, feed_dict={self.masked_x_in: masked_batch, self.x_in: ground_truth_batch, self.is_training:True})
            # Get the Discriminator and Generator loss 
            [discriminator_loss, generator_loss, predicted_image] = self.sess.run([self.discriminator_loss, self.generator_loss, self.generator],
                        feed_dict={self.masked_x_in: masked_batch, self.x_in: ground_truth_batch, self.is_training: False})

            # Calculate the current and avg. mse
            mse = self.calculate_mse(batch_size, predicted_image, ground_truth_batch)
            current_total = saving_iterations * batch_size
            if iteration < saving_iterations:
                current_total = iteration * batch_size
            self.avg_mse = ((self.avg_mse * current_total) + (mse * batch_size)) / (current_total + batch_size)

            # Print the current epoch and losses
            print(("Iteration: %d/%d, Discriminator_loss: %g, Generator_loss: %g, Current-MSE: %g, Avg.-MSE: %g" % (iteration, iterations - 1, discriminator_loss, generator_loss, round(mse, 2), round(self.avg_mse, 2))))
            
            # Save the weights if the specified epoch reached
            if (iteration + 1) % saving_iterations == 0:
                print("Save weights")
                saver.save(self.sess, weights_path)
            # Increase iteration counter
            iteration += 1

    def predict(self, image, mask, border_ratio=1.0):
        # Preprocess input
        processed_data = self.do_preprocessing(input_image=deepcopy(image), mask=mask, border_ratio=border_ratio, net_input_size=self.net_input_size, net_output_size=self.net_output_size)
        if len(processed_data) > 0:
            # Create a batch with the masked image 
            input_image_masked = np.zeros([len(processed_data), self.net_input_size[0], self.net_input_size[1], 3])
            for prob in processed_data:
                input_image_masked[0, :, :, :] = prob[2][0]
            # Let the Generator create a prediction
            predicted_image = self.sess.run(self.generator, feed_dict={self.masked_x_in: input_image_masked, self.is_training: False})
            # Undo the preprocessing to get BGR image with normal color range
            for i in range(len(processed_data)):
                image = self.undo_preprocessing(image, predicted_image[i,:,:,:], processed_data[i][0])
        return image
