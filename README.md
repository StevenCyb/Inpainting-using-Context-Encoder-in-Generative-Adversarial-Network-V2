# *Note on current work*
Currently there is a bug, which I will improve in the next days. 
# Inpainting-using-Context-Encoder-in-Generative-Adversarial-Network-V2
This repository contains an inpainting approach with context encoder in a Generative Adversarial Network.
In this project I use a different pre- and post-processing than in the [first version](https://github.com/StevenCyb/Inpainting-using-Context-Encoder-in-Generative-Adversarial-Network).
Moreover the network is more similar against the original paper [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379).

The following figure shows the new pre- and post-processing approach.
At the beginning of the training a random mask with rectangles is created, which are indicated in *point 1* by the green borders. 
Regardless of whether it is the training or the prediction, the input image is masked so that the areas to be reconstructed are white. 
In addition, the contours on the mask are detected using Chain Approx Simple and extended by a factor of *0.5* (can be defined as required), as illustrated in *point 2* by the grey borders.
The extended contours allow the Region Of Interest (ROI) to be cut rectangularly from the image, scaled to 256×256 pixels and used to train the generator, see *point 3*.
This makes the input for the generator more standardized than in the previous version.
In the case of pre-processed input data, the semantic features are always visible on the outer side, while the area to be reconstructed is masked in the middle. 
As a consequence, the network only receives the relevant information and therefore does not relate to the irrelevant environment.
The predictions of the generator can then be scaled back to the original size and placed on the input image, resulting in the output image, see *points 4* and *5*.
In training, the unscaled predictions are used together with the ground truth to train the discriminator.
The ground truth is cut from the input image using the extracted contours and scaled to 128×128 pixels, see *point 6*. 
![Overview](/media/overview.png)
## Small Evaluation
The following figure shows the result of a small evaluation. In this case the network did not see the illustrated object position during the training, so that he was forced to adapt his knowledge.
The result is pretty good, even if it looks slightly blurred due to the small network output size of 128x128 pixels.
![Evaluation-Example](/media/example_results.png)
## How to use it
In the following, a short explanation is given of how the network is trained and how predektion can be made.
The example commands are based on those used in the previously mentioned evaluation.
In both cases you can get more information about available arguments by using the argument `-h`.
### Run Train
To run training on you own dataset you need to first copy the training images into the directory `training`.
I recommend cropping the images so that only the area the context-encoder should learn is included.
Afterwards you can start the training with the following command. In this case the parameter `-mi` indicates that the training should be aborted from an average MSE of 60. If this parameter is not specified, then the parameter for the training iterations will be used instead.
`-mirr` defines the minimum and `-marr` the maximum size of the randomly generated masks. Note that `-marr` and the extension of the ROI (parameter `-br`) together may not be greater than `1.0`.
```
python3 train.py -mi 60 -mirr 0.45 -marr 0.5 -bs 32
```
### Run predictions
To perform a prediction, I used the following command
```
python3 predict.py -i ./training/1.jpg -m ./1_m.jpg -o ./1_result.jpg
```
