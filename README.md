# Sign Langauge Video Generation





This repo is about generation of sign langauge video using GAN. The input given to the model is a skeleton frame from the database of the sign langauge. The Generator is trained on the images of the user. We can use the trained generator for producing new sign language videos from input using the generator.The clip of the video is attached below.

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/48018142/70105307-72fed500-1666-11ea-8cfd-3532f6791c0f.gif)
<br> The video generated is the middle image among three shown above. I know the image is unclear but it does a good job tracing the hand and elbow joints.along with image processing techinques using some more time and epochs for training the GAN could improve the resolution.





# Dataset

The Dataset used is RWTH boston 50 dataset. It consists of 3 signer, 1 male and 2 female. I have taken signer 3 as input to produce signs for new signs that are not there in the database. The signer are attached below.

![rwthboston](https://user-images.githubusercontent.com/48018142/163719274-95d07b72-0ed5-429a-807d-fdd98b52135f.JPG)


# PreProcessing
The video is split to frames and then the time stamp under the image  is manually removed and the resized to proper format.
The skeleton poses are obtained from package called OpenPose. This package helped to get proper signs to some extent but not all the time. The openpose output is not perfect at everytime, in sign langauge the movement of hands and the signs of the hands are meant to be captured. At times these miss due to occulsion and algorithmic errors. These have effected when training the generator model. 

![image](https://user-images.githubusercontent.com/48018142/163720064-7a30721f-86d6-4dbf-a4f0-3e10ecd6f447.png)


# PIX2PIX Gan


PreProcessing of Input Images to GAN:
1.  Resize each 256 x 256 image to a larger height and width—286 x 286.
2.  Randomly crop it back to 256 x 256.
3.  Randomly flip the image horizontally i.e. left to right (random mirroring).
4.  Normalize the images to the [-1, 1] range.


## Generator
The generator of your pix2pix cGAN is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). (You can find out more about it in the Image segmentation tutorial and on the U-Net project website.)
Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
There are skip connections between the encoder and decoder (as in the U-Net).


## Discriminator
Each block in the discriminator is: Convolution -> Batch normalization -> Leaky ReLU.
The shape of the output after the last layer is (batch_size, 30, 30, 1).
Each 30 x 30 image patch of the output classifies a 70 x 70 portion of the input image.
The discriminator receives 2 inputs:
The input image and the target image, which it should classify as real.
The input image and the generated image (the output of the generator), which it should classify as fake.



## Loss Function


### Generator Loss
![loss](https://user-images.githubusercontent.com/48018142/163720289-e091c4fc-b8bb-41e6-99f6-de6c63a4bd88.png)

The generator loss is a sigmoid cross-entropy loss of the generated images and an array of ones.
The pix2pix paper also mentions the L1 loss, which is a MAE (mean absolute error) between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss is gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the paper.

```
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
```



### Discriminator Loss
![disloss](https://user-images.githubusercontent.com/48018142/163720410-43af0ec8-4558-456b-b797-d1c17c0daa3e.png)

The discriminator_loss function takes 2 inputs: real images and generated images.
real_loss is a sigmoid cross-entropy loss of the real images and an array of ones(since these are the real images).
generated_loss is a sigmoid cross-entropy loss of the generated images and an array of zeros (since these are the fake images).
The total_loss is the sum of real_loss and generated_loss.





# Inferences

We can see the hand signs that are blurred, but the GAN is good enough to make the spatial information in most accurate way, we can use a encoder and decoder architecure for more significant way for accurate generation of hand gestures. 

<img src="https://github.com/saisriteja/SignLangaugeVideoGeneration/blob/main/results/content/frames_generated_train/24.png" width="512" height="512">


The GAN is so good that it can even generate the shadows of the hand seemlesly when required.


<img src="https://github.com/saisriteja/SignLangaugeVideoGeneration/blob/main/results/content/frames_generated_train/4.png" width="512" height="512">
<img src="https://github.com/saisriteja/SignLangaugeVideoGeneration/blob/main/results/content/frames_generated_train/26.png" width="512" height="512">















