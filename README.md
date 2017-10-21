# keras-acgan

This is a simple implementation of AC-GAN on the MNIST dataset, as [introduced by Odena, et al.](https://arxiv.org/abs/1610.09585), in Keras. 

This represents a relatively happy medium between network complexity, ease of understanding, and performance. The samples generated (consult [`acgan-analysis.ipynb`](./acgan-analysis.ipynb) for examples) are clear, consistent, and illustrate the power of the auxiliary classifier. 

To run this, you should have [Keras](https://keras.io) and either [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://www.tensorflow.org/) (preferably TensorFlow) installed. Also, it is strongly advised that you use a GPU with [CuDNN](https://developer.nvidia.com/cudnn), as convolutions are rather slow on CPUs. If you do not have access to a dedicated GPU, I recommend looking at the [Spot Instances](https://aws.amazon.com/ec2/spot/) on AWS.

You can simply run `python mnist_acgan.py`, and it will create:

* `./discriminator_params/epoch_{{epoch_number}}.hdf5`, the discriminator network parameters
* `./generator_params/epoch_{{epoch_number}}.hdf5`, the generator network parameters
* `plot_epoch_{{epoch_number}}_generated.png`, a plot of some generated images

After this is done, you can click through [`acgan-analysis.ipynb`](./acgan-analysis.ipynb) to generate more images and understand the system performance.
