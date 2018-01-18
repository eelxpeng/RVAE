# Relational Variational Autoencoder
This code is associated with the following paper:

Xiaopeng Li and James She. Relational Variational Autoencoder for Link Prediction with Multimedia Data. ACM International Conference on Multimedia Thematic Workshop, 2017 (MM'17).

### Prerequisities
* The code is written in Python 2.7. 
* To run this code you need to have TensorFlow installed. The code is tested with TensorFlow 0.12.1.

### Usage
The program consists of two parts: pre-train in VAE manner and finetuning in RVAE manner. The core code files are in lib/ directory and the test code files are test_vae.py and test_rvae.py. To run the program, you should first run test_vae.py to pre-train the weights of inference network and generation network. The pre-trained weights will be saved under model/ directory. Then test_rvae.py can be run for the RVAE model. And the model will be saved also under model/ directory.

For generating the sweeping curves in the paper, the experiment code is experiment_rvae.py, where the latent dimension is varied with [5, 10, 20, 40, 50], and each experiment is repeated for 5 times.

* The data for citeulike-t is added in data/citeulike-t. And the experiment code for citeulike-t is added for reference in citeulike-t/