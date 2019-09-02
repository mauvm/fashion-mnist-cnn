# Fashion MNIST CNN

My implementation of a Convolutional Neural Network for the [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) benchmark.
I've built this in preparation for the [Deep Learning: Advanced Computer Vision](https://www.udemy.com/advanced-computer-vision/) course on Udemy.
It achieves an accuracy of **91.4%** after training for **20 epochs**.

## Usage

Clone this repository:

```
git clone https://github.com/mauvm/fashion-mnist-cnn --depth=1
```

Then download the Fashion MNIST dataset from Kaggle:

1. Register on kaggle.com
2. Search for "Fashion MNIST" (by Zalando Research)
3. Click on download and extract in `large_files` folder next to this repository

Run this in your [Anaconda](https://www.anaconda.com/) shell to download all the prerequisites:

```bash
conda create --name fashion-mnist-cnn
conda activate fashion-mnist-cnn
conda install tensorflow numpy pandas
```

Next up, train the model (runs for 20 epochs):

```bash
python train.py
# Epoch 1/20
# 60000/60000 [==============================] - 68s 1ms/sample - loss: 1.8907 - acc: 0.7365
# Epoch 2/20
# 60000/60000 [==============================] - 68s 1ms/sample - loss: 0.3270 - acc: 0.8825
# Epoch 3/20
# 24200/60000 [===========>..................] - ETA: 16s - loss: 0.3538 - acc: 0.8605
# ...
```

Finally test the model:

```bash
python test.py
# 10000/10000 [==============================] - 4s 375us/sample - loss: 0.3056 - acc: 0.9140
# Loss 0.30564905689656735
# Accuracy 0.914
```

See [config.py](./config.py) and [model.py](./model.py) for the configuration and model.

## License

> This is built using open source tooling and with a dataset that is available for everyone.
> So do whatever you want with this code.
>
> - Maurits van Mastrigt, 2019
