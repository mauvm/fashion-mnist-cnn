# Fashion MNIST CNN

My implementation of a Convolutional Neural Network for the [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) benchmark.

I've built this in preparation for the [Deep Learning: Advanced Computer Vision](https://www.udemy.com/advanced-computer-vision/) course on Udemy and improved it based on the course material.

It achieves an accuracy of **92.3%** after training for **5 epochs**.

## Usage

Clone this repository:

```
git clone https://github.com/mauvm/fashion-mnist-cnn --depth=1
```

Then download the Fashion MNIST dataset from Kaggle:

1. Register on kaggle.com
2. Search for "Fashion MNIST" (by Zalando Research)
3. Click on download and extract in `large_files` folder next to the folder of this repository

Run this in your [Anaconda](https://www.anaconda.com/) shell to download all the prerequisites:

```bash
conda create --name fashion-mnist-cnn
conda activate fashion-mnist-cnn
conda install tensorflow numpy pandas
```

Next up, train the model (runs for 20 epochs):

```bash
python train.py
# Epoch 1/5
# 60000/60000 [==============================] - 272s 5ms/sample - loss: 0.5562 - acc: 0.8066
# Epoch 2/5
# 60000/60000 [==============================] - 265s 4ms/sample - loss: 0.3098 - acc: 0.8898
# Epoch 3/5
# 60000/60000 [==============================] - 265s 4ms/sample - loss: 0.2496 - acc: 0.9104
# Epoch 4/5
# 60000/60000 [==============================] - 267s 4ms/sample - loss: 0.2165 - acc: 0.9222
# Epoch 5/5
# 60000/60000 [==============================] - 267s 4ms/sample - loss: 0.1892 - acc: 0.9314
```

Finally test the model:

```bash
python test.py
# 10000/10000 [==============================] - 5s 459us/sample - loss: 0.2107 - acc: 0.9232
# Loss 0.2106759150236845
# Accuracy 0.9232
```

See [config.py](./config.py) and [model.py](./model.py) for the configuration and model.

## License

> This is built using open source tooling and with a dataset that is available for everyone.
> So do whatever you want with this code.
>
> — Maurits van Mastrigt, 2019
