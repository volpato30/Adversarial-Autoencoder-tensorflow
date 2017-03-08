import tensorflow as tf
from tensorflow.contrib import layers


def encoder(input_tensor, output_size):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    net = layers.fully_connected(input_tensor, 1000)
    net = layers.fully_connected(net, 1000)
    net = layers.fully_connected(net, output_size, activation_fn=None)
    return net


def discriminator(input_tensor):
    '''Create a network that discriminates between latent features encoded by
        encoder and random vector sampled from the desired distribution
    Args:
        input: a batch of real images [2 * batch_size, hidden_units]
    Returns:
        A tensor that represents the network
    '''

    return encoder(input_tensor, 1)


def decoder(input_tensor):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, [batch_size, hidden_units]

    Returns:
        the reconstructed images, [batch_size, 28 * 28]
    '''
    net = layers.fully_connected(input_tensor, 1000)
    net = layers.fully_connected(net, 1000)
    net = layers.fully_connected(net, 28 * 28, activation_fn=tf.nn.sigmoid)
    return net
