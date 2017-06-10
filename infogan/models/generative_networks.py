import tensorflow as tf
import prettytensor as pt
import numpy as np

class GenNetworks(object):
    def __init__(self):
        print "__init__ - ing"


class InfoGAN_mnist_net():
    def __init__(self):
        return

    def infoGAN_mnist_net(self, image_shape):
        image_size = image_shape[0]
        generator_template = \
            (pt.template("input").
             custom_fully_connected(1024).
             fc_batch_norm().
             apply(tf.nn.relu).
             custom_fully_connected(image_size / 4 * image_size / 4 * 128).
             fc_batch_norm().
             apply(tf.nn.relu).
             reshape([-1, image_size / 4, image_size / 4, 128]).
             custom_deconv2d([0, image_size/2, image_size/2, 64], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
             flatten())
        return generator_template


class dcgan_net():
    def __init__(self,image_shape):
        self.gf_dim = 64
        self.c_dim = image_shape[2]
        self.batch_size = 64
        self.k_h = self.k_w = 5

    def gen_net(self, image_shape):
        sx = image_shape[0]
        sy = image_shape[1]
        sx2, sx4, sx8, sx16 = int(np.ceil(sx*1.0/2)), int(np.ceil(sx*1.0/4)), int(np.ceil(sx*1.0/8)), int(np.ceil(sx*1.0/16))
        sy2, sy4, sy8, sy16 = int(np.ceil(sy*1.0/2)), int(np.ceil(sy*1.0/4)), int(np.ceil(sy*1.0/8)), int(np.ceil(sy*1.0/16))

        if (sx == 96) and (sy == 96):
            self.k_h = self.k_w = 5
        elif (sx < 20) or (sy < 20):
            self.k_h = self.k_w = 2
        else:
            self.k_h = self.k_w = 3

        generator_template = \
            (pt.template("input").
             custom_fully_connected(self.gf_dim*8*sx16*sy16, scope='g_h0_lin').
             fc_batch_norm().
             apply(tf.nn.relu).
             reshape([-1, sx16, sy16, self.gf_dim * 8]).
             custom_deconv2d([self.batch_size, sx8, sy8, self.gf_dim*4], name='g_h1', k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([self.batch_size, sx4, sy4, self.gf_dim*2], name='g_h2', k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([self.batch_size, sx2, sy2, self.gf_dim*1], name='g_h3', k_h=self.k_h).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([self.batch_size, sx, sy, self.c_dim], name='g_h4', k_h=self.k_h, k_w=self.k_w).
             apply(tf.nn.tanh,name="OutGenerator"))

        return generator_template


class deeper_dcgan_net():
    def __init__(self,image_shape):
        self.gf_dim = 64
        self.c_dim = image_shape[2]
        self.batch_size = 64
        self.k_h = self.k_w = 5

    def gen_net(self, image_shape):
        s = image_shape[0]
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        if s == 32:
            self.k_h = self.k_w = 3

        generator_template = \
            (pt.template("input").
             custom_fully_connected(self.gf_dim*8*s16*s16, scope='g_h0_lin').
             fc_batch_norm().
             apply(tf.nn.relu).
             reshape([-1, s16, s16, self.gf_dim * 8]).
             custom_deconv2d([self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim*2, name='g_h1_1_conv', d_h=1, d_w=1, k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim*1, name='g_h2_1_conv', d_h=1, d_w=1, k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', k_h=self.k_h).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim, name='g_h3_1_conv', d_h=1, d_w=1, k_h=self.k_h, k_w=self.k_w).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([self.batch_size, s, s, self.c_dim], name='g_h4', k_h=self.k_h, k_w=self.k_w).
             apply(tf.nn.tanh,name="OutGenerator"))

        return generator_template
