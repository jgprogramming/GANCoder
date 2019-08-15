import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools

class Encoder:
  
  session = None
  message_size = None
  batch_size = None
  noise_multiplier = None
  
  D_optim = None
  G_optim = None
  DEC_optim = None
  
  x = None
  message = None
  isTrain = None
  z = None
  
  gen = None
  dec = None
  message_fixed = None
  noise_fixed = None
  
  def __init__(self, session, message_size=60, batch_size = 512, noise_multiplier=1):
    
    self.message_fixed = (random_message_sample((16, message_size, 1, 1)) - 0.5) * 2
    self.noise_fixed = np.random.normal(0, 1, (16, message_size*noise_multiplier, 1, 1))
    self.session = session
    self.message_size = message_size
    self.batch_size = batch_size
    self.noise_multiplier = noise_multiplier
    lr = 0.001
    
    # placeholders
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    # noise
    z = tf.placeholder(tf.float32, shape=(None, self.message_size*self.noise_multiplier, 1, 1))
    
    # message
    message = tf.placeholder(tf.float32, shape=(None, self.message_size, 1, 1))
    isTrain = tf.placeholder(dtype=tf.bool)

    self.x = x
    self.z = z
    self.message = message
    self.isTrain = isTrain
    
    # generator - fake
    G_z = self.generator(message, z, isTrain)
    DEC_g = self.decoder(G_z, isTrain)

    self.gen = G_z
    self.dec = DEC_g
    
    # discriminator - real
    D_real, D_real_logits = self.discriminator(x, isTrain)

    # discriminator - fake
    D_fake, D_fake_logits = self.discriminator(G_z, isTrain, reuse=True)

    # loses
    # disc
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
    D_loss = D_loss_real + D_loss_fake

    # gen
    DEC_loss = tf.losses.mean_squared_error(DEC_g, message)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

    # trainable variables
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator') or var.name.startswith('decoder')]

    # optimizers
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.D_optim = tf.train.AdamOptimizer(lr, beta1=0.4).minimize(D_loss, var_list=D_vars)
        self.G_optim = tf.train.AdamOptimizer(lr, beta1=0.4).minimize(G_loss, var_list=G_vars)
        self.DEC_optim = tf.train.AdamOptimizer(lr, beta1=0.4).minimize(DEC_loss, var_list=G_vars)
  
    init = tf.global_variables_initializer()
    self.session.run(init)
  
  def train(self, train_data, epochs=1):
    sess = self.session
    for epoch in range(epochs):
      for iter in range(len(train_data) // self.batch_size):
          x_ = train_data[iter*self.batch_size:(iter+1)*self.batch_size]
          message_ = (random_message_sample((self.batch_size, self.message_size, 1, 1)) - 0.5) * 2
          z_ = np.random.normal(0, 1, (self.batch_size, self.message_size*self.noise_multiplier, 1, 1))
          
          # discriminator
          sess.run([self.D_optim], {self.x: x_, self.message: message_, self.z: z_, self.isTrain: True})

          # generator
          message_ = (random_message_sample((self.batch_size, self.message_size, 1, 1)) - 0.5) * 2
          z_ = np.random.normal(0, 1, (self.batch_size, self.message_size*self.noise_multiplier, 1, 1))
          sess.run([self.G_optim], {self.x: x_, self.message: message_, self.z: z_, self.isTrain: True})

          # encoder
          message_ = (random_message_sample((self.batch_size, self.message_size, 1, 1)) - 0.5) * 2
          z_ = np.random.normal(0, 1, (self.batch_size, self.message_size*self.noise_multiplier, 1, 1))
          sess.run([self.DEC_optim], {self.x: x_, self.message: message_, self.z: z_, self.isTrain: True})
    
  def save(self, filename):
    sess = self.session
    saver = tf.train.Saver()
    saver.save(sess, filename)
    
  def load(self, filename):
    sess = self.session
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(filename))
    
  def generator(self, message, noise, isTrain=True, reuse=False):
      with tf.variable_scope('generator', reuse=reuse):
          x = tf.reshape(tf.contrib.layers.flatten(tf.concat([tf.cast(message, tf.float32), tf.cast(noise, tf.float32)],1)), [-1, 1, 1, self.message_size + self.message_size*self.noise_multiplier])

          conv1 = tf.layers.conv2d_transpose(x, 512, [4, 4], strides=(2, 2), padding='valid')
          lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain))

          conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
          lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain))

          conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2), padding='same')
          lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain))

          conv4 = tf.layers.conv2d_transpose(lrelu3, 1, [4, 4], strides=(2, 2), padding='same')
          out = tf.nn.tanh(conv4)

          return out
    
  def decoder(self, input_batch, isTrain=True, reuse=False):
      with tf.variable_scope('decoder', reuse=reuse):
          conv1 = tf.layers.conv2d(input_batch, 128, [4, 4], strides=(2, 2), padding='same')
          lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain))

          conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
          lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain))

          conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
          lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain))

          out = tf.layers.dense(tf.contrib.layers.flatten(lrelu3), self.message_size, activation=tf.nn.tanh)    
          return tf.reshape(out, shape=[-1, self.message_size, 1, 1])
      
  def discriminator(self, x, isTrain=True, reuse=False):
      with tf.variable_scope('discriminator', reuse=reuse):
          conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
          lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain))

          conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
          lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain))

          conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
          lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain))

          conv4 = tf.layers.conv2d(lrelu3, 1, [4, 4], strides=(1, 1), padding='valid')
          out = tf.nn.sigmoid(conv4)

          return out, conv4
        

  def encode(self, message):   
    self.session.run(self.gen, { self.message: self.message_fixed, self.z: self.noise_fixed, self.isTrain: False})
    
  def test(self, epoch, sample = 10000):
    
    test_images = self.session.run(self.gen, { self.message: self.message_fixed, self.z: self.noise_fixed, self.isTrain: False})
    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(4, 4))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (32, 32)).T, cmap='gray')

    message = (random_message_sample((sample, self.message_size, 1, 1)) - 0.5) * 2
    noise = (random_message_sample((sample, self.message_size, 1, 1)) - 0.5) * 2

    output_m = self.decoder(self.gen, False, True)
    
    out_m_ = self.session.run(output_m, { self.message: message, self.z: noise, self.isTrain: False})
    
    out_m_ = np.reshape(np.where(out_m_>0, 1, -1), (sample, self.message_size))
    
    acc = np.sum(np.all(out_m_ == np.reshape(message, (sample, self.message_size)), axis=1)) / sample
    
    label = 'Epoch: ' + str(epoch) + ", decode accuracy: " + str(acc)
    fig.text(0.5, 0.04, label, ha='center')
    return acc, fig


  
def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)
  
import random
def random_message_sample(shape=None):
  leng = shape[0]
  point_shape = shape[1:]
  out = []
  for i in range(0, leng):
    x = randbitlist(point_shape[0] * point_shape[1] * point_shape[2])
    out.append(np.reshape(x, point_shape))
  return np.array(out)
  
def randbitlist(n):
    n_on = random.randint(0, n)
    n_off = n - n_on
    result = [1]*n_on + [0]*n_off
    random.shuffle(result)
    return result
