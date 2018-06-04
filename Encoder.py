import tensorflow as tf
from tqdm import tqdm
import numpy as np

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
  gen_binary = None
  dec = None
  
  def __init__(self, session, message_size=60, batch_size = 512, noise_multiplier=1):
    
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
    bin_G_z, G_z = self.generator(message, z, isTrain)
    DEC_g = self.decoder(bin_G_z, isTrain)

    self.gen = G_z
    self.gen_binary = bin_G_z
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
  
  def train(self, train_data, epochs=1):
    sess = self.session
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in tqdm(range(epochs)):
      for iter in range(len(train_data) // self.batch_size):
          x_ = train_data[iter*self.batch_size:(iter+1)*self.batch_size]
          message_ = (np.random.randint(2, size=[self.batch_size, self.message_size, 1, 1]) - 0.5) * 2
          z_ = np.random.normal(0, 1, (self.batch_size, self.message_size*self.noise_multiplier, 1, 1))
          
          # discriminator
          sess.run([self.D_optim], {self.x: x_, self.message: message_, self.z: z_, self.isTrain: True})

          # generator
          message_ = (np.random.randint(2, size=[self.batch_size, self.message_size, 1, 1]) - 0.5) * 2
          z_ = np.random.normal(0, 1, (self.batch_size, self.message_size*self.noise_multiplier, 1, 1))
          sess.run([self.G_optim], {self.x: x_, self.message: message_, self.z: z_, self.isTrain: True})

          
          # encoder
          message_ = (np.random.randint(2, size=[self.batch_size, self.message_size, 1, 1]) - 0.5) * 2
          z_ = np.random.normal(0, 1, (self.batch_size, self.message_size*self.noise_multiplier, 1, 1))
          sess.run([self.DEC_optim], {self.x: x_, self.message: message_, self.z: z_, self.isTrain: True})

  def encode(self, message, tries=20):
    sess = self.session
    
    message = np.repeat(message, tries, axis=0)
    z_ = np.random.normal(0, 1, (np.shape(message)[0], self.message_size*self.noise_multiplier, 1, 1))
    gen, binary = sess.run([self.gen, self.gen_binary ], { self.message: message,self.z: z_, self.isTrain: False})
    
    decoded = self.decode(binary)
    
    message = np.reshape(message, [-1, self.message_size])
    decoded = np.reshape(decoded, [-1, self.message_size])
    try:
      index = np.where(np.all(np.equal(message, decoded), axis = 1)==True)[0][0]
      return [gen[index]], [binary[index]], [True]
    except:
      return [gen[0]], [binary[0]], [False]
    
    
  def encodeString(self, string, tries=1000):
    
    images = []
    binaries = []
    bits = tobits(string)
    chunks = int(np.ceil(len(bits) / self.message_size))    
    bits = np.pad(bits, (0, self.message_size * chunks) , 'constant', constant_values=(0, 0))
    
    for i in range(chunks):
      encode_chunk = bits[self.message_size*i:self.message_size*(i+1)]
      encode_chunk = (encode_chunk * 2) - 1
      encode_chunk = np.reshape(encode_chunk, [1, self.message_size, 1, 1])
      found = False
      while not found:
        image_, binary_, f = self.encode(encode_chunk, tries=tries)
        image_ = np.reshape(image_[0], [32, 32])
        if f[0]:
          found = f[0]
          images += [image_]
          binaries += [binary_]
          print("chunk " + str(i+1) + "/" + str(chunks) + " encoded")
        else:
          print("encoding failed, retrying chunk " + str(i+1))
    return images, binaries
  
  def decodeString(self, images):
    codes = []
    for image_ in images:
      image = np.reshape(image_, [1, 32, 32, 1])
      code = np.reshape(self.decode(image), [self.message_size])
      codes.extend(code)
      
    codes = [int((i+1)/2) for i in codes]
    return frombits(codes)
    
    
  def decode(self, image):
    sess = self.session
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    
    return sess.run(self.decoder(x, False, True), {x: image})
    
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
          
          return binarize(out), out
    
  def decoder(self, input_batch, isTrain=True, reuse=False):
      with tf.variable_scope('decoder', reuse=reuse):
          conv1 = tf.layers.conv2d(input_batch, 128, [4, 4], strides=(2, 2), padding='same')
          lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain))

          conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
          lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain))

          conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
          lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain))

          dense1 = tf.layers.dense(tf.contrib.layers.flatten(lrelu3), self.message_size, activation=tf.nn.leaky_relu)    
          out = tf.nn.tanh(dense1)
          return binarize(tf.reshape(out, shape=[-1, self.message_size, 1, 1]))
    
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
    
def binarize(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        return tf.sign(x)
      
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
