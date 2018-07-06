from __future__ import absolute_import
from __future__ import print_function
import uuid
import time
import os
import numpy as np
import tensorflow as tf
import six
import six.moves.cPickle as pickle
from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty

class cachedproperty(object):
  """Simplified version of https://github.com/pydanny/cached-property"""
  def __init__(self, function):
    self.__doc__ = getattr(function, '__doc__')
    self.function = function

  def __get__(self, instance, klass):
    if instance is None: return self
    value = instance.__dict__[self.function.__name__] = self.function(instance)
    return value

def l1_loss(x):
  return tf.reduce_sum(tf.abs(x))

def l2_loss(x):
  return tf.nn.l2_loss(x)

def squared_cos_sim(v,w,eps=1e-6):
  num = tf.reduce_sum(v*w, axis=1)**2
  den = tf.reduce_sum(v*v, axis=1)*tf.reduce_sum(w*w, axis=1)
  return num / (den + eps)

def train_diverse_models(cls, n, X, y,
    grad_quantity='binary_logit_input_gradients',
    lambda_overlap=0.01, **kw):
  models = [cls() for _ in range(n)]
  igrads = [getattr(m, grad_quantity) for m in models]

  regular_loss = tf.add_n([m.loss_function(**kw) for m in models])
  diverse_loss = tf.add_n([tf.reduce_sum(squared_cos_sim(igrads[i], igrads[j]))
                           for i in range(n)
                           for j in range(i+1, n)]) * lambda_overlap

  loss = regular_loss + diverse_loss

  ops = { 'xent': regular_loss, 'same': diverse_loss }
  for i, m in enumerate(models, 1):
    ops['acc{}'.format(i)] = m.accuracy

  data = train_batches(models, X, y, **kw)

  with tf.Session() as sess:
    minimize(sess, loss, data, operations=ops, **kw)
    for m in models:
      m.vals = [v.eval() for v in m.vars]

  return models

"""
Class attempting to make Tensorflow models more object-oriented
and similar to sklearn's fit/predict interface.
"""
@add_metaclass(ABCMeta)
class NeuralNetwork():
  def __init__(self, name=None, dtype=tf.float32, **kwargs):
    self.vals = None
    self.name = (name or str(uuid.uuid4()))
    self.dtype = dtype
    self.setup_model(**kwargs)
    assert(hasattr(self, 'X'))
    assert(hasattr(self, 'y'))
    assert(hasattr(self, 'logits'))

  def setup_model(self, X=None, y=None, **kw):
    with tf.name_scope(self.name):
      self.X = tf.placeholder(self.dtype, self.x_shape, name="X") if X is None else X
      self.y = tf.placeholder(self.dtype, self.y_shape, name="y") if y is None else y
      self.is_train = tf.placeholder_with_default(
          tf.constant(False, dtype=tf.bool), shape=(), name="is_train")
    self.model = self.rebuild_model(self.X, **kw)
    self.recompute_vars()

  @property
  def logits(self):
    return self.model[-1]

  def rebuild_model(self, X, reuse=None, **kw):
    """Define all of your Tensorflow variables here, making sure to scope them
    under `self.name`, and also making sure to return a list/tuple whose final element
    is your network's logits. In subclasses, remember to call super!"""

  @abstractproperty
  def x_shape(self):
    """Specify the shape of X; for MNIST, this could be [None, 784]"""

  @abstractproperty
  def y_shape(self):
    """Specify the shape of y; for MNIST, this would be [None, 10]"""

  @property
  def num_features(self):
    return np.product(self.x_shape[1:])

  @property
  def num_classes(self):
    return np.product(self.y_shape[1:])

  @property
  def trainable_vars(self):
    return [v for v in tf.trainable_variables() if v in self.vars]

  def input_grad(self, f):
    return tf.gradients(f, self.X)[0]

  def cross_entropy_with(self, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y))

  @cachedproperty
  def preds(self):
    return tf.argmax(self.logits, axis=1)

  @cachedproperty
  def probs(self):
    return tf.nn.softmax(self.logits)

  @cachedproperty
  def logps(self):
    return self.logits - tf.reduce_logsumexp(self.logits, 1, keep_dims=True)

  @cachedproperty
  def grad_sum_logps(self):
    return self.input_grad(self.logps)

  @cachedproperty
  def l1_weights(self):
    return tf.add_n([l1_loss(v) for v in self.trainable_vars])

  @cachedproperty
  def l2_weights(self):
    return tf.add_n([l2_loss(v) for v in self.trainable_vars])

  @cachedproperty
  def cross_entropy(self):
    return self.cross_entropy_with(self.y)

  @cachedproperty
  def cross_entropy_input_gradients(self):
    return self.input_grad(self.cross_entropy)

  @cachedproperty
  def predicted_logit_input_gradients(self):
    return self.input_grad(self.logits * self.y)

  @cachedproperty
  def l1_double_backprop(self):
    return l1_loss(self.cross_entropy_input_gradients)

  @cachedproperty
  def l2_double_backprop(self):
    return l2_loss(self.cross_entropy_input_gradients)

  @cachedproperty
  def l1_grad_sum_logps(self):
    return l1_loss(self.grad_sum_logps)

  @cachedproperty
  def l2_grad_sum_logps(self):
    return l2_loss(self.grad_sum_logps)

  @cachedproperty
  def l1_binary_logit_grads(self):
    return l1_loss(self.binary_logit_input_gradients)

  @cachedproperty
  def l2_binary_logit_grads(self):
    return l2_loss(self.binary_logit_input_gradients)

  @cachedproperty
  def binary_logits(self):
    assert(self.num_classes == 2)
    return self.logps[:,1] - self.logps[:,0]

  @cachedproperty
  def binary_logit_input_gradients(self):
    return self.input_grad(self.binary_logits)

  @cachedproperty
  def accuracy(self):
    return tf.reduce_mean(tf.cast(tf.equal(self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))

  def score(self, X, y, **kw):
    if len(y.shape) == 2:
      return np.mean(self.predict(X, **kw) == np.argmax(y, 1))
    else:
      return np.mean(self.predict(X, **kw) == y)

  def score_(self, sess, X, y, **kw):
    if len(y.shape) == 2:
      return np.mean(self.batch_eval(sess, self.preds, X, **kw) == np.argmax(y, 1))
    else:
      return np.mean(self.batch_eval(sess, self.preds, X, **kw) == y)

  def predict(self, X, **kw):
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.preds, X, **kw)

  def predict_logits(self, X, **kw):
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.logits, X, **kw)

  def predict_binary_logodds(self, X, **kw):
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.binary_logits, X, **kw)

  def predict_proba(self, X, **kw):
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_eval(sess, self.probs, X, **kw)

  def batch_eval(self, sess, quantity, X, n=256):
    vals = sess.run(quantity, feed_dict={ self.X: X[:n] })
    stack = np.vstack if len(vals.shape) > 1 else np.hstack
    for i in range(n, len(X), n):
      vals = stack((vals, sess.run(quantity, feed_dict={ self.X: X[i:i+n] })))
    return vals

  def input_gradients(self, X, y=None, n=256, **kw):
    """Batched version of input gradients"""
    with tf.Session() as sess:
      self.init(sess)
      return self.batch_input_gradients_(sess, X, y, n, **kw)

  def batch_input_gradients_(self, sess, X, y=None, n=256, **kw):
    yy = y[:n] if y is not None and not isint(y) else y
    grads = self.input_gradients_(sess, X[:n], yy, **kw)
    for i in range(n, len(X), n):
      yy = y[i:i+n] if y is not None and not isint(y) else y
      grads = np.vstack((grads,
        self.input_gradients_(sess, X[i:i+n], yy, **kw)))
    return grads

  def input_gradients_(self, sess, X, y=None, logits=False, quantity=None):
    if quantity is not None:
      return sess.run(quantity, feed_dict={ self.X: X })
    if y is None:
      return sess.run(self.grad_sum_logps, feed_dict={ self.X: X })
    elif logits and self.num_classes == 2:
      return sess.run(self.binary_logit_input_gradients, feed_dict={ self.X: X })
    elif isint(y):
      y = onehot(np.array([y]*len(X)), self.num_classes)
    feed = { self.X: X, self.y: y }
    if logits:
      return sess.run(self.predicted_logit_input_gradients, feed_dict=feed)
    else:
      return sess.run(self.cross_entropy_input_gradients, feed_dict=feed)

  def loss_function(self,
      l1_weights=0., l2_weights=0.,
      l1_grad_sum_logps=0., l2_grad_sum_logps=0.,
      l1_double_backprop=0., l2_double_backprop=0.,
      l1_binary_logit_grads=0., l2_binary_logit_grads=0.,
      **kw):
    log_likelihood = self.cross_entropy
    log_prior = 0
    for reg in ['l1_double_backprop', 'l2_double_backprop',
                'l1_weights', 'l2_weights',
                'l1_grad_sum_logps', 'l2_grad_sum_logps',
                'l1_binary_logit_grads', 'l2_binary_logit_grads']:
      if eval(reg) > 0:
        log_prior += eval(reg) * getattr(self, reg)
    return log_likelihood + log_prior

  def fit(self, X, y, loss_fn=None, init=False, **kw):
    if loss_fn is None:
      loss_fn = self.loss_function(**kw)
    if len(y.shape) == 1:
      y = onehot(y, self.num_classes)
    ops = { 'xent': self.cross_entropy, 'loss': loss_fn, 'accu': self.accuracy }
    batches = train_batches([self], X, y, **kw)
    with tf.Session() as sess:
      if init: self.init(sess)
      minimize(sess, loss_fn, batches, ops, **kw)
      self.vals = [v.eval() for v in self.vars]

  @classmethod
  def train_diverse_models(cls, n, X, y, **kw):
    if len(y.shape) == 1:
      y = onehot(y)
    return train_diverse_models(cls, n, X, y, **kw)

  def recompute_vars(self):
    self.vars = tf.get_default_graph().get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

  def init(self, sess):
    if self.vals is None:
      sess.run(tf.global_variables_initializer())
    else:
      for var, val in zip(self.vars, self.vals):
        sess.run(var.assign(val))

  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.vals, f)

  def load(self, filename):
    with open(filename, 'rb') as f:
      self.vals = pickle.load(f)

def isint(x):
  return isinstance(x, (int, np.int32, np.int64))

def onehot(Y, K=None):
  if K is None:
    K = np.unique(Y)
  elif isint(K):
    K = list(range(K))
  data = np.array([[y == k for k in K] for y in Y]).astype(int)
  return data

def minibatch_indexes(lenX, batch_size=256, num_epochs=50, **kw):
  n = int(np.ceil(lenX / batch_size))
  for epoch in range(num_epochs):
    for batch in range(n):
      i = epoch*n + batch
      yield i, epoch, slice((i%n)*batch_size, ((i%n)+1)*batch_size)

def train_feed(idx, models, **kw):
  feed = {}
  for m in models:
    feed[m.is_train] = True
    for dictionary in [kw, kw.get('feed_dict', {})]:
      for key, val in six.iteritems(dictionary):
        attr = getattr(m, key) if isinstance(key, str) and hasattr(m, key) else key
        if type(attr) == type(m.X):
          if len(attr.shape) > 1:
            if attr.shape[0].value is None:
              feed[attr] = val[idx]
  return feed

def train_batches(models, X, y, **kw):
  for i, epoch, idx in minibatch_indexes(len(X), **kw):
    yield i, epoch, train_feed(idx, models, X=X, y=y, **kw)

def minimize(sess, loss_fn, batches, operations={}, learning_rate=0.001, print_every=None, **kw):
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_fn)
  op_keys = sorted(list(operations.keys()))
  ops = [train_op] + [operations[k] for k in op_keys]
  t = time.time()
  sess.run(tf.global_variables_initializer())
  for i, epoch, batch in batches:
    results = sess.run(ops, feed_dict=batch)
    if print_every and i % print_every == 0:
      s = 'Batch {}, epoch {}, time {:.1f}s'.format(i, epoch, time.time() - t)
      for j,k in enumerate(op_keys, 1):
        s += ', {} {:.4f}'.format(k, results[j])
      print(s)
