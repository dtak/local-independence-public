# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave

from beta_vae import VAE
from dsprites import DataManager

"""
Code for training a beta-VAE as described in https://arxiv.org/abs/1804.03599.
Copied from https://github.com/miyosuda/disentangled_vae with very few modifications!
"""

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
                          "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "beta_vae_checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")

flags = tf.app.flags.FLAGS

def train(sess,
          model,
          manager,
          saver):

  summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
  
  n_samples = manager.sample_size

  reconstruct_check_images = manager.get_random_images(10)

  indices = list(range(n_samples))

  step = 0
  
  # Training cycle
  for epoch in range(flags.epoch_size):
    print(epoch)
    # Shuffle image indices
    random.shuffle(indices)
    
    avg_cost = 0.0
    total_batch = n_samples // flags.batch_size
    
    # Loop over all batches
    for i in range(total_batch):
      # Generate image batch
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)
      
      # Fit training using batch data
      reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, step)
      summary_writer.add_summary(summary_str, step)
      step += 1

    # Image reconstruction check
    reconstruct_check(sess, model, reconstruct_check_images)

    # Disentangle check
    disentangle_check(sess, model, manager)

    # Save checkpoint
    saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = step)

def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  return saver


def main(argv):
  manager = DataManager()
  manager.load()

  sess = tf.Session()

  model = VAE(gamma=flags.gamma,
              capacity_limit=flags.capacity_limit,
              capacity_change_duration=flags.capacity_change_duration,
              learning_rate=flags.learning_rate)

  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess)

  train(sess, model, manager, saver)

if __name__ == '__main__':
  tf.app.run()
