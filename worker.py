# coding=utf-8
# Copyright (c) 2018 The EasyTF Authors. All Rights Reserved.
# ==============================================================================

"""Worker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import json
import importlib
import traceback

from six.moves import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow import logging
from inputs.model_input import ModelInput

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
LEARNING_RATE = 0.001


def _get_inputs(model_input):
    """Return input_fn and input_hooks."""
    if isinstance(model_input, ModelInput):
        input_fn = model_input.input_fn
        input_hooks = model_input.input_hooks
        if not isinstance(input_hooks, (list, tuple)):
            input_hooks = [input_hooks]
    else:
        input_fn = model_input
        input_hooks = []
    return input_fn, input_hooks


def main(unused_argv):
    data_dir = os.path.join("apps", FLAGS.app, FLAGS.data_dir)
    model_dir = os.path.join("apps", FLAGS.app, FLAGS.model_dir)

    try:
        app_mod = importlib.import_module("apps." + FLAGS.app + ".app")
    except ImportError:
        raise ValueError("Fail to import '{}': {}".format(
                FLAGS.app, traceback.format_exc()))

    app_spec = app_mod.get_app_spec(data_dir)

    # Prepare.
    if app_spec.prepare_fn is not None:
        app_spec.prepare_fn()

    # Set RunConfig
    config = run_config.RunConfig()
    config = config.replace(model_dir=model_dir)

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(
        model_fn=app_spec.model_fn,
        config=config,
        params=model_params)

    # Training examples
    train_hooks = app_spec.train_hooks
    train_input_fn, train_input_hooks = _get_inputs(app_spec.train_input_fn)
    train_hooks += train_input_hooks
    nn.train(
        input_fn=train_input_fn,
        hooks=train_hooks,
        steps=FLAGS.train_step)

    # Evaluate test dataset.
    eval_hooks = app_spec.eval_hooks
    eval_input_fn, eval_input_hooks = _get_inputs(app_spec.eval_input_fn)
    eval_hooks += eval_input_hooks
    ev = nn.evaluate(
        input_fn=eval_input_fn,
        hooks=eval_hooks)
    for k, v in sorted(ev.items()):
        logging.info(" [Evaluated] eval data: {} = {}".format(k, v))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--app", type=str, required=True, help="Which app to run.")
  parser.add_argument(
      "--data_dir", type=str, default="data_dir", help="Path to training data.")
  parser.add_argument(
      "--model_dir", type=str, default="model_dir", help="Path to model.")
  parser.add_argument(
      "--job_name", type=str, required=False, help="Job name of current node.")
  parser.add_argument(
      "--task_index", type=int, required=False, help="Task index.")
  parser.add_argument(
      "--train_step", type=int, default=1000, help="Training step.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
