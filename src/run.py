#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf

try:
    import ujson as json
except:
    import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
import yaml

import jedi
import utils

utils.handle_flags()


def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    utils.limit_gpu_memory_growth()
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)

    cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader)
    data_prefix = '{}/data.{}.K{}.L{}'.format(
        cfg['path_data'], FLAGS.cv, FLAGS.K, FLAGS.L)
    path_pred = '{}/pred.{}.K{}.L{}'.format(
        cfg['path_pred'], FLAGS.cv, FLAGS.K, FLAGS.L)

    # train_data = utils.Data(data_prefix + '.train', FLAGS)
    # test_data = utils.Data(data_prefix + '.test', FLAGS)

    # read all data
    train_data_all = utils.Data.read_file(data_prefix + '.train', FLAGS.L)
    test_data_all = utils.Data.read_file(data_prefix + '.test', FLAGS.L)

    # split train and validation set
    n_all = len(train_data_all)
    n_validation = int(n_all * 0.1)
    train_idx = np.random.choice(n_all, size=n_validation)

    train_data = utils.Data(
        [train_data_all[i] for i in train_idx],
        FLAGS
    )
    validation_data = utils.Data(
        [train_data_all[i] for i in range(n_all) if i not in train_idx],
        FLAGS
    )
    test_data = utils.Data(test_data_all, FLAGS)

    model = jedi.JEDI(FLAGS)

    # Optimization settings.
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=FLAGS.learning_rate, amsgrad=True)

    # Logging metric settings.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # TF Functions.
    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            predictions = model(
                data['acceptors'],
                data['donors'],
                data['length_a'],
                data['length_d'])
            loss = loss_object(data['label'], predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        return predictions

    @tf.function
    def valid_step(data, loss_metric):
        predictions = model(
            data['acceptors'],
            data['donors'],
            data['length_a'],
            data['length_d'])
        loss = loss_object(data['label'], predictions)

        loss_metric(loss)
        return predictions

    def eval(y_true, y_pred):
        y_true = [1 if x > 0.5 else -1 for x in y_true]
        y_pred = [1 if x > 0.5 else -1 for x in y_pred]
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        sen = recall_score(y_true, y_pred, pos_label=1)
        spe = recall_score(y_true, y_pred, pos_label=-1)
        return acc, pre, f1, mcc, sen, spe

    # Training and Evaluating.
    train_metrics = []
    validation_metrics = []

    for epoch in range(FLAGS.num_epochs):
        # Reset metrics.
        train_loss.reset_states()
        validation_loss.reset_states()
        # Training.
        num_batches = (len(train_data.records) + FLAGS.batch_size - 1)
        num_batches = num_batches // FLAGS.batch_size
        preds, lbls = [], []
        for data in tqdm(
                train_data.batch_iter(),
                desc='Training',
                total=num_batches
        ):
            preds.extend(list(train_step(data)))
            lbls.extend(list(data['label']))
        acc, pre, f1, mcc, sen, spe = eval(lbls, preds)
        loss = train_loss.result()
        tmpl = 'Epoch {} (CV={}, K={}, L={})\n' + \
               'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
        print(tmpl.format(
            epoch + 1, FLAGS.cv, FLAGS.K, FLAGS.L, loss,
            acc, pre, f1, mcc, sen, spe),
            file=sys.stderr)
        train_metrics.append((loss, acc, pre, f1, mcc, sen, spe))

        preds, lbls = [], []
        for data in tqdm(
                validation_data.batch_iter(is_random=False),
                desc='Validation',
                total=num_batches
        ):
            preds.extend(list(valid_step(data, validation_loss)))
            lbls.extend(list(data['label']))
        validation_metrics.append((validation_loss.result(), *eval(lbls, preds)))

        # early stopping
        if epoch > 5:
            # compare accuracies (at index 1)
            prev_vals = [validation_metrics[i][1] for i in range(epoch - 5, epoch)]
            cur_val = validation_metrics[epoch][1]
            val_avg = np.array(prev_vals).mean()
            if np.abs(cur_val - val_avg) < 0.0001:
                break

    with open(FLAGS.train_stats, 'w') as train_out:
        train_out.write('epoch\tloss\tacc\tpre\tf1\tmcc\tsen\tspe\tdataset\n')
        for epoch, (loss, acc, pre, f1, mcc, sen, spe) in enumerate(train_metrics):
            train_out.write(
                f'{epoch + 1}\t{loss}\t{acc}\t{pre}\t{f1}\t{mcc}\t{sen}\t{spe}\ttrain\n'
            )
        for epoch, (loss, acc, pre, f1, mcc, sen, spe) in enumerate(validation_metrics):
            train_out.write(
                f'{epoch + 1}\t{loss}\t{acc}\t{pre}\t{f1}\t{mcc}\t{sen}\t{spe}\tvalidation\n'
            )

    # logging.info('Saving model to to {}.'.format(FLAGS.model))
    # tf.saved_model.save(model, FLAGS.model)
    # model.save(FLAGS.model)

    # Testing and Evaluating.
    # Reset metrics.
    test_loss.reset_states()
    # Training.
    num_batches = (len(test_data.records) + FLAGS.batch_size - 1)
    num_batches = num_batches // FLAGS.batch_size
    preds, lbls = [], []
    for data in tqdm(test_data.batch_iter(is_random=False),
                     desc='Testing', total=num_batches):
        preds.extend(list(valid_step(data, test_loss)))
        lbls.extend(list(data['label']))

    lbls = [int(x) for x in lbls]
    preds = [float(x) for x in preds]
    test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe = \
        eval(lbls, preds)

    tmpl = 'Testing (CV={}, K={}, L={})\n' + \
           'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
    print(tmpl.format(FLAGS.cv, FLAGS.K, FLAGS.L,
                      test_loss.result(),
                      test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe),
          file=sys.stderr)

    logging.info('Saving testing predictions to to {}.'.format(path_pred))
    with open(path_pred, 'w') as wp:
        json.dump(list(zip(preds, lbls)), wp)


if __name__ == '__main__':
    app.run(main)
