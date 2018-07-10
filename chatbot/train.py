import os
import pickle

import numpy as np
import tensorflow as tf
from manager import Manager
from keras.layers import Embedding, LSTM, Input, Dense
from keras.models import Model
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler
from keras import backend as K
from sklearn.model_selection import train_test_split

from chatbot.config import *


manager = Manager()


def generate_batch(tokenized_from, tokenized_to, word2index, preprocessing_params, hparams):
    n_samples = len(tokenized_from)
    # must have same amount of comments/answers (from/to)
    assert n_samples == len(tokenized_to)
    num_batches = n_samples // hparams['batch_size']
    while True:
        for batch_i in range(0, num_batches):
            start = batch_i * hparams['batch_size']
            end = (batch_i + 1) * hparams['batch_size']
            yield create_model_input_output(
                tokenized_from[start: end], tokenized_to[start: end], word2index, preprocessing_params)


def create_model_input_output(tokenized_from, tokenized_to, word2index, preprocessing_params):
    # TODO throw away samples with length > max_seq_length
    unk_index, eos_index, bos_index = [word2index[w] for w in [preprocessing_params['unk'],
                                                               preprocessing_params['eos'],
                                                               preprocessing_params['bos']]]
    n_samples = len(tokenized_from)
    max_seq_length = preprocessing_params['max_seq_length']
    encoder_input = np.zeros((n_samples, max_seq_length), dtype=int)
    decoder_input = np.zeros((n_samples, max_seq_length), dtype=int)
    decoder_output = np.zeros((n_samples, max_seq_length), dtype=int)
    decoder_output_oh = np.zeros((n_samples, max_seq_length, preprocessing_params['full_vocab_size']), dtype=float)

    for i, sentence_from, sentence_to in zip(range(n_samples), tokenized_from, tokenized_to):
        if (i + 1) % 10000 == 0:
            print(f"Creating input/output for model {i + 1}/{n_samples} done.")
        encoder_input[i, :len(sentence_from)] = [
            word2index.get(w, unk_index) for w in sentence_from[:max_seq_length]]
        sentence_to_indexed = [word2index.get(w, unk_index) for w in sentence_to[:max_seq_length - 1]]
        decoder_input[i, :len(sentence_to) + 1] = [bos_index] + sentence_to_indexed
        decoder_output[i, :len(sentence_to) + 1] = sentence_to_indexed + [eos_index]
        decoder_output_oh[i, np.arange(max_seq_length), decoder_output[i]] = 1
    return [encoder_input, decoder_input], decoder_output_oh


def create_model(preprocessing_params, hparams, embedding_matrix=None, for_inference=False):
    max_seq_length = preprocessing_params['max_seq_length']
    vocab_size = preprocessing_params['full_vocab_size']
    emb_size = preprocessing_params['glove_embedding_size']
    hidden_units = hparams['hidden_units']
    # layers declaration
    if embedding_matrix is not None:
        # set pretrained weights
        shared_embedding = Embedding(
            input_dim=vocab_size, output_dim=emb_size, input_length=max_seq_length, mask_zero=True,
            weights=[embedding_matrix], name='shared_embedding')
    else:
        shared_embedding = Embedding(
            input_dim=vocab_size, output_dim=emb_size, input_length=max_seq_length, mask_zero=True,
            name='shared_embedding')

    encoder_lstm = LSTM(units=hidden_units, return_state=True, name='encoder_lstm')

    decoder_lstm = LSTM(units=hidden_units, return_state=True, return_sequences=True, name='decoder_lstm')

    # model structure
    encoder_inputs = Input(shape=(max_seq_length,), name='encoder_inputs')
    encoder_embeddings = shared_embedding(encoder_inputs)
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embeddings)
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder_inputs = Input(shape=(max_seq_length,), name='decoder_inputs')
    decoder_embeddings = shared_embedding(decoder_inputs)
    decoder_outputs_lstm, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)
    decoder_dense = Dense(units=vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs_lstm)

    # full model (for training)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    if not for_inference:
        return model

    # separately encoder model and decoder model (for prediction)
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_inputs = [Input(shape=(hidden_units,)), Input(shape=(hidden_units,))]
    decoder_outputs_lstm, decoder_state_h, decoder_state_c = decoder_lstm(decoder_embeddings,
                                                                     initial_state=decoder_state_inputs)
    decoder_outputs = decoder_dense(decoder_outputs_lstm)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


class WeightsSaver(Callback):
    def __init__(self, model, n_batches_to_save, file_path):
        self.model = model
        self.n_batches_to_save = n_batches_to_save
        self.batch = 0
        self.file_path = file_path

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.n_batches_to_save == 0:
            file_name = f"{self.file_path}-{self.batch}.h5"
            self.model.save(file_name)
        self.batch += 1


class TensorBoardPerBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()

        self.seen += self.batch_size


def schedule(epoch, lr):
    if epoch in HPARAMS['lr_schedule']:
        new_lr = HPARAMS['lr_schedule'][epoch]
    else:
        new_lr = lr
    print(f"Current epoch: {epoch}. Current lr: {lr}. New lr: {new_lr}")
    return new_lr


def limit_gpu_memory():
    # limit amount of GPU memory used by tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)


@manager.command()
def start(name):
    limit_gpu_memory()

    # read files with samples
    print("Opening files...")
    path = os.path.join(DATA_DIR, "tokenized_from.pickle")
    tokenized_from = pickle.load(open(path, "rb"))
    path = os.path.join(DATA_DIR, "tokenized_to.pickle")
    tokenized_to = pickle.load(open(path, "rb"))

    # read matrix, index
    embedding_matrix = pickle.load(open(PATHS["embedding_matrix"], "rb"))
    word2index = pickle.load(open(PATHS["word2index"], "rb"))

    # create model
    print("Creating model...")
    model = create_model(PREPROCESSING_PARAMS, HPARAMS, embedding_matrix, for_inference=False)
    print("Compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Callbacks
    tensorboard = TensorBoardPerBatch(log_dir="{}/{}".format(PATHS['log_dir'], name), write_grads=True,
                                      write_graph=True, write_images=True)
    file_path = os.path.join(PATHS['models_dir'], f"{name}")
    weights_saver = WeightsSaver(model, HPARAMS['n_batches_to_save'], file_path)
    lr_scheduler = LearningRateScheduler(schedule)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(tokenized_from, tokenized_to, test_size=0.05, random_state=42)

    print(f"N training samples: {len(Xtrain)}")
    print(f"N Test samples: {len(Xtest)}")

    train_gen = generate_batch(Xtrain, Ytrain, word2index, PREPROCESSING_PARAMS, HPARAMS)
    test_gen = generate_batch(Xtest, Ytest, word2index, PREPROCESSING_PARAMS, HPARAMS)

    train_num_batches = len(Xtrain) // HPARAMS['batch_size']
    test_num_batches = len(Xtest) // HPARAMS['batch_size']

    model.fit_generator(
        generator=train_gen, steps_per_epoch=train_num_batches,
        epochs=HPARAMS['num_epochs'],
        verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
        callbacks=[tensorboard, weights_saver, lr_scheduler],
    )


if __name__ == '__main__':
    manager.main()

