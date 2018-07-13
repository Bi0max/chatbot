import os
import pickle

import numpy as np
import tensorflow as tf
from manager import Manager
from keras.layers import Embedding, LSTM, Input, Dense, RepeatVector, Concatenate, Activation, Dot, Bidirectional
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


def generate_batch_attention(tokenized_from, tokenized_to, word2index, preprocessing_params, hparams):
    n_samples = len(tokenized_from)
    # must have same amount of comments/answers (from/to)
    assert n_samples == len(tokenized_to)
    num_batches = n_samples // hparams['batch_size']
    while True:
        for batch_i in range(0, num_batches):
            start = batch_i * hparams['batch_size']
            end = (batch_i + 1) * hparams['batch_size']
            edi, do = create_model_input_output(
                tokenized_from[start: end], tokenized_to[start: end], word2index, preprocessing_params)
            do_list = []
            for t in range(do.shape[1]):
                do_list.append(do[:, t, :])
            s0 = np.zeros((hparams['batch_size'], hparams['hidden_units']))
            c0 = np.zeros((hparams['batch_size'], hparams['hidden_units']))
            yield [edi[0], s0, c0], do_list


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


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def create_model_attention(preprocessing_params, hparams, embedding_matrix=None, for_inference=False):
    max_seq_length = preprocessing_params['max_seq_length']
    max_seq_input_length = max_seq_output_length = max_seq_length
    vocab_size = preprocessing_params['full_vocab_size']
    emb_size = preprocessing_params['glove_embedding_size']
    hidden_units = hparams['hidden_units']
    hidden_units_encoder = hidden_units // 2
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

    encoder_lstm = Bidirectional(LSTM(units=hidden_units_encoder, return_sequences=True, name='encoder_lstm'))

    decoder_lstm = LSTM(units=hidden_units, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(units=vocab_size, activation=softmax, name='decoder_dense')

    repeator = RepeatVector(max_seq_input_length)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    # We are using a custom softmax(axis = 1)
    activator = Activation(softmax, name='attention_weights')
    dotor = Dot(axes=1)

    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.

        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a),
        where m is batch_size, Tx is input_sequence_length.
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
        """
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis
        concat = concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas"
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
        context = dotor([alphas, a])

        return context

    # model structure

    # Inputs
    encoder_inputs = Input(shape=(max_seq_input_length,), name='encoder_inputs')
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    s0 = Input(shape=(hidden_units,), name='s0')
    c0 = Input(shape=(hidden_units,), name='c0')
    s = s0
    c = c0

    # Step 1: Define pre-attention Bi-LSTM
    encoder_embeddings = shared_embedding(encoder_inputs)
    encoder_outputs = encoder_lstm(encoder_embeddings)

    # Initialize empty list of outputs
    outputs = []
    
    # Step 2: Iterate for Ty steps
    for t in range(max_seq_output_length):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(encoder_outputs, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state]
        s, _, c = decoder_lstm(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = decoder_dense(s)

        # Step 2.D: Append "out" to the "outputs" list
        outputs.append(out)

    # decoder_inputs = Input(shape=(max_seq_output_length,), name='decoder_inputs')
    # decoder_embeddings = shared_embedding(decoder_inputs)
    # decoder_outputs_lstm, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)
    # decoder_outputs = decoder_dense(decoder_outputs_lstm)

    # full model (for training)
    model = Model([encoder_inputs, s0, c0], outputs)
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
    # discussion on per batch TensorBoard: https://github.com/keras-team/keras/issues/6692
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.counter += 1

        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {model_input: embeddings_data[idx][batch]
                                     for idx, model_input in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir, 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.counter)
        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)


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


@manager.command()
def start_attention(name):
    limit_gpu_memory()

    # read files with samples
    print("Opening files...")
    path = os.path.join(DATA_DIR, "tokenized_from_clean.pickle")
    tokenized_from = pickle.load(open(path, "rb"))
    path = os.path.join(DATA_DIR, "tokenized_to_clean.pickle")
    tokenized_to = pickle.load(open(path, "rb"))

    # read matrix, index
    embedding_matrix = pickle.load(open(PATHS["embedding_matrix"], "rb"))
    word2index = pickle.load(open(PATHS["word2index"], "rb"))

    # create model
    print("Creating model...")
    model = create_model_attention(PREPROCESSING_PARAMS, HPARAMS, embedding_matrix, for_inference=False)
    print("Compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Callbacks
    tensorboard = TensorBoardPerBatch(log_dir="{}/{}".format(PATHS['log_dir'], name), write_grads=True,
                                      write_graph=True, write_images=True)
    file_path = os.path.join(PATHS['models_dir'], f"{name}")
    weights_saver = WeightsSaver(model, HPARAMS['n_batches_to_save'], file_path)
    lr_scheduler = LearningRateScheduler(schedule)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(tokenized_from, tokenized_to, test_size=0.05, random_state=42)

    print(f"N training samples: {len(Xtrain)}")
    print(f"N Test samples: {len(Xtest)}")

    train_gen = generate_batch_attention(Xtrain, Ytrain, word2index, PREPROCESSING_PARAMS, HPARAMS)
    test_gen = generate_batch_attention(Xtest, Ytest, word2index, PREPROCESSING_PARAMS, HPARAMS)

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

