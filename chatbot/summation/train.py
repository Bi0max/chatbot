"""Idea comes from here:
https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py

"""

import os
import pickle

import numpy as np
from manager import Manager
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

from chatbot.config import PATHS
from chatbot.train import limit_gpu_memory, create_model_attention

vocab = [str(i) for i in range(10)]
vocab_size = len(vocab)
index2word = {i: w for i, w in zip(range(vocab_size), vocab)}
word2index = {w: i for i, w in zip(range(vocab_size), vocab)}
n_timesteps_input = 6
n_timesteps_output = 4
PREPROCESSING_PARAMS = {
    'max_seq_length_input': n_timesteps_input,
    'max_seq_length_output': n_timesteps_output,
    'full_vocab_size': vocab_size
}
HPARAMS = {
    'hidden_units': 512,
    'batch_size': 128,
    'num_epochs': 100
}


manager = Manager()


def text_to_input(input_text):
    batch_size = 1
    indices_input = np.array(list(input_text)).astype(int)
    print(indices_input)
    onehot_inputs = np.zeros((batch_size, n_timesteps_input, vocab_size))
    onehot_inputs[0, np.arange(n_timesteps_input), indices_input] = 1
    print(onehot_inputs)
    return onehot_inputs


def create_batch(batch_size, hidden_units):
    # create random samples
    numbers = np.random.randint(100, 1000, batch_size * 2)
    numbers1 = numbers[:batch_size]
    numbers2 = numbers[batch_size:]
    numbers_result = numbers1 + numbers2
    # inputs
    input_str = [str(i) + str(j) for i, j in zip(numbers1, numbers2)]
    indices_input = [np.array(list(s)).astype(int) for s in input_str]
    onehot_inputs = np.zeros((batch_size, n_timesteps_input, vocab_size))
    for i in range(batch_size):
        onehot_inputs[i, np.arange(n_timesteps_input), indices_input[i]] = 1
    # outputs
    output_str = [f'{i:0{n_timesteps_output}}' for i in numbers_result]
    indices_output = [np.array(list(s)).astype(int) for s in output_str]
    onehot_outputs = np.zeros((batch_size, n_timesteps_output, vocab_size))
    for i in range(batch_size):
        onehot_outputs[i, np.arange(n_timesteps_output), indices_output[i]] = 1
    output_list = []
    for t in range(onehot_outputs.shape[1]):
        output_list.append(onehot_outputs[:, t, :])
    s0 = np.zeros((batch_size, hidden_units))
    c0 = np.zeros((batch_size, hidden_units))
    return [onehot_inputs, s0, c0], output_list


@manager.command()
def create_dataset(size):
    size = int(size)
    dataset = create_batch(size, HPARAMS['hidden_units'])
    path = os.path.join(PATHS['data_dir'], 'summation_data.pickle')
    pickle.dump(dataset, open(path, "wb"))


def generate_batch(batch_size, hidden_units):
    while True:
        yield create_batch(batch_size, hidden_units)



@manager.command()
def start_attention_gen(name):
    limit_gpu_memory()

    # create model
    print("Creating model...")
    model = create_model_attention(PREPROCESSING_PARAMS, HPARAMS, for_inference=False, use_embedding=False)
    print("Compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Callbacks
    tensorboard = TensorBoard(log_dir="{}/{}".format(PATHS['log_dir'], name), write_grads=True,
                              write_graph=True, write_images=True)
    file_path = os.path.join(PATHS['models_dir'], f"{name}")
    checkpoint = ModelCheckpoint(file_path + "-{epoch:02d}-{val_loss:.2f}.h5", verbose=1, period=10)

    train_gen = generate_batch(HPARAMS['batch_size'], HPARAMS['hidden_units'])
    test_gen = generate_batch(HPARAMS['batch_size'], HPARAMS['hidden_units'])

    train_num_batches = 1000
    test_num_batches = 100

    model.fit_generator(
        generator=train_gen, steps_per_epoch=train_num_batches,
        epochs=HPARAMS['num_epochs'],
        verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
        callbacks=[tensorboard, checkpoint],
    )


@manager.command()
def start_attention(name, file_name=None):
    limit_gpu_memory()
    # read data
    path = os.path.join(PATHS['data_dir'], 'summation_data.pickle')
    dataset = pickle.load(open(path, "rb"))

    # create model
    print("Creating model...")
    model = create_model_attention(PREPROCESSING_PARAMS, HPARAMS, for_inference=False, use_embedding=False,
                                   bilstm=True)
    print("Compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if file_name is not None:
        print("Loading model's weights...")
        model.load_weights(os.path.join(PATHS['models_dir'], file_name))

    # Callbacks
    tensorboard = TensorBoard(log_dir="{}/{}".format(PATHS['log_dir'], name), write_grads=True,
                              write_graph=True, write_images=True)
    file_path = os.path.join(PATHS['models_dir'], f"{name}")
    checkpoint = ModelCheckpoint(file_path + "-{epoch:02d}-{val_loss:.2f}.h5", verbose=1, period=10)

    model.fit(x=dataset[0], y=dataset[1], batch_size=128, validation_split=0.05, shuffle=True,
        epochs=HPARAMS['num_epochs'],
        verbose=1,
        callbacks=[tensorboard, checkpoint],
    )


def reply_attention(input_text, model, preprocessing_params, hparams):
    encoder_input = text_to_input(input_text)
    s0 = np.zeros((1, hparams['hidden_units']))
    c0 = np.zeros((1, hparams['hidden_units']))

    prediction = model.predict([encoder_input, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [index2word[int(i)] for i in prediction]
    return str.join(' ', output)


@manager.command()
def do_inference_attention(file_name=None):
    limit_gpu_memory()
    if file_name is not None:
        path = os.path.join(PATHS['models_dir'], file_name)
    else:
        path = PATHS["model"]

    model = create_model_attention(PREPROCESSING_PARAMS, HPARAMS, use_embedding=False, for_inference=False)
    model.load_weights(path)

    finished = False
    while not finished:
        text = input("Input text (to finish enter 'f'): ")
        if text == 'f':
            finished = True
            continue
        replies = reply_attention(text, model, PREPROCESSING_PARAMS, HPARAMS)
        # replies_without_unk = [r for r in replies if PREPROCESSING_PARAMS['unk'] not in r]
        # print(len(replies_without_unk))
        # for r in replies_without_unk:
        print(replies)


if __name__ == '__main__':
    manager.main()
    #i = 0
    #for g in generate_batch(3, 3):
    #    i += 1
    #    print(g)
    #    if i > 1:
    #        break