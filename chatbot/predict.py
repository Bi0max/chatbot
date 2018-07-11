import pickle
from copy import deepcopy

import numpy as np
from keras.models import load_model
from manager import Manager

from chatbot.config import *
from chatbot.tokenize_data import initial_preprocess
from chatbot.train import create_model_input_output, create_model


manager = Manager()


def cosine_similarity(u, v):
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    similarity = dot / norm_u / norm_v
    return similarity


# return indices of top 3 values
def argmax_b(arr, b):
    return arr.argsort()[-b:][::-1]


# return indices of top 3 values in 2d array
def argmax_b_2d(arr, b):
    arr_flat = arr.flatten(order='F')
    indices_flat = argmax_b(arr_flat, b)
    indices = np.zeros((b, 2), dtype=int)
    n_rows, n_columns = arr.shape
    for i, idx in enumerate(indices_flat):
        indices[i, 0] = idx % n_rows
        indices[i, 1] = idx // n_rows
    return indices


def reply(input_text, encoder_model, decoder_model, word2index, index2word, preprocessing_params, b=1):
    words = initial_preprocess(input_text, preprocessing_params['whitelist'])
    encoder_input, _ = create_model_input_output(
        [words], [[]], word2index, preprocessing_params)[0]

    states_value = encoder_model.predict(encoder_input)
    target_words = []
    terminated = False

    while not terminated:
        _, decoder_input = create_model_input_output(
            [[]], [target_words], word2index, preprocessing_params)[0]

        decoder_outputs, h, c = decoder_model.predict([decoder_input] + states_value)
        states_value = [h, c]

        last_decoder_outputs = decoder_outputs[0, -1, :]

        sample_token_idx = np.argmax(last_decoder_outputs)
        sample_word = index2word[sample_token_idx]

        b_top_idx = argmax_b(last_decoder_outputs, b)
        print(f"{len(target_words) + 1} word: ")
        for i in range(b):
            print(f"  {i + 1} most popular: {index2word[b_top_idx[i]]}")
        print()

        if sample_word != preprocessing_params['bos'] and sample_word != preprocessing_params['eos']:
            target_words.append(sample_word)

        if sample_word == preprocessing_params['eos'] or len(target_words) >= preprocessing_params[
            'max_seq_length'] * 2:
            terminated = True

    return str.join(' ', target_words)


def reply_beam(input_text, encoder_model, decoder_model, word2index, index2word, preprocessing_params, b=1):
    words = initial_preprocess(input_text, preprocessing_params['whitelist'])
    print(f"Your text: {words}")
    encoder_input, _ = create_model_input_output(
        [words], [[]], word2index, preprocessing_params)[0]

    states_value = encoder_model.predict(encoder_input)
    probable_predictions = []
    probabilities = np.ones(b)
    finished_predictions = []
    first_run = True
    terminated = False

    while not terminated:
        if first_run:
            print("First run")
            _, decoder_input = create_model_input_output(
                [[]], [[]], word2index, preprocessing_params)[0]

            decoder_outputs, h, c = decoder_model.predict([decoder_input] + states_value)
            states_value = [h, c]

            last_decoder_outputs = decoder_outputs[0, -1, :]

            sample_token_idx = np.argmax(last_decoder_outputs)

            b_top_idx = argmax_b(last_decoder_outputs, b)
            for i in range(b):
                top_word = index2word[b_top_idx[i]]
                probable_predictions.append([top_word])
                # print(f"  {i + 1} most popular: {top_word}")
            probabilities *= last_decoder_outputs[b_top_idx]
            # print()
            first_run = False
        else:
            # print(probable_predictions)
            # print(probabilities)
            last_decoder_outputs_array = np.full(
                shape=(preprocessing_params['full_vocab_size'], b), fill_value=-np.inf)

            for i in range(b):
                if i not in finished_predictions:
                    # print(f"Run for {probable_predictions[i]}")
                    if preprocessing_params['eos'] in probable_predictions[i] or \
                            len(probable_predictions[i]) >= preprocessing_params['max_seq_length']:
                        # print(f"Beam {i} finished search.")
                        finished_predictions.append(i)
                        continue

                    _, decoder_input = create_model_input_output(
                        [[]], [probable_predictions[i]], word2index, preprocessing_params)[0]

                    decoder_outputs, h, c = decoder_model.predict([decoder_input] + states_value)
                    states_value = [h, c]
                    last_decoder_outputs_array[:, i] = decoder_outputs[0, -1, :] * probabilities[i]
            # print(f"Finished predictions: {finished_predictions}")
            if len(finished_predictions) == b:
                terminated = True
                continue
            # print(last_decoder_outputs_array.shape)
            b_left = b - len(finished_predictions)
            indices_top = argmax_b_2d(last_decoder_outputs_array, b_left)
            # print(indices_top)
            i = 0
            probable_predictions_previous = deepcopy(probable_predictions)
            for beam in range(b):
                if beam not in finished_predictions:
                    chosen_beam = indices_top[i, 1]
                    chosen_word = indices_top[i, 0]
                    probabilities[beam] = last_decoder_outputs_array[chosen_word, chosen_beam]
                    probable_predictions[beam] = probable_predictions_previous[chosen_beam] + [index2word[chosen_word]]
                    i += 1
    return probable_predictions


@manager.command()
def do_inference(file_name=None):
    if file_name is not None:
        path = os.path.join(PATHS['models_dir'], file_name)
    else:
        path = PATHS["model"]
    # read matrix, index
    embedding_matrix = pickle.load(open(PATHS["embedding_matrix"], "rb"))
    word2index = pickle.load(open(PATHS["word2index"], "rb"))
    index2word = pickle.load(open(PATHS["index2word"], "rb"))

    model, encoder_model, decoder_model = create_model(PREPROCESSING_PARAMS, HPARAMS, for_inference=True)
    model.load_weights(path)

    finished = False
    while not finished:
        text = input("Input text (to finish enter 'f'): ")
        if text == "" or text is None:
            text = "How are you doing?"
        if text == 'f':
            finished = True
            continue
        replies = reply_beam(text, encoder_model, decoder_model, word2index, index2word, PREPROCESSING_PARAMS, b=30)
        replies_without_unk = [r for r in replies if PREPROCESSING_PARAMS['unk'] not in r]
        print(len(replies_without_unk))
        for r in replies_without_unk:
            print(r)


if __name__ == '__main__':
    manager.main()