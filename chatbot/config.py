"""Module with configurations of the project.

"""

import os


PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_DIR, "data")
MODELS_DIR = os.path.join(PROJ_DIR, "models")

DB_PATH = os.path.join(DATA_DIR, "comments.db")
TEST_FROM_PATH = os.path.join(DATA_DIR, "test.from")
TEST_TO_PATH = os.path.join(DATA_DIR, "test.to")
TRAIN_FROM_PATH = os.path.join(DATA_DIR, "train.from")
TRAIN_TO_PATH = os.path.join(DATA_DIR, "train.to")
JSON_COMMENTS_PATH = "/home/bi0max/projects/tutorials/chatbot/data/RC_2015-01"

PATHS = {
    'embedding_matrix': os.path.join(DATA_DIR, 'emb_matrix.pickle'),
    'index2word': os.path.join(DATA_DIR, 'i2w.pickle'),
    'word2index': os.path.join(DATA_DIR, 'w2i.pickle'),
    'model': os.path.join(MODELS_DIR, 'model.h5'),
    'log_dir': os.path.join(PROJ_DIR, "logs"),
    'models_dir': os.path.join(PROJ_DIR, "models"),
    'data_dir': os.path.join(PROJ_DIR, "data")
}

_glove_embedding_size = 100
_glove_model = os.path.join(DATA_DIR, "glove.6B." + str(_glove_embedding_size) + "d.txt")

PREPROCESSING_PARAMS = {
    'max_seq_length': 50,
    'max_seq_length_input': 50,
    'max_seq_length_output': 50,
    'glove_embedding_size': _glove_embedding_size,
    'glove_model': _glove_model,
    'whitelist': "abcdefghijklmnopqrstuvwxyz1234567890?!.,'",
    'pad': "<PAD>",
    'unk': "<UNK>",
    'eos': "<EOS>",
    'bos': "<BOS>",
    'vocab_size': 12000,
}
_special_tokens = [PREPROCESSING_PARAMS['pad'], PREPROCESSING_PARAMS['unk'],
                   PREPROCESSING_PARAMS['eos'], PREPROCESSING_PARAMS['bos']]
PREPROCESSING_PARAMS.update(
    {
        'special_tokens': _special_tokens,
        'full_vocab_size': PREPROCESSING_PARAMS['vocab_size'] + len(_special_tokens)
    })

HPARAMS = {
    'batch_size': 128,
    'num_epochs': 8,
    'hidden_units': 256,
    'n_batches_to_save': 1000, # each N batches Keras will save the model
    'lr_schedule': {0: 0.001, 1: 0.0005, 2: 0.00025, 3: 0.0001}
}