"""Module with configurations of the project.

"""

import os


PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_DIR, "data")

DB_PATH = os.path.join(DATA_DIR, "comments.db")
TEST_FROM_PATH = os.path.join(DATA_DIR, "test.from")
TEST_TO_PATH = os.path.join(DATA_DIR, "test.to")
TRAIN_FROM_PATH = os.path.join(DATA_DIR, "train.from")
TRAIN_TO_PATH = os.path.join(DATA_DIR, "train.to")
JSON_COMMENTS_PATH = "/home/bi0max/projects/tutorials/chatbot/data/RC_2015-01"
