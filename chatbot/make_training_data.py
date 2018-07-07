"""Module to create training/test sets from a database.

"""

import sqlite3
import pandas as pd

from chatbot.config import *


def save_data(df, path, col_name):
    with open(path, 'a', encoding='utf8') as f:
        for content in df[col_name].values:
            f.write(content + "\n")


def main():
    connection = sqlite3.connect(DB_PATH)
    limit = 5000
    last_unix = 0
    cur_length = limit
    counter = 0
    test_done = False
    while cur_length == limit:
        sql = f"""
            SELECT * FROM parent_reply 
            WHERE unix > {last_unix} AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {limit}
        """
        df = pd.read_sql(sql, connection)
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        if not test_done:
            save_data(df, TEST_FROM_PATH, 'parent')
            save_data(df, TEST_TO_PATH, 'comment')
            test_done = True
        else:
            save_data(df, TRAIN_FROM_PATH, 'parent')
            save_data(df, TRAIN_TO_PATH, 'comment')
        counter += 1
        if counter % 10 == 0:
            print(counter * limit, "rows completed so far.")


if __name__ == '__main__':
    main()