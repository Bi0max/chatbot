"""Module to create database from downloaded comments in JSON format.

"""

import json
import sqlite3
from datetime import datetime

from chatbot.config import *

START_ROW = 0
CLEANUP_ROW = 1000000
PRINT_STATUS_ROW = 100000


def create_table(cursor):
    sql = "DROP TABLE IF EXISTS parent_reply;"
    cursor.execute(sql)
    sql = """
        CREATE TABLE IF NOT EXISTS parent_reply(
        parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, 
        comment TEXT, subreddit TEXT, unix INT, score INT);
    """
    cursor.execute(sql)


def format_data(data):
    new_line = " newlinechar "
    data = data.replace("\n", new_line).replace("\r", new_line).replace('"', "'")
    return data


def fetch_one_row(cursor, sql):
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        if result is not None:
            return result[0]
        else:
            return None
    except Exception as e:
        print("fetch_one_row ", e)
        return None


def find_parent(cursor, parent_id):
    sql = f"SELECT comment FROM parent_reply WHERE comment_id = '{parent_id}' LIMIT 1;"
    return fetch_one_row(cursor, sql)


def find_existing_score(cursor, parent_id):
    sql = f"SELECT score FROM parent_reply WHERE parent_id = '{parent_id}' LIMIT 1;"
    return fetch_one_row(cursor, sql)


def is_acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else:
        return True


def transaction_builder(cursor, connection, sql, sql_transactions):
    sql_transactions.append(sql)
    if len(sql_transactions) > 1000:
        cursor.execute('BEGIN TRANSACTION')
        for s in sql_transactions:
            try:
                cursor.execute(s)
            except Exception as e:
                pass
                # print("transaction_builder ", e)
                # print("S:", s)
        connection.commit()
        sql_transactions = []
    return sql_transactions


def sql_insert_replace_comment(comment_id, parent_id, parent_body, comment_body, subreddit, created_utc, score):
    if parent_body is not None:
        parent_body_sql = f"""parent = "{parent_body}", """
    else:
        parent_body_sql = ""
    sql = """
        UPDATE parent_reply 
        SET parent_id = "{}", comment_id = "{}", {}comment = "{}", subreddit = "{}", unix = {}, score = {} 
        WHERE parent_id = "{}";
        """.format(parent_id, comment_id, parent_body_sql, comment_body, subreddit, int(created_utc), score, parent_id)
    return sql


def sql_insert_has_parent(comment_id, parent_id, parent_body, comment_body, subreddit, created_utc, score):
    sql = """
        INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score)
        VALUES ("{}","{}","{}","{}","{}",{},{});
    """.format(parent_id, comment_id, parent_body, comment_body, subreddit, int(created_utc), score)
    return sql


def sql_insert_no_parent(comment_id, parent_id, comment_body, subreddit, created_utc, score):
    sql = """
        INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score)
        VALUES ("{}","{}","{}","{}",{},{});
    """.format(parent_id, comment_id, comment_body, subreddit, int(created_utc), score)
    return sql


def clean_up(cursor, connection):
    print("Cleaning up!")
    sql = "DELETE FROM parent_reply WHERE parent IS NULL"
    cursor.execute(sql)
    connection.commit()
    cursor.execute("VACUUM")
    connection.commit()


def construct_sql(cursor, row, paired_rows):
    row = json.loads(row)
    comment_id = row['name']
    parent_id = row['parent_id']
    comment_body = format_data(row['body'])
    created_utc = row['created_utc']
    score = row['score']
    subreddit = row['subreddit']
    parent_body = find_parent(cursor, parent_id)

    sql = None
    # filter some "useless" comments
    existing_comment_score = find_existing_score(cursor, parent_id)
    if existing_comment_score is not None:
        if score > existing_comment_score:
            if is_acceptable(comment_body):
                sql = sql_insert_replace_comment(
                    comment_id, parent_id, parent_body, comment_body, subreddit, created_utc, score)

    else:
        if is_acceptable(comment_body):
            if parent_body is not None:
                if score >= 2:
                    sql = sql_insert_has_parent(
                        comment_id, parent_id, parent_body, comment_body, subreddit, created_utc, score)
                    paired_rows += 1
            else:
                sql = sql_insert_no_parent(comment_id, parent_id, comment_body, subreddit, created_utc, score)
    return sql, paired_rows


def main():
    sql_transactions = []
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    create_table(cursor)
    row_counter = 0
    paired_rows = 0

    with open(JSON_COMMENTS_PATH, buffering=1000) as f:
        for row in f:
            row_counter += 1
            if row_counter > START_ROW:
                try:
                    sql, paired_rows = construct_sql(cursor, row, paired_rows)
                    if sql is not None:
                        sql_transactions = transaction_builder(cursor, connection, sql, sql_transactions)
                except Exception as e:
                    print(str(e))

                if row_counter % PRINT_STATUS_ROW == 0:
                    print(f"Total rows read: {row_counter}, Paired rows: {paired_rows}, Time: {str(datetime.now())}")

                if row_counter % CLEANUP_ROW == 0:
                    clean_up(cursor, connection)


if __name__ == '__main__':
    main()
