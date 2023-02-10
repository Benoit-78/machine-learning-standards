# -*- coding: utf-8 -*-
"""
Author: Benoît DELORME
Mail: delormebenoit211@gmail.com
Creation date: 3rd February 2023
Main objective: provides graphical tools for pylint analysis.
"""

import pandas as pd


def clean_df(my_df):
    """
    Drop useless lines & useless columns.
    Extract useful information out of raw columns.
    """
    my_df = my_df.drop(my_df.index[-1])
    my_df = my_df.drop(my_df.index[-1])
    my_df = my_df.rename(columns={my_df.columns[0]: 'Non conformity'})
    my_df = pd.DataFrame(my_df['Non conformity'].str.split(':').to_list(),
                         index=my_df.index)
    my_df = my_df.drop(0, axis=1)
    my_df = my_df.drop(2, axis=1)
    my_df = my_df.rename(columns={1: 'Line',
                                  3: 'Category',
                                  4: 'Message'})
    my_df['Line'] = my_df['Line'].astype('int')
    return my_df


def get_message_level(my_df):
    """
    Extract message level out of category column.
    """
    my_df['Level'] = my_df['Category'].astype(str).str[1]
    my_df = my_df.drop('Category', axis=1)
    level_dict = {'F': 'Fatal',
                  'E': 'Error',
                  'W': 'Warning',
                  'R': 'Refactor',
                  'C': 'Convention',
                  'I': 'Information'}
    my_df['Level'] = my_df['Level'].replace(level_dict)
    return my_df


def get_clean_messages(my_df):
    """
    Remove parasite characters out of message column.
    """
    my_df = my_df.replace('\'', '', regex=True)
    my_df = my_df.replace('\\\\', '', regex=True)
    my_df = my_df.replace(r'\.', '', regex=True)
    return my_df


def get_category(my_df):
    """
    Extract category from message column.
    """
    invert_msg = [message[::-1][1:] for message in my_df['Message']]
    my_df['Invert_message'] = invert_msg
    cat_df = pd.DataFrame(my_df['Invert_message'].str.split(r'(').to_list(),
                          index=my_df.index)
    cat_df = cat_df[cat_df.columns[0]]
    categories = [invert_category[::-1] for invert_category in cat_df]
    my_df['Category'] = categories
    my_df = my_df.drop('Invert_message', axis=1)
    return my_df


def get_clean_message(my_df):
    """
    Remove the category part from the message.
    """
    my_df['Message'] = [a.replace(b, '').strip()
                        for a, b in zip(my_df['Message'],
                                        my_df['Category'])]
    my_df['Message'] = my_df['Message'].str[:-2]
    return my_df
