# -*- coding: utf-8 -*-
"""
Author: Benoît DELORME
Mail: delormebenoit211@gmail.com
Creation date: 3rd February 2023
Main objective: provides graphical tools for pylint analysis.
"""

import pandas as pd


def clean_df(df):
    df = df.drop(df.index[-1])
    df = df.drop(df.index[-1])
    df = df.rename(columns={df.columns[0]: 'Non conformity'})
    df = pd.DataFrame(df['Non conformity'].str.split(':').to_list(),
                      index=df.index)
    df = df.drop(0, axis=1)
    df = df.drop(2, axis=1)
    df = df.rename(columns={1: 'Line',
                            3: 'Category',
                            4: 'Message'})
    df['Line'] = df['Line'].astype('int')
    return df


def get_messages(df):
    df = df.replace('\'', '', regex=True)
    df = df.replace('\\\\', '', regex=True)
    df = df.replace('\.', '', regex=True)
    new_series = []
    for i, element in enumerate(df['Message']):
        if element is None:
            new_series.append(df['Message'].iloc[i])
        else:
            new_series.append(df['Message'].iloc[i])
    df['Message'] = new_series
    return df


def get_message_level(df):
    df['Level'] = df['Category'].astype(str).str[1]
    df = df.drop('Category', axis=1)
    df['Level'] = df['Level'].replace({'F': 'Fatal',
                                       'E': 'Error',
                                       'W': 'Warning',
                                       'R': 'Refactor',
                                       'C': 'Convention',
                                       'I': 'Information'})
    return df


def get_message_category(df):
    msg_df = pd.DataFrame(df['Message'].str.split(r'(').to_list(),
                          index=df.index)
    df = df.merge(msg_df, left_index=True, right_index=True)
    df = df.drop(0, axis=1)
    df = df.rename(columns={1: 'Message category'})
    if df.shape[1] > 2:
        new_series = []
        for i, element in enumerate(df[2]):
            if element is None:
                new_series.append(df['Message category'].iloc[i])
            else:
                new_series.append(df[2].iloc[i])
        df['Message category'] = new_series
        df = df.drop(2, axis=1)
    df['Message category'] = df['Message category'].str[:-1]
    df = df.sort_values(by='Message category')
    return df

def get_clean_message(df):
    df['Message'] = [a.replace(b, '').strip()
                     for a, b in zip(df['Message'], df['Message category'])]
    df['Message'] = df['Message'].replace('\(\);', '', regex=True)
    return df
