import numpy as np
import pandas as pd


def generate_features(raw_dataset, df_scheme):
    dataset = pd.merge(
        raw_dataset,
        df_scheme,
        suffixes=('','_scheme'),
        on='scheme',
        how='left'
    )

    if 'no_scheme' in dataset.columns:
        dataset.drop(['no_scheme'], axis=1)

    dataset['area'] = np.pi * (dataset['diameter'] / 2) ** 2

    return dataset


def load_data(dataset_path, scheme_path, random_state=None, split=True, raw=False):
    dataset = pd.read_csv('../datasets/dataset.csv', sep=';')
    scheme = pd.read_csv('../datasets/scheme.csv', sep=';')

    if raw:
        return dataset, scheme

    dataset = generate_features(dataset, scheme)

    # generate train and test split
    np.random.seed(random_state)
    sampled = dataset.groupby('scheme').apply(lambda x: x.sample(2)).reset_index(drop=True)

    dataset = pd.merge(
        dataset,
        sampled[['no', 'sample_code']],
        suffixes=('','_split'),
        on='no',
        how='left'
    )

    dataset = dataset.rename(columns={'sample_code_split': 'split'})
    dataset.split = dataset.split.apply(lambda x: 'train' if pd.isnull(x) else 'test')

    if not split:
        return dataset
    else:
        df_train = dataset[dataset['split'] == 'train'].reset_index(drop=True)
        df_test = dataset[dataset['split'] == 'test'].reset_index(drop=True)

        return df_train, df_test
