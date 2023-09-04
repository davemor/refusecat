import os
import json
from pathlib import Path
import joblib

from typing import Dict, List

import numpy as np
from peewee import *

# create the data directory, if it doesn't exit
data_path = Path('/Users/dm236/Documents/Research/radarcat/trashcat/starcat_data_contact')
if not os.path.exists(data_path):
    os.makedirs(data_path)

# open the database and create it if required
db = SqliteDatabase(os.path.join(data_path, 'store.db'))


# set up the database models
class Sample(Model):
    label = CharField()
    filename = CharField()
    device = CharField()

    class Meta:
        database = db


# set up the tables if they do not exist
db.connect()
if 'sample' not in db.get_tables():
    db.create_table(Sample)
db.close()


def save_sample(label: str, array: np.ndarray, device: str):
    """
    Save sample data into the database and saves the data as a csv.
    :param label: Label for the sample, used as the class name when training.
    :param array: Numpy array containing the data.
    :param device: Name of the device used to collect the data.
    """
    db.connect()
    record = Sample(label=label, filename='empty', device=device)
    record.save()
    filename = label + str(record.id) + '.csv'
    filepath = os.path.join(data_path, filename)
    np.savetxt(filepath, array)
    record.filename = filename
    record.save()
    db.close()


def load_all_samples() -> List[Sample]:
    """
    Get all the samples in the database and return them.
    :return: List of samples ordered by id
    """
    db.connect()
    samples = [s for s in Sample.select().dicts()]
    db.close()
    return samples


def load_sample_data(record: Sample) -> np.ndarray:
    """
    Load the data for a sample from it's file.
    :param sample: Sample record from the database.
    :return: Numpy array with the sample data in it
    """
    filepath = os.path.join(data_path, record['filename'])
    data = np.loadtxt(filepath)
    return data


def load_samples(ids: List) -> List[Sample]:
    """
    Load a set of samples from the database and return them.
    :param: List of ids of the samples to load
    :return: List of samples ordered by id with the
    """
    # load the sample records from the database
    db.connect()
    samples = [s for s in Sample.select().where(Sample.id << ids).dicts()]
    db.close()

    # load the ndarray for the samples from the file
    # and attach it to the sample object.
    for sample in samples:
        data = load_sample_data(sample)
        sample['data'] = data  # add data to the dictionary

    return samples
