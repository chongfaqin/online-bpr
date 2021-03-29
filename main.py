#!/usr/bin/env python
# coding: utf-8

from preprocessor import Data_preprocessor
from BPR import BPR
import pandas as pd
import tensorflow_io as tfio
from tensorflow_io.kafka.python.ops import (
    kafka_ops,
)  # pylint: disable=wrong-import-position
import tensorflow_io.kafka as kafka_io  # pylint: disable=wrong-import-position

__author__ = "Bo-Syun Cheng"
__email__ = "k12s35h813g@gmail.com"

if __name__ == "__main__":
    data = pd.read_csv('data/ratings_small.csv')
    dp = Data_preprocessor(data)
    train,test = dp.preprocess()
    
    bpr = BPR(train)
    bpr.fit()
