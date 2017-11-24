import urllib.request
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
respond = urllib.request.urlopen(url)
dataset_html = respond.read()
db = np.float32(bs(dataset_html).text.splitlines())



