import csv
import numpy
from sklearn.utils import Bunch


# loads csv file
def LoadDataset(dataset):
    with open(r"{:}".format(dataset)) as csv_file:
        data_reader = csv.reader(csv_file, delimiter=";")
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            headers = row[:-1]
            label = row[-1]
            data.append([float(num) for num in headers])
            target.append(float(label))
        data = numpy.array(data)
        target = numpy.array(target)
    return Bunch(data=data, target=target, header_names=feature_names)
