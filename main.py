from Classification import Classify
from HasMissingValues import has_missing_value
from KNNimpute import KNNimpute
from MissingValueEstimation import performEM;

# Project 1
datasets1 = [
    ('./data/TrainData1.txt', './data/TrainLabel1.txt', './data/TestData1.txt', './data/TestLabel1.txt'),
    ('./data/TrainData2.txt', './data/TrainLabel2.txt', './data/TestData2.txt', './data/TestLabel2.txt'),
    ('./data/TrainData3.txt', './data/TrainLabel3.txt', './data/TestData3.txt', './data/TestLabel3.txt'),
    ('./data/TrainData4.txt', './data/TrainLabel4.txt', './data/TestData4.txt', './data/TestLabel4.txt'),
    ('./data/TrainData5.txt', './data/TrainLabel5.txt', './data/TestData5.txt', './data/TestLabel5.txt')
]

for train_data_path, train_label_path, test_data_path, test_label_path in datasets1:
  if has_missing_value(train_data_path) or has_missing_value(test_data_path):
    print(f"{train_data_path} or {test_data_path} has missing values. Finding best K...")
    KNNimpute(train_data_path, test_data_path)
  else:
    print(f"{train_data_path} and {test_data_path} do not have missing values.")
  Classify(train_data_path, train_label_path, test_data_path, test_label_path)

# Project 2
datasets2 = [
  ('./data/MissingData1.txt','./data/MissingData1_filled.txt'),
  ('./data/MissingData2.txt','./data/MissingData2_filled.txt'),
  ('./data/MissingData3.txt','./data/MissingData3_filled.txt')
  ]

for missing_data_path, filled_data_path in datasets2:
  print(f"Finding missing value for {missing_data_path}...")
  performEM(missing_data_path, filled_data_path)
