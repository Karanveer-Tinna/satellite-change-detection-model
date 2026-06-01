# Introduction

This dataset is divided into two groups, viz. Main_Dataset and Prepared_Dataset. Main\_Dataset contains 110 satellite images of size 600×600 pixels and manually labelled semantic segmentation masks. Prepared\_Dataset contains three sets: training, validation, and testing. Each set consists of images of size 120×120 pixels which are derived and processed from the Main_Dataset.

# File/Directory Information

| File/Directory Path                    | Description                                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Dataset/Main_Dataset/images/           | Directory of original satellite images of 600×600 px size                                                 |
| Dataset/Main_Dataset/masks/            | Directory of labelled masks of 600×600 px size                                                            |
| Dataset/Main_Dataset/class_dict.csv    | CSV file containing RGB color codes of classes                                                            |
| Dataset/Main_Dataset/train_files.csv   | CSV file containing the file names of the image-mask pairs used in the Prepared_Dataset's training set.   |
| Dataset/Main_Dataset/val_files.csv     | CSV file containing the file names of the image-mask pairs used in the Prepared_Dataset's validation set. |
| Dataset/Main_Dataset/test_files.csv    | CSV file containing the file names of the image-mask pairs used in the Prepared_Dataset's testing set.    |
| Dataset/Prepared_Dataset/train/images/ | Directory of training set images                                                                          |
| Dataset/Prepared_Dataset/train/masks/  | Directory of training set masks                                                                           |
| Dataset/Prepared_Dataset/val/images/   | Directory of validation set images                                                                        |
| Dataset/Prepared_Dataset/val/masks/    | Directory of validation set masks                                                                         |
| Dataset/Prepared_Dataset/test/images/  | Directory of test set images                                                                              |
| Dataset/Prepared_Dataset/test/masks/   | Directory of test set masks                                                                               |

