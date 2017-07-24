# MACHINE LEARNING HASH RECOGNITION 
### libraries
[install scipy stack](https://www.scipy.org/install.html)

[install scikit-learn](http://scikit-learn.org/stable/install.html)

[install matplotlib](https://matplotlib.org/users/installing.html)

### content
###### input.csv
Learning data. Each line consists of an example of data written in plain text or a hash value. Coma separates a value from its label. 
```0: this is a plain text```
```1: this is a hash```

###### makeDataset.py
Execute this script first. It extracts numerical features from a value of each example. This are:
```number of upper case characters```
```number of lower case characters```
```number of white characters```
```number of digits```
```number of vovels```
```number of consonants```
```number of polish diacritic signs```
```number of special characters```
The result will be stored in dataset.csv file

###### classify.py
Machine learning itselves.
