# AMLS_assignment22_23
## How to run the source code for spark detection?
1. Install the necessary environment:
```
pip install -r requirements.txt
```
2. Store the data in the Datasets file.
3. Enter the main.py file path and start classification:
```
python main.py
```
4. The results will be saved in A1, A2, B1 and B2 respectively, and the best model will be saved at the main.py file path.

## Role of each file
**main.py** is the file to start the whole process of the assignment. It uses the functions provided by **A1.py**, **A2.py**, **B1.py** and **B2.py** to accomplish the classification tasks.

**A1.py** includes functions to load dataset, preprocess data, train data and get test results of Task A1.

**A2.py** includes functions to load dataset, preprocess data, train data and get test results of Task A2.

**B1.py** includes functions to load dataset, preprocess data, train data and get test results of Task B1.

**B2.py** includes functions to load dataset, preprocess data, train data and get test results of Task B2.

All the result visualisations and model constructions are integrated into corresponding .py file, code comments will explain them clearly.

