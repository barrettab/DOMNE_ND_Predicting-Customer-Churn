# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- This projects is based around creating tests, logging, and best coding practices using Python.  
We will implement your learnings from the course to identify credit card customers that are most likely to churn using a dataset from [Kaggle website](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code) 

## Project Description
In this project we move from a jupyter notebook with a data science task to production code. 

### Project Structure
- data
  - `bank_data.csv`: Dataset used for this project. 
- images
  - eda
   - churn_distribution.png
   - customer_age_distribution.png
   - heatmap.png
   - marital_status_distribution.png
   - total_transaction_distribution.png
 - results
   - feature_importance.png
   - logistics_results.png
   - rf_results.png
   - roc_curve_result.png
- logs
  - `churn_library.log`: When running scripts all logs will be written here.
- models
  - logistic_model.pkl
  - rfc_model.pkl
- `churn_library.py`: library containing all functions of the data science task
- `churn_notebook.ipynb`: The original notebook containing the initial data science project
- `churn_script_logging_and_tests.py`: Task and logging that test the functions of the churn_library.py library and log erros that occour.
- requirements.txt
- README.md


## Running Files
All python libraries used in this repository can be `pip` installed and are found in requirements.txt. 

You can run the the following commands in order to retrieve the results:

* test each of the functions and provide any errors to a file stored in the `logs` folder.

```
python churn_script_logging_and_tests.py
```
All functions and refactored code associated with the original notebook.
```
python churn_library.py
```

You can also check the pylint score, as well as perform the auto-formatting using the following commands:

```
pylint churn_library_solution.py
pylint churn_script_logging_and_tests_solution.py
```

The files here were formated using:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests_solution.py
autopep8 --in-place --aggressive --aggressive churn_library_solution.py


