import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

housing = pd.read_csv("data.csv")
# print(housing.head()) #prints the first 5 rows of data

# print(housing.info()) #prints the number of entries in each column, memory used, etc



# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
# #test_size describes the ratio to split the train and test data in
# print(f"Rows in train set:  {len(train_set)}\n Rows in test set: ({len(test_set)})" )


#making train and test set such that it descibes the whole dataset, for eg. in the above dataset, the 35 values that are 1 for chas
#can be present in test set and not in train set leading to train set not knowwing the presence of value 1
split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#copying the strat_train_set into housing
housing = strat_train_set.copy()

#creating a new feature TAXRM i.e tax per room
housing['TAXRM'] = housing['TAX']/housing['RM']

housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# #imputer helps to fll the missing values, choosing strategy median implies we wish to add median as the missing value
# imputer = SimpleImputer(strategy= "median")
# imputer.fit(housing)

# #here we impute all the missing values from the housing dataset
# X = imputer.transform(housing)

# housing_tr = pd.DataFrame(X, columns = housing.columns)

#creating a pipeline to transform the data automatically whenever neeeded
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

housing_transformed = my_pipeline.fit_transform(housing)

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

model.fit(housing_transformed, housing_labels)

# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]

# prepared_data = my_pipeline.transform(some_data)

# model.predict(prepared_data)


# #evaluating the model ----------------------------------------------------------
# housing_predictions = model.predict(housing_transformed)
# mse = mean_squared_error(housing_labels, housing_predictions)
# rmse = np.sqrt(mse)

# print(rmse)

#Cross validation------------------------------------------------------------------
# scores = cross_val_score(model, housing_transformed, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

# print_scores(rmse_scores)

# print(housing)


x_test = strat_test_set.drop("MEDV", axis = 1)
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse)

print(final_predictions, list(y_test))