# Machine Learning Notes

## Kaggle Introduction
The target value that you want to find out is called `y`.  
The input value(s) are called `X`.  
To figure out `X` you decide which features of the dataset could be interesting to predict the value that you want. (Door color is not correlated with housing price, number of rooms is)
```python
# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
```

Split the training data into chunks, 1 chunk that is the training set, the other that we'll test the data on.
```python
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```
`random_state` is the seed for the pseudorandom generator.

A simple model is a `DecisionTreeRegressor`. It creates an abstract tree which contains the features, the depth of the tree can be controlled and is a big factor on how accurate the predictions are gonna be. Too deep and we're _overfitting_ too shallow and we're _underfitting_. Depending on the depth, more or less entries are in the leaves. An example of _overfitting_ is, that there is only 1 example for a decision tree leaf.
![decision tree](https://i.imgur.com/R3ywQsR.png)

The goal is always to find the sweet spot between _overfitting_ and _underfitting_:  
![](https://i.imgur.com/2q85n9s.png)
```python
# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)
```
To figure out the if we're _over-_ or _underfitting_ we can calculate the _mean absolute error_
```python
# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
```
Explore with different `max_leaf_nodes`(affects the depth) and choose the best size. If you find it you can use it to get better predictions:
```python
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
```

There are also other models like `RandomForestRegressor`, `DecisionTreeRegressor` is actually quite outdated.