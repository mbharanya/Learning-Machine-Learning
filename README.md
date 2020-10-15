# Machine Learning Notes

# Kaggle Introduction
From [here](https://www.kaggle.com/learn/intro-to-machine-learning)  

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
Explore using different `max_leaf_nodes`(affects the depth) and choose the best size. If you find it you can use it to get better predictions:
```python
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
```

There are also other models like `RandomForestRegressor`, `DecisionTreeRegressor` is actually quite outdated.

## Deep Learning
 | Expression | Description | 
 | ---------- | ----------- | 
 | Tensor | Matrix with `n` dimensions <img src="tensor.png" width=150> | 
 | Convolution / Filter | Small tensor, that can be multiplied over a small section of the main image. Multiply it with the values in the matrix in the area that you want to check. High values=more likely that what you're searching | 

About the size of convolutions:  
```
While any one convolution measures only a single pattern, there are more possible convolutions that can be created with large sizes. So there are also more patterns that can be captured with large convolutions.

For example, it's possible to create a 3x3 convolution that filters for bright pixels with a dark one in the middle. There is no configuration of a 2x2 convolution that would capture this.

On the other hand, anything that can be captured by a 2x2 convolution could also be captured by a 3x3 convolution.

Does this mean powerful models require extremely large convolutions? Not necessarily. In the next lesson, you will see how deep learning models put together many convolutions to capture complex patterns... including patterns to complex to be captured by any single convolution.
```
https://www.kaggle.com/dansbecker/building-models-from-convolutions

## Transfer Learning
Neural nets are bases on increasingly complex layers. The first layer will look for simple shapes, the next for more complex ones and so on. The last layer will produce the predictions. With transfer learning we can modify the last layer, without needing to recreate all the other ones.
![transfer-learning-layers](transfer-learning-layers.png)
The last layer before the prediction layer is a 1 dimensional tensor (vector)


When compiling a model you can specify the following arguments:

- **optimizer** determines how we determine the numerical values that make up the model. So it can affect the resulting model and predictions
- **loss** determines what goal we optimize when determining numerical values in the model. So it can affect the resulting model and predictions
- **metrics** determines only what we print out while the model is being built, but it doesn't affect the model itself.

```python
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# is the amount of choices that the output has (rotated, not-rotated)  
num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# means the layers of the NN are in sequential order.  
my_new_model = Sequential()
# `include_top=False` exclude the layer that makes predictions  
# `resnet_weights_path` does not include the weights for the last layer as well.  
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
# Add the dense layer that makes the predictions
# softmax => probabilities
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False
```

This is for checking accuracy (from categorical crossentropy)
```python
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```

Keras has some nice utils to split folders into the categories that we need.
```python
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/rural_and_urban_photos/train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/rural_and_urban_photos/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)
```

## Data augmentation
Images can be flipped or shifted to increase the training data, Keras helps us do that:
```python
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)
```

Validation is usually only done with the base data _not_ the augmented one.


# Kaggle: Intermediate Machine Learning
https://www.kaggle.com/alexisbcook/introduction

## Missing values
> Imputation fills in the missing values with some number

Sklearn has a `SimpleImputer` already, which just imputes the mean value:
```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

```python
imputer = SimpleImputer()
imputer.fit(X_test)
final_X_test = pd.DataFrame(imputer.fit_transform(X_test))

final_X_test.columns = X_test.columns

# Drop columns that we found to contain no data
final_X_test = final_X_test.drop(cols_with_missing, axis=1)

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)
```

## Categorical Variables
>Consider a survey that asks how often you eat breakfast and provides four options: "Never", "Rarely", "Most days", or "Every day". In this case, the data is categorical, because responses fall into a fixed set of categories.
