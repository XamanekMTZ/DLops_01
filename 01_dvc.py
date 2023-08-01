import matplotlib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml( "mnist_784", version = 1, return_X_y = True, as_frame = False )

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 10000
)

lengthXTrain = len( X_train )
lengthXTest = len( X_test )
lengthyTrain = len( y_train )
lengthyTest = len( y_test )

print( "X_train: ", lengthXTrain )
print( "X_test: ", lengthXTest )
print( "y_train: ", lengthyTrain )
print( "y_test: ", lengthyTest )