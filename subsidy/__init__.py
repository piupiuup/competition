from models import *
from feature_final import *


if __name__ == "__main__":
    train = get_feature('train')
    test = get_feature('test')
    result = predict(train,test)