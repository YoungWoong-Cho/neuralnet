import utils as utils

if __name__=='__main__':
    data = utils.read_file('./weight-height.csv')
    X = data[:, 1:]
    y = data[:, 0]

    train_val_ratio = 0.8
    X_train = X[:int(len(X) * train_val_ratio)]
    X_test = X[int(len(X) * train_val_ratio):]
    y_train = y[:int(len(y) * train_val_ratio)]
    y_test = y[int(len(y) * train_val_ratio):]
    
    epoch = int(1e4)
    lr = 1e-4
    min_error = 1e-2
    model = utils.LogisticRegression(intercept = True)
    w = model.fit(X_train, y_train, epoch, lr, min_error)

    y_hat = model.predict(X_test, w)
    model.evaluate_acc(y_test, y_hat)