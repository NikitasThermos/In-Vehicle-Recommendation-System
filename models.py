
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras_tuner as kt 
import pandas as pd
from scipy.stats import uniform, randint
from imblearn.pipeline import make_pipeline


def logLoss(X_train, y_train, X_test, preprocessor):
    full_pipeline = make_pipeline(preprocessor,
                              SGDClassifier(loss='log_loss',
                                            learning_rate='adaptive', random_state=42))

    param_grid = {
        'sgdclassifier__alpha': uniform(loc=0.0001, scale=3),
        'sgdclassifier__eta0': uniform(loc=0.0001, scale=10),
    }
    print('Starting parameter search for LogLoss')
    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.predict(X_test)

def decisionTree(X_train, y_train, X_test, preprocessor):
    full_pipeline = make_pipeline(preprocessor,
                              DecisionTreeClassifier(random_state=42))

    param_grid = {
        'decisiontreeclassifier__criterion': ['gini', 'entropy'],
        'decisiontreeclassifier__min_samples_split': randint(low=2, high=10),
        'decisiontreeclassifier__max_depth': randint(low=1, high=10),
    }

    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    return random_search.predict(X_test)

def randomForest(X_train, y_train, X_test, preprocessor):
    full_pipeline = make_pipeline(preprocessor,
                              RandomForestClassifier(random_state=42))

    param_grid = {
        'randomforestclassifier__n_estimators': randint(low=5, high=20),
        'randomforestclassifier__max_depth': randint(low=1, high=15),
        'randomforestclassifier__criterion': ['gini', 'entropy'],
    }

    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    return random_search.predict(X_test)

def svm(X_train, y_train, X_test, preprocessor):
    full_pipeline = make_pipeline(preprocessor,
                              SVC(kernel='rbf', gamma='scale', degree=2, random_state=42))

    param_grid = {
        'svc__C': uniform(loc=0.001, scale=10),
    }

    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(X_train, y_train)
    random_search.predict(X_test) 

def dense_network(X_train, y_train, X_test, preprocessor):
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train_nn = pd.DataFrame(X_train_nn, columns=X_train.columns)

    X_train_nn = preprocessor.fit_transform(X_train_nn)
    X_val_nn = preprocessor.transform(X_val_nn)
    X_test_nn = preprocessor.transform(X_test)

    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train_nn)
    X_val_nn = scaler.transform(X_val_nn)
    X_test_nn = scaler.transform(X_test_nn)

    y_train_nn = y_train_nn.values.reshape(-1, 1)
    y_val_nn = y_val_nn.values.reshape(-1, 1)
    

    y_train_nn = tf.cast(y_train_nn, dtype=tf.float32)
    y_val_nn = tf.cast(y_val_nn, dtype=tf.float32)
   

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_nn, y_train_nn))
    train_dataset = train_dataset.shuffle(buffer_size=6000).batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_nn, y_val_nn))
    val_dataset = val_dataset.batch(32)

    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_nn)
    test_dataset = test_dataset.batch(32)

    def build_model(hp):
        n_hidden = hp.Int("n_hidden", min_value=1, max_value=20)
        n_neurons = hp.Int("n_neurons", min_value=16, max_value=1000)
        learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-1, sampling="log")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        for _ in range(n_hidden):
            model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                       optimizer=optimizer,
                        metrics=[tf.keras.metrics.F1Score(name='f1', average='micro', threshold=0.5)])
        return model
        
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience = 5, mode='max', restore_best_weights = True)
    callbacks = [early_stopping_cb]

    random_search_tuner = kt.RandomSearch(build_model, objective=kt.Objective(name="val_f1", direction="max"), max_trials = 10,
                                            overwrite=True, seed=42)

    random_search_tuner.search(train_dataset, epochs=50,
                                validation_data=val_dataset,
                                callbacks=callbacks)
        
    best_model = random_search_tuner.get_best_models()[0]
    y_pred = [1 if p > 0.5 else 0 for p in best_model.predict(X_test_nn)]
    return y_pred