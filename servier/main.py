import os
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
# Model choices here
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from termcolor import colored

from servier.data import get_data, get_X_y, LOCAL_PATH, load_final_model, load_model_from_path, get_pipeline_model_from_path
from servier.dl import mlp_model_2, cnn_model, rnn_model
from servier.encoders import MorganFingerprintEncoder, TokenizerEncoder
from servier.plot import plot_confusion_wiki
from servier.utils import simple_time_tracker, perf_eval_classif, get_class_weights


class Trainer(object):

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containing all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """
        self.kwargs = kwargs
        self.estimator = self.kwargs.get("estimator", "RandomForest")
        self.model = self.kwargs.get("model", "cnn")
        self.gridsearch = kwargs.get("gridsearch", False)  # apply gridsearch if True
        self.local = kwargs.get("local", True)  # if True training is done locally
        self.get_pipeline()
        # Dimensionnality reduction
        self.model_params = None  # for
        self.fp_size = kwargs.get("fp_size", 1024)
        # Data
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # if self.estimator != 'NN' else False
        self.resample = self.kwargs.get("resample", False)
        self.w = get_class_weights(self.y_train)
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  stratify=self.y_train, test_size=0.15)
            if self.resample:
                pass
        self.nrows = self.X_train.shape[0]  # nb of rows to train on

    def get_estimator(self):
        """
        return chosen model
        Defines set of parameters to tune if Gridsearch parmaeter is set to True,
        :return:
        """
        estimator = self.estimator
        n_jobs = self.kwargs.get("n_jobs", 4)
        if estimator == "SVM":
            svc_kernel = self.kwargs.get("svc_kernel", "linear")
            model = SVC(kernel=svc_kernel, C=1.0, class_weight='balanced', cache_size=1000)
        elif estimator == "RandomForest":
            model = RandomForestClassifier()
            self.model_params = {'bootstrap': [True, False],
                                 'max_features': ['auto', 'sqrt']}
        elif estimator == "NN":
            m_args = dict(batch_size=self.kwargs.get('epochs', 10),
                          epochs=self.kwargs.get('epochs', 20),
                          shuffle=True,
                          class_weight=self.w,
                          verbose=1,
                          validation_split=0.1,
                          # callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
                          )
            if self.model == 'rnn':
                model = KerasClassifier(rnn_model, **m_args)
            elif self.model == 'cnn':
                model = KerasClassifier(cnn_model, **m_args)
            elif self.model == 'mlp':
                model = KerasClassifier(mlp_model_2, size=self.fp_size, **m_args)
        else:
            model = SGDClassifier()
        estimator_params = self.kwargs.get("estimator_params", {})
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def set_pipeline(self):
        """
        Sets Whole Workflow (Preprocessing + Feature Engineering + model)
        """
        if self.model in ['cnn', 'rnn']:
            encoder = TokenizerEncoder()
        else:
            encoder = MorganFingerprintEncoder(size=self.fp_size)

        preprocessor = Pipeline(steps=[
            ("feat_encoder", encoder)])

        self.pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("clf", self.get_estimator())])

    def get_pipeline(self):
        """
        Load pipeline for inference or evaluation
        """
        if self.kwargs.get("infer"):
            model_paths = self.kwargs.get("model_paths", None)
            if not model_paths:
                self.model_type=self.kwargs.get("model_type", None)
                self.pipeline = load_final_model(model=self.model_type)
            else:
                self.pipeline = load_model_from_path(path_pipeline=model_paths[0],
                                                     path_model=model_paths[1])
        else:
            self.pipeline=None

    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator
        {'rgs__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
          'rgs__max_features' : ['auto', 'sqrt'],
          'rgs__max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        """
        # Here to apply ramdom search to pipeline, need to follow naming "rgs__paramname"
        print("Adding Hyper parameters tuning")
        params = {"clf__" + k: v for k, v in self.model_params.items()}
        if self.feature_selection:
            params["preprocessing__feature_selection__k"] = [60, 100, 140]
        n_jobs = self.kwargs.get("n_jobs", None)
        print(n_jobs)
        self.pipeline = RandomizedSearchCV(estimator=self.pipeline, param_distributions=params,
                                           n_iter=10,
                                           cv=2,
                                           verbose=2,
                                           random_state=42,
                                           n_jobs=n_jobs,
                                           pre_dispatch=8)  # here we copy data 8 times

    @simple_time_tracker
    def train(self):
        """
        trainer method, launching whole pipeline training (preprocessing + training of model)
        :return:
        """
        self.set_pipeline()
        if self.gridsearch:
            self.add_grid_search()
        if self.kwargs.get("estimator") == 'NN':
            self.pipeline.fit(self.X_train, self.y_train)
        else:
            self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluates the model on validation or test set
        :return:
        """
        self.metrics_train = self.compute_metric(self.X_train, self.y_train)
        if self.split:
            self.metrics_val = self.compute_metric(self.X_val, self.y_val, show=False)
            print(colored(
                "f1-score_train: {} || f1-score_val: {}".format(self.metrics_train["f1"], self.metrics_val["f1"]),
                "blue"))
            print(colored(
                "AUC-score_train: {} || AUC-score_val: {}".format(self.metrics_train["ROC"], self.metrics_val["ROC"]),
                "blue"))
        else:
            print(colored("f1-score_train: {}".format(self.metrics_train["f1"]), "blue"))
            print(colored("AUC-score_train: {}".format(self.metrics_train["ROC"]), "blue"))

    def get_proba(self):
        self.proba_train = self.compute_proba(self.X_train, self.y_train)
        if self.split:
            self.proba_val = self.compute_proba(self.X_val, self.y_val)

    def plot_graphs(self, savefig=False):
        """
        Plots ROC curve and confusion matrix for chosen classifier
        :param savefig:
        :return:
        """
        plot_confusion_wiki(self.metrics_val["confusion_matrix"], save=False)
        rfc_disp = plot_roc_curve(self.pipeline, self.X_val, self.y_val, alpha=0.8)
        title = f"""{self.kwargs.get("estimator")}, {self.kwargs.get("nrows")} rows"""
        if self.kwargs.get("feature_selection"):
            title += f"""{self.kwargs.get("n_reduce_dim")} feat"""
        plt.title(title)
        if savefig:
            path = "/tmp/roc.png"
            plt.savefig(path)

    def compute_metric(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        metrics = perf_eval_classif(y_test, y_pred, verbose=False)
        return metrics

    def compute_proba(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_proba = self.pipeline.predict_proba(X_test)
        res = pd.DataFrame(y_proba)
        res["label"] = y_test.reset_index(drop=True)
        if show:
            print(colored(res.sample(5), "blue"))
        return res

    def save_model(self, path=LOCAL_PATH, upload=True, auto_remove=True):
        """
        Saves model as a pipeline to keep preprocessing and model tight together
        If DL is used:
         - .joblib file for preprocessing
         - .h5 file for model (weights)
         """
        f1_score, auc = round(self.metrics_val["f1"], 4), round(self.metrics_val["ROC"], 4),
        model_path = path + "/models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if self.estimator == 'NN':
            id = f'{f1_score}_{auc}'
            model = model_path + 'keras_model_' + self.model + '-' + id + '.h5'
            pipeline = model_path + 'keras_pipeline_' + self.model + '-' + id + '.joblib'
            # Save the Keras model first:
            self.pipeline.named_steps['clf'].model.save(model)
            # This hack allows us to save the sklearn pipeline:
            self.pipeline.named_steps['clf'].model = None
            # Finally, save the pipeline:
            joblib.dump(self.pipeline, pipeline)
        else:
            model_name = f"{self.estimator}_{self.nrows}_{self.n_reduced_dim}_{f1_score}.joblib"
            joblib.dump(self.pipeline, model_path + model_name)
        print(colored(f"model saved locally", "green"))


if "__main__" == __name__:
    test = True
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #params = dict(fp_size=2048,
    #              estimator="NN",
    #              model='mlp',
    #              epochs=30,
    #              batch_size=32,
    #              n_reduce_dim=100,
    #              resample=False,
    #              n_jobs=-1)
    #l_params = [params]
    ## Get and clean data
    #print("############   Loading Data   ############")
    #l_params[0]["local"] = True
    #df = get_data(**l_params[0])
    #for run_params in l_params:
    #    X_train, y_train = get_X_y(df)
    #    print("shape: {}".format(X_train.shape))
    #    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    #    # Train and save model, locally and
    #    t = Trainer(X=X_train, y=y_train, **run_params)
    #    del X_train, y_train
    #    print(colored("############  Training model   ############", "red"))
    #    t.train()
    #    print(colored("############  Evaluating model ############", "blue"))
    #    t.evaluate()
    #    print(colored("############   Saving model    ############", "green"))
    #    t.save_model()

    #params = dict(split=False,
    #              infer=True,
    #              model_type=0
    #              )
    #df = get_data(test=True)
    #X_test, y_test = get_X_y(df)
    #l_model_paths = get_pipeline_model_from_path(archi='mlp')
    #for model_paths in l_model_paths:
    #    print(model_paths)
    #    evaluator = Trainer(X=X_test, y=y_test, model_paths=model_paths, **params)
    #    evaluator.evaluate()

    params = dict(split=False,
                  infer=True,
                  model_type=1
                  )
    df = get_data(test=True)
    X_test, y_test = get_X_y(df)
    evaluator = Trainer(X=X_test, y=y_test, **params)
    evaluator.evaluate()