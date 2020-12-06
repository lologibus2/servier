import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from servier.plot import plot_confusion_wiki
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
# Model choices here
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from termcolor import colored

from servier.data import get_data, get_X_y, LOCAL_PATH
from servier.dl import get_model
from servier.encoders import MorganFingerprintEncoder
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
        self.pipeline = None
        self.gridsearch = kwargs.get("gridsearch", False)  # apply gridsearch if True
        self.local = kwargs.get("local", True)  # if True training is done locally
        self.optimize = kwargs.get("optimize", False)  # Optimizes size of Training Data if set to True
        # Dimensionnality reduction
        self.reduce_dim = kwargs.get("reduce_dim", False)
        self.feature_selection = kwargs.get("feature_selection", False)
        self.n_reduced_dim = kwargs.get("n_reduce_dim", 20)
        self.model_params = None  # for
        self.fp_size = kwargs.get("fp_size", 1024)
        # Data
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True) #if self.estimator != 'NN' else False
        self.resample = self.kwargs.get("resample", False)
        self.w = get_class_weights(self.y_train)
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  stratify=self.y_train, test_size=0.15)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on

    def get_estimator(self):
        """
        return chosen model
        Defines set of parameters to tune if Gridsearch parmaeter is set to True,
        :return:
        """
        estimator = self.estimator
        n_jobs = self.kwargs.get("n_jobs", 4)
        print(n_jobs)
        if estimator == "GBM":
            model = GradientBoostingClassifier()
        elif estimator == "SVM":
            svc_kernel = self.kwargs.get("svc_kernel", "linear")
            model = SVC(kernel=svc_kernel, C=1.0, class_weight='balanced', cache_size=1000)
        elif estimator == "RandomForest":
            model = RandomForestClassifier()
            self.model_params = {'bootstrap': [True, False],
                                 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                                 'max_features': ['auto', 'sqrt'],
                                 'min_samples_leaf': [1, 2, 4],
                                 'min_samples_split': [2, 5, 10],
                                 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif estimator == "NN":
            model = KerasClassifier(get_model, size=self.fp_size, batch_size=10, epochs=self.kwargs.get('epochs', 20), shuffle=True, class_weight=self.w,
                                    verbose=1, validation_split=0.1)
        else:
            model = SGDClassifier()
        estimator_params = self.kwargs.get("estimator_params", {})
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def set_pipeline(self):
        """
        Sets Whole Workflow (Preprocessing + Feature Engineering + model)
        Consists in 3 main steps:
        - preprocessing : distinction between categorical | ordinal | numerical variable encoding
        - feature selection : select most decisive features based of input parameters
        - model
        :return:
        """
        pipeline_encoder = make_pipeline(MorganFingerprintEncoder(size=self.fp_size))

        preprocessor = Pipeline(steps=[
            ("feat_encoder", pipeline_encoder)])

        self.pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("clf", self.get_estimator())])

        if self.feature_selection:
            self.pipeline.named_steps["preprocessing"].steps.append(
                ["feature_selection", SelectKBest(k=self.n_reduced_dim)])

        if self.reduce_dim:
            self.pipeline.named_steps["preprocessing"].steps.append(["pca", PCA(n_components=self.n_reduced_dim)])

        # if self.optimize:
        #    self.pipeline.steps.insert(-1, ['optimize_size', OptimizeSize(verbose=False)])

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
        Evaluates the model on validation set
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

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        f1_score, auc = round(self.metrics_val["f1"], 4), round(self.metrics_val["ROC"], 4),
        model_path = LOCAL_PATH + "/models/"
        if self.estimator == 'NN':
            id = f'{f1_score}_{auc}'
            model = model_path+'keras_model'+'-'+id+'.h5'
            pipeline = model_path+'keras_pipeline'+'-'+id+'.joblib'
            # Save the Keras model first:
            self.pipeline.named_steps['clf'].model.save(model)
            # This hack allows us to save the sklearn pipeline:
            self.pipeline.named_steps['clf'].model = None
            # Finally, save the pipeline:
            joblib.dump(self.pipeline, pipeline)
        else:
            model_name = f"{name}_t{self.kwargs['target']}_{self.nrows}_{self.n_reduced_dim}_{f1_score}.joblib"
            joblib.dump(self.pipeline, model_path+model_name)
        print(colored(f"model saved locally", "green"))


if "__main__" == __name__:
    test = True
    warnings.simplefilter(action='ignore', category=FutureWarning)
    params = dict(fp_size=1024,
                  local=True,  # set to False to get data from GCP (Storage or BigQuery)
                  estimator="NN",
                  epochs=3,
                  gridsearch=False,
                  optimize=True,
                  feature_selection=False,
                  reduce_dim=False,
                  n_reduce_dim=100,
                  resample=False,
                  n_jobs=-1)
    l_params = [params]
    # Get and clean data
    print("############   Loading Data   ############")
    l_params[0]["local"] = True
    df = get_data(**l_params[0])
    for run_params in l_params:
        X_train, y_train = get_X_y(df)
        print("shape: {}".format(X_train.shape))
        print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
        # Train and save model, locally and
        t = Trainer(X=X_train, y=y_train, **run_params)
        del X_train, y_train
        print(colored("############  Training model   ############", "red"))
        t.train()
        print(colored("############  Evaluating model ############", "blue"))
        t.evaluate()
        print(colored("############   Saving model    ############", "green"))
        t.save_model()
