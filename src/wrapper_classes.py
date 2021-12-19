from ensembles import RandomForestMSE, GradientBoostingMSE
from exceptions import TargetError, DatasetError

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


class Dataset:

    __strategies = {'MF': SimpleImputer(strategy='most_frequent'),
                    'MED': SimpleImputer(strategy='median'),
                    'MEAN': SimpleImputer(strategy='mean')}
    
    __cat_strategies = {'NONE': 'Без обработки',
                        'MF': 'Самая частая категория'}

    __num_strategies = {'NONE': 'Без обработки',
                        'MED': 'Медиана',
                        'MEAN': 'Среднее'}

    __to_drop = ['ID', 'id', 'index', 'INDEX']

    def __init__(self, name, data, target_col=None,
                 cat_strategy=None, num_strategy=None):
        self.name = name
        self.data = data
        self.raw_data = data.copy()
        self.target_col = target_col
        self.cat_strategy = cat_strategy
        self.num_strategy = num_strategy
        self._create_target()
        self._drop_useless()
        self._proccess_data()
        self._make_info()

    def _make_info(self):
        info = [('Имя датасета', self.name)]
        info.append(('Стратегия замены NaN для категориальных признаков',
                     self.__cat_strategies[self.cat_strategy]))
        info.append(('Стратегия замены NaN для численных признаков',
                     self.__num_strategies[self.num_strategy]))
        self.info = pd.DataFrame(info, columns=['Параметр', 'Значение'])

    def _create_target(self):
        if not self.target_col:
            self.target = None
            return None

        if self.target_col not in self.data.columns:
            raise TargetError('Invalid target column name')

        self.target = self.data[self.target_col]
        self.data.drop(self.target_col, axis=1, inplace=True)

    def _drop_useless(self):
        for col in self.__to_drop:
            if col in self.data.columns:
                self.data.drop(col, axis=1, inplace=True)

    def _proccess_data(self):
        imputers = []

        categorical = self.data.select_dtypes(include=[object]).columns
        numeric = self.data.select_dtypes(include=[int, float]).columns

        if self.cat_strategy != 'NONE':
            cat_imputer = self.__strategies[self.cat_strategy]
            imputers.append(('CI', cat_imputer, categorical))
        
        if self.num_strategy != 'NONE':
            num_imputer = self.__strategies[self.num_strategy]
            imputers.append(('NI', num_imputer, numeric))
        
        if categorical.any():
            encoder = ColumnTransformer([('OE',
                                          OrdinalEncoder(),
                                          categorical)],
                                        remainder='passthrough')
        
        if imputers:
            imputer = ColumnTransformer(imputers, remainder='passthrough')
            self.data[:] = imputer.fit_transform(self.data)

        self.data = encoder.fit_transform(self.data)

        if self.target is not None:
            self.target = self.target.to_numpy()

    def get_data(self):
        return self.data, self.target
    
    def get_raw(self):
        return self.raw_data

    def has_target(self):
        return self.target is not None

    
class Model:
    __model_classes = {'RF': RandomForestMSE,
                       'GB': GradientBoostingMSE}

    __model_names = {'RF': 'Случайный лес',
                     'GB': 'Градиентный бустинг'}
    
    __ru_params = {'n_estimators': 'Количество базовых алгоритмов',
                   'max_depth': 'Максимальная глубина',
                   'feature_subsample_size': 'Размерность подвыборки признаков',
                   'learning_rate': 'Темп обучения'}

    def __init__(self, model, name, param_form):
        self.model_type = model
        self.model_class = self.__model_classes[model]
        self.name = name
        self.params = param_form.data
        self.val_score = None
        self._create_info()

    def _create_info(self):
        info = [('Имя модели', self.name)]
        info.append(('Модель', self.__model_names[self.model_type]))

        for key, value in self.params.items():
            info.append((self.__ru_params[key], value))

        self.info = pd.DataFrame(info, columns=['Параметр', 'Значение'])

    def fit(self, train_data, val_data):
        self.model = self.model_class(**self.params)
        X, y = train_data.get_data()
        X_val, y_val = val_data.get_data()
        self.train_shape = X.shape[1]

        if X_val.shape[1] != self.train_shape:
            raise DatasetError('Inconsistent shapes: train_data, val_data')
    
        self.target_col = train_data.target_col
        self.data_name = train_data.name

        val_score = self.model.fit(X, y, X_val, y_val)
        self.val_score = val_score

        return val_score
    
    def predict(self, dataset):
        X = dataset.get_data()[0]

        if X.shape[1] != self.train_shape:
            raise DatasetError('Inconsistent shapes: train_data, test_data')

        y_pred = self.model.predict(X)
        self.y_pred = pd.DataFrame(y_pred,
                                   columns=[self.target_col])

        return self.y_pred

    def is_fitted(self):
        return self.val_score is not None
    
    def get_val_score(self):
        n_estimators = self.params['n_estimators']
        score = list(zip(range(1, n_estimators+1), self.val_score))
        return pd.DataFrame(score,
                            columns=['n_estimators', self.target_col])
