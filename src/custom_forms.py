from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange


class ModelSelectionForm(FlaskForm):
    model_names = [('RF', 'Случайный лес'), ('GB', 'Градиентный бустинг')]

    model = SelectField('Выберите модель машинного обучения',
                        choices=model_names)
    name = StringField('Введите название модели', validators=[DataRequired()])
    submit = SubmitField('Продолжить')


class ModelParamsForm(FlaskForm):
    n_estimators = IntegerField('Количество базовых алгоритмов', default=100)
    max_depth = IntegerField('Максимальная глубина', validators=[Optional()])
    feature_subsample_size = IntegerField('размерность признакового ' + 
                                          'подпространства для одного ' +
                                          'базового алгоритма',
                                          validators=[Optional()])
    learning_rate = FloatField('Темп обучения', default=0.1)
    submit = SubmitField('Сохранить и продолжить')


class FileUploadForm(FlaskForm):
    cat_strategies = [('NONE', 'Без обработки'),
                      ('MF', 'Самая частая категория')]
    num_strategies = [('NONE', 'Без обработки'),
                      ('MED', 'Медиана'),
                      ('MEAN', 'Среднее')]

    data_file = FileField('Загрузите датасет',
                          validators=[FileRequired(),
                                      FileAllowed(['csv'],
                                                  'CSV only!')])
    target_col = StringField('Столбец целевых значений',
                             validators=[Optional()])
    cat_nan_processing = SelectField('Стратегия замены пропущенных ' +
                                     'значений для категорильных признаков',
                                     choices=cat_strategies)
    num_nan_processing = SelectField('Стратегия замены пропущенных ' +
                                     'значений для численных признаков',
                                     choices=num_strategies)
    data_name = StringField('Имя датасета', validators=[DataRequired()])
    submit = SubmitField('Сохранить')


class TrainForm(FlaskForm):
    train = SelectField('Выберите обучающую выборку',
                        validators=[DataRequired()])
    val = SelectField('Выберите валидационную выборку',
                      validators=[Optional()])
    submit = SubmitField('Обучить')


class TestForm(FlaskForm):
    test = SelectField('Выберите тестовую выборку',
                       validators=[DataRequired()])
    submit = SubmitField('Предсказать')
