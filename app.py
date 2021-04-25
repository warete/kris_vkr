from flask import Flask, redirect, render_template, request
import sqlite3
import datetime
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


# Инициализация веб-приложения
app = Flask(__name__)

# Подключение к бд
db_file_path = "vkr.db"


# Функция для получения подключения к БД
def get_conn(db_file_path):
    conn = sqlite3.connect(db_file_path)
    conn.row_factory = dict_factory
    return conn


with get_conn(db_file_path) as conn:
    cursor = conn.cursor()
    # Создаем таблицу с пациентами, если ее нет
    cursor.execute("""CREATE TABLE IF NOT EXISTS patients 
    (id INTEGER PRIMARY KEY, fio text, birth_date text, diagnosis_date text, diagnosis integer)""")

    # Выбираем из бд данные для обучения
    cursor.execute('select * from train_data')
    train_items = pd.DataFrame(cursor.fetchall())
    # Создаем модель и обучаем
    model = SVC(gamma='scale')
    # x - температуры
    train_x = train_items[['t' + str(i) for i in range(13)]].values
    # y - признак болен/здоров
    train_y = train_items['target']
    # Обучаем через метод fit
    model.fit(train_x, train_y)
    # Посчитаем точность
    print(accuracy_score(train_y, model.predict(train_x)))


# Функция для получения списка пациентов из БД
def get_patients():
    with get_conn(db_file_path) as conn:
        cursor = conn.cursor()
        cursor.execute('select * from patients order by id desc')
        return cursor.fetchall()


# Главная страница
@app.route('/')
def index():
    # Получаем список пациентов и пробрасываем на страницу
    patients = get_patients()
    return render_template('index.html', patients=patients)


# Страница для обработки формы
@app.route('/predict', methods=['POST'])
def predict():
    # Собираем всю информацию из формы
    patient = {'temps': {}, 'fio': request.form.get('fio'), 'birth_date': request.form.get('birth_date')}
    for i in range(13):
        patient['temps']['t' + str(i)] = request.form.get('t' + str(i))

    # Если все что нам надо есть в форме (фио, дата рождения и точки не пустые), то пробуем определить диагноз
    if len(patient['fio']) and len(patient['birth_date']) and len(patient['temps']) == 13:
        with get_conn(db_file_path) as conn:
            # Передаем в модель точки и получаем результат в виде числа (0 - здоров, 1 - болен)
            pred_data = model.predict([list(patient['temps'].values())])
            cursor = conn.cursor()
            # Добавляем результат в бд. Одна строка в бд - одно обследование для пациентов
            cursor.execute('INSERT INTO patients (fio, birth_date, diagnosis, diagnosis_date) VALUES(?,?,?,?)',
                           (patient['fio'], patient['birth_date'], int(pred_data[0]), datetime.datetime.now()))
            conn.commit()
            # Перенаправляем пользователя опять на главную страницу
            return redirect('/')
    else:
        return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
