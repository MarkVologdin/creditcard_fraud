import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import gdown

file_id = "18yJSdWlpfFeDLhgqlNgt-BKR5c3bVUL4"
url = f"https://drive.google.com/uc?id={file_id}"

output = "creditcard.csv"
gdown.download(url, output, quiet=False)

# Загрузка данных
df = pd.read_csv(output)

# Заголовок страницы
st.title("Моделирование и визуализация данных о мошенничестве с картами")

# Показываем первые строки датасета
st.subheader("Первые строки данных")
st.write(df.head())

# Визуализация распределения классов (мошенничество / не мошенничество)
st.subheader("Распределение классов (мошенничество / не мошенничество)")
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
st.pyplot(fig)

# Описание неравномерности
fraud_percentage = df['Class'].value_counts(normalize=True) * 100
st.write("Процентное распределение классов (мошенничество и не мошенничество):")
st.write(fraud_percentage)

# Загружаем сохраненную модель
st.subheader("Загрузка сохраненной модели")
model = joblib.load('xgboost_model.pkl')  # Замените на путь к вашему сохраненному файлу модели

# Прогнозирование на тестовых данных
X = df.drop('Class', axis=1)
y = df['Class']

# Для демонстрации, создадим произвольные обучающие и тестовые данные (вместо реальных, если нет разделенных данных)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Прогнозирование с использованием загруженной модели
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для класса 1

# Оценка модели
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Отчет о классификации (оставляем только precision и recall)
st.subheader("Отчет о классификации (Precision и Recall)")
report = classification_report(y_test, y_pred, output_dict=True)
precision_recall = {
    "Класс 0 (Не мошенничество)": {
        "Precision": report["0"]["precision"],
        "Recall": report["0"]["recall"]
    },
    "Класс 1 (Мошенничество)": {
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"]
    }
}
st.write(precision_recall)

# ROC-AUC
st.subheader(f"ROC-AUC: {roc_auc:.2f}")
