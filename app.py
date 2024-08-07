import streamlit as st
import pandas as pd
import datetime
import os
import base64
from catboost import CatBoostClassifier, Pool

# Функция загрузки данных
@st.cache_data
def load_data():
    df = pd.read_csv("/home/savr/rink_master/rink_master_47816_wteams.csv")
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    
    # Извлечение года и месяца, и создание нового столбца Season
    df["Year"] = df["gameDate"].dt.year
    df["Month"] = df["gameDate"].dt.month
    df["Season"] = df["Year"].astype(str) + "-" + (df["Year"] + 1).astype(str)
    
    return df

data = load_data()

# Определение результата
def determine_result(row):
    if (
        row["Win"] != 0
        or row["regulationWins"] != 0
        or row["regulationAndOtWins"] != 0
        or row["shootoutWins"] != 0
    ):
        return 1  # Победа
    elif row["Loss"] != 0 or row["OTLoss"] != 0:
        return 0  # Поражение
    else:
        return -1  # Неопределено

data["Result"] = data.apply(determine_result, axis=1)

# Маппинг команд на числовые значения
fullname_to_code = {
    "New Jersey Devils": 1, "New York Islanders": 2, "New York Rangers": 3,
    "Philadelphia Flyers": 4, "Pittsburgh Penguins": 5, "Boston Bruins": 6,
    "Buffalo Sabres": 7, "Montréal Canadiens": 8, "Ottawa Senators": 9,
    "Toronto Maple Leafs": 10, "Carolina Hurricanes": 11, "Florida Panthers": 12,
    "Tampa Bay Lightning": 13, "Washington Capitals": 14, "Chicago Blackhawks": 15,
    "Detroit Red Wings": 16, "Nashville Predators": 17, "St. Louis Blues": 18,
    "Calgary Flames": 19, "Colorado Avalanche": 20, "Edmonton Oilers": 21,
    "Vancouver Canucks": 22, "Anaheim Ducks": 23, "Dallas Stars": 24,
    "Los Angeles Kings": 25, "San Jose Sharks": 26, "Columbus Blue Jackets": 27,
    "Minnesota Wild": 28, "Winnipeg Jets": 29, "Arizona Coyotes": 30,
    "Vegas Golden Knights": 31, "Seattle Kraken": 32,
}

data["Team"] = data["Team"].map(fullname_to_code)
data["Opponent"] = data["Opponent"].map(fullname_to_code)

# Разделение данных на обучающую и тестовую выборки
train = data[data["gameDate"] < "2023-10-10"]
test = data[data["gameDate"] >= "2023-10-10"]

# Определение колонок, которые будут удалены
features_to_drop = [
    "Result", "gameDate", "gameID", "gamesPlayed", "Win", "Loss", "Tie",
    "OTLoss", "points", "pointPct", "regulationWins", "regulationAndOtWins",
    "shootoutWins", "goalsFor", "goalsAgainst", "goalsForPerGame",
    "goalsAgainstPerGame", "powerPlayPct", "penaltyKillPct", "powerPlayNetPct",
    "penaltyKillNetPct", "shotsForPerGame", "shotsAgainstPerGame",
    "faceoffWinPct", "Year", "Month", "Season", "NonRegulationTime",
]

# Убедитесь, что колонки для удаления существуют в данных
features_to_drop = [col for col in features_to_drop if col in train.columns]

# Разделение признаков и целевой переменной
X_train = train.drop(columns=features_to_drop)
y_train = train["Result"]
X_test = test.drop(columns=features_to_drop)
y_test = test["Result"]

# Функция для загрузки модели CatBoost
@st.cache_resource
def load_catboost_model(file_path):
    try:
        model = CatBoostClassifier()
        model.load_model(file_path)
        print(f"Тип загруженной модели: {type(model)}")  # Для отладки
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели CatBoost: {e}")
        return None

model_path = "/home/savr/rink_master/catboost_model.cb"
model = load_catboost_model(model_path)

# Функция для предсказания исхода
def predict_winner(row, model):
    print(f"Тип модели в predict_winner: {type(model)}")  # Для отладки
    try:
        # Подготовка входных данных
        features = pd.DataFrame([row], columns=X_train.columns).fillna(0)
        
        # Создание объекта Pool для CatBoost
        pool = Pool(data=features, feature_names=X_train.columns.tolist())
        
        # Сделайте предсказание
        prediction = model.predict(pool)
        print(f"Предсказание: {prediction}")  # Для отладки
        return 'Победа' if prediction[0] == 1 else 'Поражение'
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return "Ошибка"

# Основной код Streamlit
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton:wght@400;700&display=swap');
    
    .title {
        font-size: 48px;
        font-weight: 700;
        color: #0A74DA; /* Темно-голубой цвет */
        font-family: 'Anton', sans-serif; /* шрифт Anton */
        text-transform: uppercase; /* все буквы заглавные */
        text-shadow: 2px 2px 4px #000000; /* тень текста */
        margin-bottom: 20px;
    }

    /* Прозрачные контейнеры */
    .stApp {
        background: rgba(255, 255, 255, 0.2);
    }

    .stMarkdown, .stTable, .stDataFrame, .stForm, .stTextInput, .stDateInput {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }

    /* Прозрачный контейнер для заголовка */
    .title-container {
        background: rgba(255, 255, 255, 0.2);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px.
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title-container"><h1 class="title">Предсказание исходов хоккейных матчей NHL 🏒🥅🏆</h1></div>', unsafe_allow_html=True)

selected_date = st.date_input("Выберите дату", value=datetime.date(2023, 10, 8))

filtered_data = data[data['gameDate'] == pd.to_datetime(selected_date)]

if not filtered_data.empty:
    st.write(f"Игры на {selected_date}:")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.write("Команда")
    col2.write("Оппонент")
    col3.write("На выезде")
    col4.write("Актуальное значение победы")
    col5.write("Предсказание")

    for index, row in filtered_data.iterrows():
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.write(row['Team'])
        col2.write(row['Opponent'])
        col3.write(row['homeRoad'])
        col4.write(row['Win'])
        if col5.button('Предсказание', key=index):
            # Убедитесь, что row передается как pandas.Series и преобразуется в словарь
            row_dict = row.to_dict()
            prediction = predict_winner(row_dict, model)
            st.write(f"Предсказание для игры {row['Team']} vs {row['Opponent']}: {prediction}")
else:
    st.write("Нет игр на выбранную дату.")

# Установка фонового изображения
background_image_path = "4.jpg"

if os.path.exists(background_image_path):
    with open(background_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error(f"Изображение не найдено по пути: {background_image_path}")


