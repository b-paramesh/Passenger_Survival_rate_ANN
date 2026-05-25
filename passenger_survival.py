import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Titanic AI Survival Prediction",
    page_icon="🚢",
    layout="wide",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(to right, #eef2f7, #ffffff);
    }

    .title-style {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1f4e79;
    }

    .subtitle-style {
        text-align: center;
        font-size: 18px;
        color: #5c677d;
        margin-bottom: 30px;
    }

    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8edf4;
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        height: 55px;
        background: linear-gradient(90deg,#1f77b4,#4facfe);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg,#155a8a,#2d8cf0);
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA + TRAIN MODEL
# =========================================================

@st.cache_resource
def load_and_train_model():

    df = pd.read_csv("Titanic-Dataset.csv")

    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode Sex
    df['Sex'] = df['Sex'].map({
        'male': 0,
        'female': 1
    })

    # Encode Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    embarked_map = {
        'S': 0,
        'C': 1,
        'Q': 2
    }

    df['Embarked'] = df['Embarked'].map(embarked_map)

    features = [
        'Pclass',
        'Sex',
        'Age',
        'Fare',
        'SibSp',
        'Parch',
        'Embarked'
    ]

    X = df[features].values
    y = df['Survived'].values

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Train Test Split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------------------------
    # TensorFlow Model
    # -----------------------------------------------------

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(
            32,
            activation='relu',
            input_shape=(7,)
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            16,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            8,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=0
    )

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    predictions = model.predict(X_test, verbose=0)
    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    return model, scaler, accuracy, history


# =========================================================
# LOAD MODEL
# =========================================================

with st.spinner("Training Deep Learning Model..."):

    model, scaler, accuracy, history = load_and_train_model()

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title-style">🚢 Titanic Survival Prediction AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-style">Deep Learning Powered Passenger Survival Prediction System</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Model Information")

st.sidebar.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.success("TensorFlow ANN Model Trained Successfully")

st.sidebar.info(
    """
    Features Used:
    - Passenger Class
    - Gender
    - Age
    - Fare
    - Siblings/Spouses
    - Parents/Children
    - Embarked Port
    """
)

# =========================================================
# INPUT FORM
# =========================================================

st.markdown("## 🧾 Passenger Information")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.slider(
            "Age",
            1,
            100,
            25
        )

    with col2:

        fare = st.slider(
            "Fare (£)",
            0,
            600,
            100
        )

        sibsp = st.slider(
            "Siblings / Spouses",
            0,
            10,
            0
        )

        parch = st.slider(
            "Parents / Children",
            0,
            10,
            0
        )

    with col3:

        embarked = st.selectbox(
            "Embarked Port",
            ["Southampton", "Cherbourg", "Queenstown"]
        )

    submit = st.form_submit_button("🔍 Predict Survival")

# =========================================================
# PREDICTION
# =========================================================

if submit:

    sex_encoded = 1 if sex == "Female" else 0

    embarked_map = {
        "Southampton": 0,
        "Cherbourg": 1,
        "Queenstown": 2
    }

    embarked_encoded = embarked_map[embarked]

    user_data = np.array([[
        pclass,
        sex_encoded,
        age,
        fare,
        sibsp,
        parch,
        embarked_encoded
    ]])

    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled, verbose=0)

    survival_probability = float(prediction[0][0])

    death_probability = 1 - survival_probability

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("## 📈 Prediction Results")

    col_a, col_b = st.columns(2)

    if survival_probability >= 0.5:
        result = "✅ SURVIVED"
        confidence = survival_probability
    else:
        result = "❌ NOT SURVIVED"
        confidence = death_probability

    with col_a:

        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Prediction</h4>
                <h1>{result}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_b:

        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Confidence Score</h4>
                <h1>{confidence * 100:.2f}%</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # DONUT CHART
    # =====================================================

    st.markdown("## 📊 Probability Distribution")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Survival", "Death"],
                values=[survival_probability, death_probability],
                hole=0.6,
                textinfo='percent+label',
                marker=dict(
                    colors=["#2ecc71", "#e74c3c"]
                )
            )
        ]
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # FEATURE DISPLAY
    # =====================================================

    st.markdown("## 🧠 Passenger Feature Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Passenger Class",
            "Gender",
            "Age",
            "Fare",
            "Siblings/Spouses",
            "Parents/Children",
            "Embarked"
        ],
        "Value": [
            pclass,
            sex,
            age,
            fare,
            sibsp,
            parch,
            embarked
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True
    )

# =========================================================
# TRAINING HISTORY
# =========================================================

st.markdown("---")
st.markdown("## 📉 Model Training Performance")

history_df = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss']) + 1),
    "Loss": history.history['loss'],
    "Validation Loss": history.history['val_loss']
})

fig2 = px.line(
    history_df,
    x="Epoch",
    y=["Loss", "Validation Loss"],
    title="Training vs Validation Loss"
)

fig2.update_layout(
    height=400
)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>🚀 Built with Streamlit + TensorFlow + Plotly</h4>
    </center>
    """,
    unsafe_allow_html=True
)