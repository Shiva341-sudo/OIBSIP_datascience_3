import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction")
st.markdown("Upload your `car data.csv`, explore the data, evaluate models, and predict selling prices.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload car data CSV", type=["csv"])
model_choice  = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])
test_size     = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

# ── Load & Preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df_raw = pd.read_csv(file)
    df = df_raw.copy()

    current_year = 2025
    df['Car_Age'] = current_year - df['Year']
    df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

    df = pd.get_dummies(
        df,
        columns=['Fuel_Type', 'Selling_type', 'Transmission'],
        drop_first=True
    )
    return df_raw, df

if uploaded_file:
    df_raw, df = load_and_preprocess(uploaded_file)
else:
    st.info("No file uploaded — please upload `car data.csv` using the sidebar.")
    st.stop()

# ── Features & Target ─────────────────────────────────────────────────────────
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Train selected model
if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📈 Model Results", "🔍 Feature Importance", "🔮 Predict Price"])

# ── TAB 1: EDA ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Raw Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", df_raw.shape[0])
    c2.metric("Features", df_raw.shape[1] - 1)
    c3.metric("Missing Values", df_raw.isnull().sum().sum())

    st.write("**Statistical Summary**")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selling Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Selling_Price'], bins=20, kde=True, color='steelblue', ax=ax)
        ax.set_title("Selling Price Distribution")
        ax.set_xlabel("Selling Price (in Lakhs)")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Present Price vs Selling Price")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df['Present_Price'], y=df['Selling_Price'], ax=ax2, color='coral')
        ax2.set_title("Present Price vs Selling Price")
        st.pyplot(fig2)
        plt.close()

    st.subheader("Car Age vs Selling Price")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=df['Car_Age'], y=df['Selling_Price'], ax=ax3, color='mediumseagreen')
    ax3.set_title("Car Age vs Selling Price")
    st.pyplot(fig3)
    plt.close()

    st.subheader("Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)
    plt.close()

# ── TAB 2: Model Results ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Model: {model_choice}")

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"{mae:.4f}")
    m2.metric("MSE",      f"{mse:.4f}")
    m3.metric("RMSE",     f"{rmse:.4f}")
    m4.metric("R² Score", f"{r2:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    st.info(f"**5-Fold CV R² Score:** {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Actual vs Predicted")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.scatter(y_test, y_pred, alpha=0.6, color='royalblue')
        ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax5.set_xlabel("Actual Price")
        ax5.set_ylabel("Predicted Price")
        ax5.set_title("Actual vs Predicted Price")
        st.pyplot(fig5)
        plt.close()

    with col2:
        st.subheader("Residual Plot")
        residuals = y_test - y_pred
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax6, color='orange')
        ax6.axhline(y=0, color='red', linestyle='--')
        ax6.set_xlabel("Predicted Values")
        ax6.set_ylabel("Residuals")
        ax6.set_title("Residual Plot")
        st.pyplot(fig6)
        plt.close()

# ── TAB 3: Feature Importance ─────────────────────────────────────────────────
with tab3:
    st.subheader("Top 10 Important Features (Random Forest)")

    @st.cache_data
    def get_feature_importance(_X, _y):
        rf = RandomForestRegressor(random_state=42)
        rf.fit(_X, _y)
        return pd.Series(rf.feature_importances_, index=_X.columns)

    importance = get_feature_importance(X, y)
    top10 = importance.nlargest(10).sort_values()

    fig7, ax7 = plt.subplots(figsize=(8, 5))
    top10.plot(kind='barh', ax=ax7, color='mediumslateblue')
    ax7.set_title("Top 10 Important Features")
    ax7.set_xlabel("Importance Score")
    st.pyplot(fig7)
    plt.close()

    st.dataframe(
        importance.reset_index()
        .rename(columns={'index': 'Feature', 0: 'Importance'})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
        .round(4),
        use_container_width=True
    )

# ── TAB 4: Predict ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Predict Selling Price")
    st.markdown("Fill in the car details below to get an estimated selling price.")

    col1, col2 = st.columns(2)

    with col1:
        present_price = st.number_input("Present Price (Lakhs)", 0.5, 100.0, 5.0, 0.5)
        kms_driven    = st.number_input("Kilometres Driven", 100, 500000, 30000, 1000)
        car_age       = st.slider("Car Age (Years)", 1, 20, 5)
        owner         = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

    with col2:
        fuel_type    = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    # Build input matching one-hot encoded columns
    input_data = {col: 0 for col in X.columns}
    input_data['Present_Price']  = present_price
    input_data['Driven_kms']     = kms_driven
    input_data['Car_Age']        = car_age
    input_data['Owner']          = owner

    if fuel_type == "Diesel" and 'Fuel_Type_Diesel' in input_data:
        input_data['Fuel_Type_Diesel'] = 1
    elif fuel_type == "CNG" and 'Fuel_Type_CNG' in input_data:
        input_data['Fuel_Type_CNG'] = 1

    if selling_type == "Individual" and 'Selling_type_Individual' in input_data:
        input_data['Selling_type_Individual'] = 1

    if transmission == "Manual" and 'Transmission_Manual' in input_data:
        input_data['Transmission_Manual'] = 1

    input_df = pd.DataFrame([input_data])

    if st.button("💰 Predict Price"):
        pred_price = model.predict(input_df)[0]
        pred_price = max(pred_price, 0)
        st.success(f"**Estimated Selling Price: ₹ {pred_price:.2f} Lakhs**")
        st.caption(f"Model used: {model_choice}")
