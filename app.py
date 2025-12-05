import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model + vectorizer
model = pickle.load(open('../model/classifier.pkl', 'rb'))
vectorizer = pickle.load(open('../model/vectorizer.pkl', 'rb'))

st.title("💰 AI Expense Category Predictor")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predictions", "Analytics", "Forecasting"])
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("Use the tabs above to explore!")
st.sidebar.info("💡 AI-powered Finance Insights")

# ===========================================
# TAB 1 - PREDICTIONS
# ===========================================
with tab1:
    st.subheader("🔮 Expense Category Prediction")

    accounts = ['acct_1', 'acct_2', 'acct_3']
    tags = ['tag_1', 'tag_2', 'tag_3']

    selected_account = st.selectbox("Select Account:", accounts)
    selected_tag = st.selectbox("Select Tag:", tags)

    user_input = f"{selected_account} {selected_tag}"

    if st.button("Predict Category"):
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec.toarray())[0]
        st.success(f"🎯 Predicted Category: **{prediction}**")

# ===========================================
# TAB 2 - ANALYTICS
# ===========================================
with tab2:
    st.subheader("📈 Monthly Expense Trend")

    # Load expenses data
    df_expenses = pd.read_csv("../data/Expenses_clean.csv")

    # Convert dates
    df_expenses['date_time'] = pd.to_datetime(df_expenses['date_time'])
    df_expenses['month'] = df_expenses['date_time'].dt.to_period('M')

    # Group by month
    monthly_expenses = df_expenses.groupby('month')['amount'].sum()

    # Plot line chart
    fig, ax = plt.subplots()
    ax.plot(monthly_expenses.index.astype(str), monthly_expenses.values, marker='o')
    ax.set_xlabel("Month")
    ax.set_ylabel("Expense Amount")
    ax.set_title("Monthly Expense Trend")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.success("Insight: You are spending more lately. Try setting a budget next month!")
    st.info(f"💡 Highest spending month → **{monthly_expenses.idxmax()}**")

    st.markdown("---")
    st.subheader("📤 Upload Transactions for Smart Categorization")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        if "account" in new_data.columns and "tags" in new_data.columns:
            new_data['text'] = new_data['account'] + " " + new_data['tags']
            new_vec = vectorizer.transform(new_data['text'])
            new_data['Predicted_Category'] = model.predict(new_vec.toarray())

            st.success("🎯 Categories predicted successfully!")
            st.write(new_data.head())

            csv = new_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Download Updated CSV",
                data=csv,
                file_name="categorized_expenses.csv",
                mime="text/csv"
            )
        else:
            st.error("CSV must contain 'account' and 'tags' columns!")

    st.markdown("---")
    st.subheader("💰 Monthly Savings Trend")

    income = pd.read_csv("../data/Income_clean.csv")
    income['date_time'] = pd.to_datetime(income['date_time'])
    income['month'] = income['date_time'].dt.to_period('M')

    monthly_income = income.groupby('month')['amount'].sum()
    savings = monthly_income - monthly_expenses

    fig2, ax2 = plt.subplots()
    ax2.plot(savings.index.astype(str), savings.values, marker='o')
    ax2.set_title("Savings Trend")
    ax2.axhline(0, color='red', linestyle='--')
    st.pyplot(fig2)
    st.success("Insight: Great job saving! Maintain this positive trend! 💸🚀")


# ===========================================
# TAB 3 - FORECASTING (COMING NEXT)
# ===========================================
with tab3:
    st.subheader("📈 Expense Forecast (Next 3 Months)")

    # Load forecast CSV
    try:
        forecast_df = pd.read_csv("../data/forecast.csv")
        forecast_df['month'] = pd.to_datetime(forecast_df['month'])

        fig4, ax4 = plt.subplots()
        ax4.plot(forecast_df['month'], forecast_df['forecast'], marker='o')
        ax4.set_title("Future Expense Forecast")
        ax4.set_xlabel("Month")
        ax4.set_ylabel("Expected Expense Amount")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        # Insight based on forecast
        next_month = forecast_df.iloc[0]
        st.success(f"🔮 Next Month Expense Forecast: **₹{round(next_month['forecast'],2)}**")

    except:
        st.warning("⚠️ Forecast data is not available yet. Please generate forecast CSV using Jupyter.")
    
