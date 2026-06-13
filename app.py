import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

#Page Config
st.set_page_config(page_title="Finance Anomaly Detector", page_icon="💸", layout="wide")

#Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detector", "Spending Insights", "Learn"])

if page == "Detector":
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
    method = st.sidebar.selectbox("Detection Method", ["IQR", "Z-Score", "Ensemble (Both)"])
    threshold = st.sidebar.slider("Z-Score Threshold", 1.5, 3.5, 2.5, 0.1)
else:
    method = "IQR"
    threshold = 2.5

#Shared Data Loading
def load_data(mode, uploaded=None):
    df = None
    if mode == "Use Sample Data":
        np.random.seed(42)
        categories = ['Food', 'Transport', 'Shopping', 'Utilities', 'Entertainment', 'Healthcare']
        cat_params = {
            'Food':{'mean': 400,  'std': 80},
            'Transport':{'mean': 200,  'std': 50},
            'Shopping':{'mean': 600,  'std': 200},
            'Utilities':{'mean': 1200, 'std': 100},
            'Entertainment':{'mean': 300,  'std': 100},
            'Healthcare':{'mean': 150,  'std': 60},
        }
        rows = []
        for day in pd.date_range('2024-01-01', periods=365):
            for _ in range(np.random.randint(2, 5)):
                cat = np.random.choice(categories)
                p = cat_params[cat]
                amount = max(10, np.random.normal(p['mean'], p['std']))
                rows.append({'date': day, 'category': cat, 'amount': round(amount, 2)})
        df = pd.DataFrame(rows)
        anomaly_indices = np.random.choice(df.index, size=20, replace=False)
        df.loc[anomaly_indices, 'amount'] *= np.random.uniform(4, 8, size=20)
    elif mode == "Upload CSV/Excel" and uploaded is not None:
        if uploaded.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded, parse_dates=['date'])
        else:
            df = pd.read_csv(uploaded, parse_dates=['date'])
    elif mode == "Enter Manually":
        if st.session_state.get('manual_data'):
            df = pd.DataFrame(st.session_state.manual_data)
    return df

#Input
def input_section():
    st.header("Input Your Transactions")
    mode = st.radio("Choose input mode:", ["Upload CSV/Excel", "Enter Manually", "Use Sample Data"])
    uploaded = None
    df = None
    if mode == "Use Sample Data":
        df = load_data(mode)
        st.success(f"Sample data loaded — {len(df)} transactions")
    elif mode == "Upload CSV/Excel":
        st.info("""
**Required Format** — Your file must have exactly these 3 columns:
- **date** → YYYY-MM-DD format (e.g. 2024-01-15)
- **category** → Any category name of your choice (e.g. Food, Rent, Netflix, Gym)
- **amount** → Numbers only (e.g. 450.00) — no Rs symbol or commas
        """)
        sample = pd.DataFrame({
            'date':['2024-01-01', '2024-01-01', '2024-01-02'],
            'category':['Food', 'Rent', 'Netflix'],
            'amount':[450, 12000, 499]
        })
        st.markdown("**Example:**")
        st.dataframe(sample, hide_index=True)
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        if uploaded:
            df = load_data(mode, uploaded)
            st.success(f"Uploaded {len(df)} transactions")
            st.dataframe(df.head())
    elif mode == "Enter Manually":
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = []
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_input = st.date_input("Date")
        with col2:
            cat_input = st.text_input("Category", placeholder="e.g. Food, Rent, Netflix")
        with col3:
            amt_input = st.number_input("Amount (Rs)", min_value=1.0, value=500.0)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add"):
                if cat_input.strip() == "":
                    st.warning("Please enter a category name.")
                else:
                    st.session_state.manual_data.append({
                        'date':pd.Timestamp(date_input),
                        'category':cat_input.strip().title(),
                        'amount':amt_input
                    })
        if st.session_state.manual_data:
            df = pd.DataFrame(st.session_state.manual_data)
            st.dataframe(df)
            if st.button("Clear All"):
                st.session_state.manual_data = []
    return df

# PAGE DETECTOR
if page == "Detector":
    st.title("Personal Finance Anomaly Detector")
    st.markdown("Upload your transactions or enter them manually to detect unusual spending patterns.")
    df = input_section()
    if df is not None and len(df) > 10:
        st.markdown("---")

        # Recurring category toggle
        all_categories = sorted(df['category'].unique().tolist())
        recurring = st.multiselect(
            "Mark categories as recurring — these will be excluded from anomaly detection (e.g. Rent, Electricity):",
            options=all_categories
        )
        df_detect = df[~df['category'].isin(recurring)].copy()
        if len(df_detect) < 10:
            st.warning("Too many categories marked as recurring — not enough data left for detection. Please unmark some.")
        else:
            st.header("Anomaly Detection Results")

            # Z-Score
            df_detect['z_score'] = stats.zscore(df_detect['amount'])
            df_detect['flag_zscore'] = (df_detect['z_score'].abs() > threshold).astype(int)

            # IQR
            df_detect['flag_iqr'] = 0
            for cat in df_detect['category'].unique():
                mask = df_detect['category'] == cat
                subset = df_detect.loc[mask, 'amount']
                Q1, Q3 = subset.quantile(0.25), subset.quantile(0.75)
                IQR = Q3 - Q1
                df_detect.loc[mask, 'flag_iqr'] = ((subset < Q1 - 1.5*IQR) |
                                             (subset > Q3 + 1.5*IQR)).astype(int)
                
            # Ensemble
            df_detect['flag_combined'] = ((df_detect['flag_zscore'] == 1) &
                                    (df_detect['flag_iqr'] == 1)).astype(int)
            if method == "IQR":
                df_detect['flagged'] = df_detect['flag_iqr']
            elif method == "Z-Score":
                df_detect['flagged'] = df_detect['flag_zscore']
            else:
                df_detect['flagged'] = df_detect['flag_combined']
            flagged_df = df_detect[df_detect['flagged'] == 1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Transactions Analysed", f"{len(df_detect):,}")
            col2.metric("Anomalies Detected", f"{len(flagged_df)}")
            col3.metric("Anomaly Rate", f"{len(flagged_df)/len(df_detect)*100:.2f}%")
            col4.metric("Total Spend", f"Rs {df['amount'].sum():,.2f}")
            if recurring:
                st.info(f"Excluded from detection: {', '.join(recurring)} — marked as recurring expenses")
            st.subheader("Flagged Transactions")
            if len(flagged_df) > 0:
                st.dataframe(flagged_df[['date', 'category', 'amount', 'z_score']].sort_values('amount', ascending=False))
            else:
                st.info("No anomalies detected with current settings.")
            st.subheader("Spending Over Time")
            fig, ax = plt.subplots(figsize=(14, 4))
            normal = df_detect[df_detect['flagged'] == 0]
            anomalies = df_detect[df_detect['flagged'] == 1]
            ax.scatter(normal['date'], normal['amount'], alpha=0.4, s=10, color='steelblue', label='Normal')
            ax.scatter(anomalies['date'], anomalies['amount'], alpha=0.9, s=60, color='red', label='Anomaly')
            ax.set_title('Transactions Over Time')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            st.subheader("Spend by Category")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            cat_spend = df_detect.groupby('category')['amount'].sum().sort_values(ascending=False)
            sns.barplot(x=cat_spend.index, y=cat_spend.values, hue=cat_spend.index, palette='Set2', legend=False, ax=ax2)
            ax2.set_title('Total Spend by Category')
            ax2.set_ylabel('Amount (Rs)')
            plt.tight_layout()
            st.pyplot(fig2)
            st.subheader("Anomaly Rate by Category")
            cat_anom = df_detect.groupby('category').agg(
                total=('flagged', 'count'),
                anomalies=('flagged', 'sum')
            ).reset_index()
            cat_anom['rate'] = (cat_anom['anomalies'] / cat_anom['total'] * 100).round(2)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.barplot(data=cat_anom, x='category', y='rate', hue='category',
                        palette='Reds_d', legend=False, ax=ax3)
            ax3.set_title('Anomaly Rate by Category (%)')
            ax3.set_ylabel('Anomaly %')
            plt.tight_layout()
            st.pyplot(fig3)
            st.subheader("Summary Report")
            if len(flagged_df) > 0:
                top = flagged_df.loc[flagged_df['amount'].idxmax()]
                st.markdown(f"""
- **Total transactions analysed:** {len(df_detect):,}
- **Recurring categories excluded:** {', '.join(recurring) if recurring else 'None'}
- **Anomalies detected:** {len(flagged_df)} ({len(flagged_df)/len(df_detect)*100:.2f}%)
- **Most affected category:** {flagged_df['category'].value_counts().idxmax()}
- **Highest anomalous transaction:** Rs {top['amount']:,.2f} on {top['date'].date()} ({top['category']})
- **Detection method used:** {method}
                """)
            st.subheader("Download Results")
            csv = df_detect.to_csv(index=False).encode('utf-8')
            st.download_button("Download Flagged Data as CSV", csv, "flagged_transactions.csv", "text/csv")
    elif df is not None and len(df) <= 10:
        st.warning("Please add at least 10 transactions for meaningful analysis.")


# PAGE SPENDING INSIGHTS
elif page == "Spending Insights":
    st.title("Spending Insights")
    st.markdown("Understand where your money is going — by category, by day, and over time.")
    df = input_section()
    if df is not None and len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
        df['day']  = df['date'].dt.date
        st.markdown("---")
        total_spend = df['amount'].sum()
        avg_daily = df.groupby('day')['amount'].sum().mean()
        biggest_txn = df.loc[df['amount'].idxmax()]
        top_category = df.groupby('category')['amount'].sum().idxmax()
        num_days = df['day'].nunique()
        st.header("Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Spend", f"Rs {total_spend:,.2f}")
        col2.metric("Daily Average", f"Rs {avg_daily:,.2f}")
        col3.metric("Days Tracked", f"{num_days}")
        col4.metric("Top Category", top_category)
        col5.metric("Biggest Transaction", f"Rs {biggest_txn['amount']:,.2f}")
        st.markdown("---")
        st.header("Spend by Category")
        cat_summary = df.groupby('category')['amount'].agg(['sum', 'mean', 'count']).reset_index()
        cat_summary.columns = ['Category', 'Total Spend (Rs)', 'Avg per Transaction (Rs)', 'No. of Transactions']
        cat_summary['% of Total'] = (cat_summary['Total Spend (Rs)'] / total_spend * 100).round(2)
        cat_summary = cat_summary.sort_values('Total Spend (Rs)', ascending=False).reset_index(drop=True)
        st.dataframe(cat_summary.style.format({
            'Total Spend (Rs)': 'Rs {:,.2f}',
            'Avg per Transaction (Rs)': 'Rs {:,.2f}',
            '% of Total': '{:.2f}%'
        }), hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            cat_summary['Total Spend (Rs)'],
            labels=cat_summary['Category'],
            autopct='%1.1f%%',
            colors=sns.color_palette('Set2', len(cat_summary)),
            startangle=140
        )
        ax.set_title('Spending Distribution by Category', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---")
        st.header("Daily Spending Trend")
        daily = df.groupby('day')['amount'].sum().reset_index()
        daily.columns = ['date', 'total']
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        ax2.plot(daily['date'], daily['total'], color='steelblue', linewidth=1.5)
        ax2.fill_between(daily['date'], daily['total'], alpha=0.2, color='steelblue')
        ax2.axhline(avg_daily, color='red', linestyle='--', label=f'Daily Avg Rs {avg_daily:,.0f}')
        ax2.set_title('Daily Total Spend', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Amount (Rs)')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown("---")
        st.header("Top 5 Highest Spending Days")
        top_days = daily.sort_values('total', ascending=False).head(5).reset_index(drop=True)
        top_days.columns = ['Date', 'Total Spent (Rs)']
        top_days['Total Spent (Rs)'] = top_days['Total Spent (Rs)'].round(2)
        st.dataframe(top_days, hide_index=True)
        st.markdown("---")
        st.header("Category Spend Heatmap Over Time")
        df['month'] = df['date'].dt.to_period('M').astype(str)
        pivot = df.groupby(['month', 'category'])['amount'].sum().unstack().fillna(0)
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot.T, cmap='YlOrRd', linewidths=0.5, annot=True, fmt='.0f', ax=ax3)
        ax3.set_title('Monthly Spend by Category (Rs)', fontsize=14)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Category')
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown("---")
        st.header("Biggest Single Transaction")
        st.info(f"""
        Rs {biggest_txn['amount']:,.2f} spent on {biggest_txn['category']} 
        on {pd.Timestamp(biggest_txn['date']).date()}
        """)
        st.subheader("Download Insights")
        csv = cat_summary.to_csv(index=False).encode('utf-8')
        st.download_button("Download Category Summary as CSV", csv, "category_summary.csv", "text/csv")
    elif df is not None and len(df) == 0:
        st.warning("No data found. Please add transactions first.")

#PAGE LEARN
elif page == "Learn":
    st.title("How Does Anomaly Detection Work?")
    st.markdown("---")
    st.header("What is an Anomaly?")
    st.markdown("""
    An anomaly (also called an outlier) is a transaction that looks
    unusually different from your normal spending pattern.
    For example:
    - You normally spend Rs 400 on food but one day you spent Rs 4,000
    - Your electricity bill is always Rs 1,200 but one month it was Rs 8,000
    These could mean fraud, billing errors, or unusual one-time expenses.
    Note: Large recurring expenses like rent or electricity bills may appear flagged
    if not marked as recurring. Use the recurring category toggle on the Detector page to exclude them.
    """)
    st.markdown("---")
    st.header("Method 1 - Z-Score")
    st.markdown("""
    Z-Score measures how far a transaction is from the average in terms of standard deviations.
    Formula:
    """)
    st.latex(r"Z = \frac{X - \mu}{\sigma}")
    st.markdown("""
    Where:
    - X = transaction amount
    - mu = mean (average) of all transactions
    - sigma = standard deviation
    How to read it:
    - Z-Score of 0 means perfectly average
    - Z-Score of 2.5 means 2.5 standard deviations above average — suspicious
    - Z-Score of -2.5 means unusually low — also suspicious
    Threshold used in this app: transactions with Z greater than 2.5 are flagged.
    Strength: Simple and fast
    Weakness: Treats all categories the same
    """)
    st.markdown("---")
    st.header("Method 2 - IQR (Interquartile Range)")
    st.markdown("""
    IQR is smarter — it flags anomalies within each category separately.
    Steps:
    1. Sort transactions by amount within a category
    2. Find Q1 (25th percentile) and Q3 (75th percentile)
    3. IQR = Q3 - Q1
    4. Flag anything below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR
    """)
    st.latex(r"\text{Lower fence} = Q1 - 1.5 \times IQR")
    st.latex(r"\text{Upper fence} = Q3 + 1.5 \times IQR")
    st.markdown("""
    Example for Food category:
    - Q1 = Rs 350, Q3 = Rs 500, IQR = Rs 150
    - Upper fence = Rs 500 + 1.5 x 150 = Rs 725
    - Any food transaction above Rs 725 is flagged
    Strength: Context-aware, works per category
    Weakness: May miss anomalies if data is already skewed
    """)
    st.markdown("---")
    st.header("Method 3 - Ensemble (Both Together)")
    st.markdown("""
    The ensemble method combines both Z-Score and IQR.
    A transaction is flagged only if both methods agree it is suspicious.
    Strength: Highest precision — almost zero false alarms
    Weakness: May miss some anomalies (lower recall)
    Best used when you want to be very sure before flagging something.
    """)
    st.markdown("---")
    st.header("Which Method Should You Use?")
    comparison = pd.DataFrame({
        'Method': ['Z-Score', 'IQR', 'Ensemble'],
        'Precision':['High', 'Medium', 'Very High'],
        'Recall': ['Medium', 'High', 'Medium'],
        'Best For': [
            'Quick overview',
            'Detailed category analysis',
            'Minimizing false alarms'
        ]
    })
    st.dataframe(comparison, hide_index=True)
    st.markdown("---")
    st.info("Go to Detector or Insights in the sidebar to start analyzing your transactions!")