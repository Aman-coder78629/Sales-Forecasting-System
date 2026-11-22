
# ============================================================
# SALES FORECASTING & INTELLIGENCE DASHBOARD
# Complete Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import warnings
import io
import re
import zipfile

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #08111f,
        #111d33,
        #101729
    );
    color: #ffffff;
}

section[data-testid="stSidebar"] {
    background: #0b1424;
    border-right: 1px solid #253650;
}

h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #172943,
        #101d32
    );
    border: 1px solid #2d4260;
    padding: 20px;
    border-radius: 15px;
}

div[data-testid="stMetric"] label {
    color: #a9bfdc !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab"] {
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263c59;
    border-radius: 12px;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 SALES INTELLIGENCE")
st.sidebar.markdown("### Forecasting & Analytics")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "MAIN MENU",
    [
        "Executive Dashboard",
        "Data Loading & EDA",
        "Time Series Analysis",
        "Sales Forecasting",
        "Category & Region Forecasting",
        "Anomaly Detection",
        "Product Demand Segmentation"
    ]
)


# ============================================================
# 4. DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):

    file_name = file_name.lower()

    if file_name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(io.BytesIO(file_bytes))

    elif file_name.endswith(".zip"):

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:

            csv_files = [
                f for f in z.namelist()
                if f.lower().endswith(".csv")
                and not f.startswith("__MACOSX/")
            ]

            if not csv_files:
                raise ValueError(
                    "No CSV file found inside ZIP."
                )

            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f)

    elif file_name.endswith(".csv"):

        df = pd.read_csv(io.BytesIO(file_bytes))

    else:

        raise ValueError(
            "Unsupported format. Upload CSV, ZIP or Excel."
        )

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    return df


# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

def preprocess_data(data):

    df = data.copy()

    # Clean column names

    df.columns = [
        re.sub(
            r"[^a-z0-9]",
            "",
            str(col).lower()
        )
        for col in df.columns
    ]

    # Automatic column mapping

    column_mapping = {

        "orderdate": "Order Date",
        "date": "Order Date",
        "salesdate": "Order Date",
        "transactiondate": "Order Date",

        "sales": "Sales",
        "revenue": "Sales",
        "totalsales": "Sales",
        "amount": "Sales",

        "category": "Category",
        "productcategory": "Category",
        "family": "Category",

        "region": "Region",
        "store": "Region",
        "storenbr": "Region",
        "storeid": "Region",

        "productname": "Product Name",
        "product": "Product Name",
        "item": "Product Name",

        "subcategory": "Sub-Category",
        "orderid": "Order ID",
        "customerid": "Customer ID",
        "segment": "Segment",
        "shipmode": "Ship Mode"
    }

    df.rename(
        columns=column_mapping,
        inplace=True
    )

    # Validate required columns

    required = ["Order Date", "Sales"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Required columns missing: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert dates

    dates = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    if dates.isna().mean() > 0.5:

        dates = pd.to_datetime(
            df["Order Date"],
            errors="coerce",
            dayfirst=True
        )

    df["Order Date"] = dates

    # Convert sales

    df["Sales"] = (
        df["Sales"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    # Remove invalid records

    df.dropna(
        subset=["Order Date", "Sales"],
        inplace=True
    )

    if df.empty:
        raise ValueError(
            "No valid date and sales records found."
        )

    # Optional columns

    optional_columns = [
        "Category",
        "Sub-Category",
        "Region",
        "Product Name",
        "Order ID",
        "Customer ID",
        "Segment",
        "Ship Mode"
    ]

    if "Product Name" not in df.columns:

        if "Category" in df.columns:
            df["Product Name"] = df["Category"]
        else:
            df["Product Name"] = "Unknown"

    for col in optional_columns:

        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    # Feature engineering

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter

    return df.sort_values("Order Date")


# ============================================================
# 6. UPLOAD DATASET
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sales Dataset",
    type=["csv", "zip", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:

        raw_data = load_data(
            uploaded_file.getvalue(),
            uploaded_file.name
        )

        df = preprocess_data(raw_data)

        st.sidebar.success(
            "Dataset loaded successfully!"
        )

        with st.expander("Check Dataset Information"):

            st.write("Original Columns:")
            st.write(raw_data.columns.tolist())

            st.write("Processed Columns:")
            st.write(df.columns.tolist())

            st.write("Dataset Shape:", df.shape)

            st.dataframe(df.head())

    except Exception as e:

        st.error(f"Dataset loading error: {e}")
        st.stop()

else:

    st.title("📊 Sales Intelligence Dashboard")

    st.warning(
        "Please upload your sales dataset using the sidebar."
    )

    st.info(
        "Required columns: Order Date and Sales. "
        "Category, Region and Product Name are optional."
    )

    st.stop()


# ============================================================
# 7. GLOBAL FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered_df = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    filtered_df = filtered_df[
        filtered_df["Order Date"].dt.date.between(
            date_range[0],
            date_range[1]
        )
    ]

elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:

    filtered_df = filtered_df[
        filtered_df["Order Date"].dt.date == date_range[0]
    ]

regions = sorted(filtered_df["Region"].unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

filtered_df = filtered_df[
    filtered_df["Region"].isin(selected_regions)
]

categories = sorted(filtered_df["Category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

filtered_df = filtered_df[
    filtered_df["Category"].isin(selected_categories)
]

if filtered_df.empty:

    st.warning(
        "No records found. Please change your filters."
    )

    st.stop()


# ============================================================
# 8. COMMON FUNCTIONS
# ============================================================

def monthly_sales(data):

    monthly = (
        data.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    return monthly


def format_currency(value):

    return f"${value:,.2f}"


def plot_layout(fig, title=None):

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Poppins",
            color="white"
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def download_csv(data, filename):

    csv = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def prepare_time_series(data):

    monthly = monthly_sales(data)

    ts = monthly.set_index(
        "Order Date"
    )["Sales"]

    ts = ts.asfreq("ME").fillna(0)

    return ts


def make_sarima_model(series):

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(disp=False)


def generate_forecast(data, periods=3):

    ts = prepare_time_series(data)

    if len(ts) < 24:

        raise ValueError(
            "At least 24 monthly observations "
            "are required for seasonal SARIMA."
        )

    result = make_sarima_model(ts)

    forecast = result.get_forecast(
        steps=periods
    )

    forecast_values = forecast.predicted_mean

    confidence = forecast.conf_int()

    return ts, forecast_values, confidence, result


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) > 1
        else np.nan
    )

    return mae, rmse, r2


def create_forecast_chart(ts, forecast, confidence):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts.values,
            mode="lines+markers",
            name="Historical Sales",
            line=dict(
                color="#38bdf8",
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Forecasted Sales",
            line=dict(
                color="#f59e0b",
                width=3,
                dash="dash"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(forecast.index) +
              list(forecast.index[::-1]),
            y=list(confidence.iloc[:, 1]) +
              list(confidence.iloc[:, 0][::-1]),
            fill="toself",
            fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval"
        )
    )

    return plot_layout(
        fig,
        "Historical vs Forecasted Sales"
    )


# ============================================================
# 9. EXECUTIVE DASHBOARD
# ============================================================

if menu == "Executive Dashboard":

    st.title("📊 Sales Intelligence Dashboard")

    st.markdown(
        "### Business Performance Overview"
    )

    st.caption(
        "Monitor revenue, transactions, products "
        "and regional sales performance."
    )

    total_sales = filtered_df["Sales"].sum()

    # Avoid presenting Unknown as a real order count

    if (filtered_df["Order ID"] != "Unknown").any():

        total_orders = filtered_df.loc[
            filtered_df["Order ID"] != "Unknown",
            "Order ID"
        ].nunique()

    else:

        total_orders = len(filtered_df)

    avg_sales = filtered_df["Sales"].mean()

    total_products = (
        filtered_df["Product Name"].nunique()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        format_currency(total_sales)
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Average Sales",
        format_currency(avg_sales)
    )

    col4.metric(
        "Unique Products",
        f"{total_products:,}"
    )

    st.markdown("---")

    monthly = monthly_sales(filtered_df)

    left, right = st.columns(2)

    with left:

        fig = px.line(
            monthly,
            x="Order Date",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend"
        )

        fig.update_traces(
            line=dict(
                color="#38bdf8",
                width=3
            )
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

    with right:

        category_sales = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            category_sales,
            names="Category",
            values="Sales",
            hole=0.5,
            title="Sales by Category"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:

        region_sales = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales")
        )

        fig = px.bar(
            region_sales,
            x="Sales",
            y="Region",
            orientation="h",
            color="Region",
            title="Regional Sales Performance"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

    with right:

        top_products = (
            filtered_df.groupby("Product Name")["Sales"]
            .sum()
            .nlargest(10)
            .reset_index()
            .sort_values("Sales")
        )

        fig = px.bar(
            top_products,
            x="Sales",
            y="Product Name",
            orientation="h",
            title="Top 10 Products"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

    st.markdown("### Recent Transactions")

    st.dataframe(
        filtered_df.sort_values(
            "Order Date",
            ascending=False
        ).head(20),
        use_container_width=True
    )


# ============================================================
# 10. DATA LOADING & EXPLORATORY DATA ANALYSIS
# ============================================================

elif menu == "Data Loading & EDA":

    st.title("📁 Data Loading & Exploratory Data Analysis")

    st.markdown(
        "### Understand the structure and quality of your sales data."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Rows",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Total Columns",
        f"{filtered_df.shape[1]}"
    )

    c3.metric(
        "Missing Values",
        f"{filtered_df.isnull().sum().sum():,}"
    )

    c4.metric(
        "Duplicate Records",
        f"{filtered_df.duplicated().sum():,}"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dataset Preview",
            "Data Quality",
            "Sales Analysis",
            "Correlation Analysis"
        ]
    )

    with tab1:

        st.subheader("Dataset Preview")

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
        )

        st.subheader("Statistical Summary")

        st.dataframe(
            filtered_df.describe(
                include="all"
            ).T,
            use_container_width=True
        )

        download_csv(
            filtered_df,
            "sales_dataset.csv"
        )

    with tab2:

        st.subheader("Missing Value Analysis")

        missing = pd.DataFrame({

            "Column": filtered_df.columns,

            "Missing Values":
                filtered_df.isnull().sum().values,

            "Missing Percentage":
                (
                    filtered_df.isnull().mean().values * 100
                ).round(2)

        })

        st.dataframe(
            missing,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        st.write("Duplicate Records")

        st.write(
            filtered_df[
                filtered_df.duplicated(keep=False)
            ].head(100)
        )

    with tab3:

        left, right = st.columns(2)

        with left:

            category_sales = (
                filtered_df.groupby("Category")["Sales"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                category_sales,
                x="Category",
                y="Sales",
                color="Category",
                title="Category-wise Sales"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        with right:

            fig = px.histogram(
                filtered_df,
                x="Sales",
                nbins=50,
                title="Sales Distribution"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        fig = px.box(
            filtered_df,
            x="Category",
            y="Sales",
            color="Category",
            title="Sales Distribution by Category"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        subcategory_sales = (
            filtered_df.groupby("Sub-Category")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )

        fig = px.bar(
            subcategory_sales,
            x="Sub-Category",
            y="Sales",
            title="Sub-Category Sales Analysis"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

    with tab4:

        numeric = filtered_df.select_dtypes(
            include=np.number
        )

        if numeric.shape[1] >= 2:

            corr = numeric.corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Heatmap"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        else:

            st.info(
                "At least two numeric columns are needed."
            )


# ============================================================
# 11. TIME SERIES ANALYSIS
# ============================================================

elif menu == "Time Series Analysis":

    st.title("📈 Time Series Analysis")

    st.markdown(
        "### Monthly sales trends, seasonality and decomposition."
    )

    ts = prepare_time_series(filtered_df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Monthly Observations",
        len(ts)
    )

    c2.metric(
        "Average Monthly Sales",
        format_currency(ts.mean())
    )

    c3.metric(
        "Highest Sales Month",
        ts.idxmax().strftime("%B %Y")
    )

    st.markdown("---")

    fig = px.line(
        ts.reset_index(),
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Time Series"
    )

    st.plotly_chart(
        plot_layout(fig),
        use_container_width=True
    )

    if len(ts) >= 24:

        st.subheader("Seasonal Decomposition")

        period = st.selectbox(
            "Select Seasonal Period",
            [3, 4, 6, 12],
            index=3
        )

        try:

            decomposition = seasonal_decompose(
                ts,
                model="additive",
                period=period,
                extrapolate_trend="freq"
            )

            components = pd.DataFrame({

                "Trend": decomposition.trend,

                "Seasonality": decomposition.seasonal,

                "Residual": decomposition.resid

            })

            for column in components.columns:

                fig = px.line(
                    components.reset_index(),
                    x="Order Date",
                    y=column,
                    title=f"{column} Component"
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                f"Decomposition could not be generated: {e}"
            )

    else:

        st.warning(
            "At least 24 monthly observations are recommended "
            "for seasonal decomposition."
        )

    st.subheader("Autocorrelation Analysis")

    if len(ts) >= 5:

        max_lags = min(
            24,
            len(ts) // 2 - 1
        )

        if max_lags >= 1:

            fig, ax = plt.subplots(figsize=(12, 4))

            plot_acf(
                ts,
                lags=max_lags,
                ax=ax
            )

            ax.set_title(
                "Autocorrelation Function (ACF)"
            )

            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 4))

            plot_pacf(
                ts,
                lags=max_lags,
                ax=ax,
                method="ywm"
            )

            ax.set_title(
                "Partial Autocorrelation Function (PACF)"
            )

            st.pyplot(fig)
            plt.close(fig)

    else:

        st.info(
            "More monthly observations are needed for ACF/PACF."
        )


# ============================================================
# 12. SALES FORECASTING
# ============================================================

elif menu == "Sales Forecasting":

    st.title("🔮 Sales Forecasting Model")

    st.markdown(
        "### Predict future sales using seasonal SARIMAX."
    )

    ts = prepare_time_series(filtered_df)

    periods = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=12,
        value=3
    )

    st.write(
        f"Available monthly observations: {len(ts)}"
    )

    if len(ts) < 24:

        st.error(
            "At least 24 monthly observations are recommended "
            "for the seasonal SARIMA model. Upload a dataset "
            "with a longer historical period."
        )

    else:

        if st.button(
            "🚀 Generate Sales Forecast",
            use_container_width=True
        ):

            with st.spinner(
                "Training SARIMA model and generating forecast..."
            ):

                try:

                    # ----------------------------------------
                    # Train-test evaluation
                    # ----------------------------------------

                    test_size = min(
                        3,
                        max(1, len(ts) // 5)
                    )

                    train = ts.iloc[:-test_size]
                    test = ts.iloc[-test_size:]

                    evaluation_result = None

                    if len(train) >= 24:

                        evaluation_model = make_sarima_model(
                            train
                        )

                        evaluation_forecast = (
                            evaluation_model.get_forecast(
                                steps=len(test)
                            ).predicted_mean
                        )

                        mae, rmse, r2 = calculate_metrics(
                            test.values,
                            evaluation_forecast.values
                        )

                        evaluation_result = (
                            mae,
                            rmse,
                            r2
                        )

                    # ----------------------------------------
                    # Final model on full dataset
                    # ----------------------------------------

                    ts, forecast, confidence, model = (
                        generate_forecast(
                            filtered_df,
                            periods
                        )
                    )

                    st.session_state["sales_forecast"] = (
                        ts,
                        forecast,
                        confidence
                    )

                    st.session_state["sales_model"] = model

                    st.session_state["sales_evaluation"] = (
                        evaluation_result
                    )

                    st.success(
                        "Sales forecast generated successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Forecasting error: {e}"
                    )

        if "sales_forecast" in st.session_state:

            ts_saved, forecast, confidence = (
                st.session_state["sales_forecast"]
            )

            st.subheader("Forecast Results")

            if st.session_state.get("sales_evaluation"):

                mae, rmse, r2 = (
                    st.session_state["sales_evaluation"]
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Mean Absolute Error",
                    format_currency(mae)
                )

                c2.metric(
                    "Root Mean Squared Error",
                    format_currency(rmse)
                )

                c3.metric(
                    "R² Score",
                    f"{r2:.3f}" if np.isfinite(r2) else "N/A"
                )

                st.caption(
                    "Evaluation uses a chronological holdout. "
                    "R² may be unreliable with very small test sets."
                )

            fig = create_forecast_chart(
                ts_saved,
                forecast,
                confidence
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            forecast_df = pd.DataFrame({

                "Date": forecast.index,

                "Forecasted Sales": forecast.values,

                "Lower Confidence": confidence.iloc[:, 0].values,

                "Upper Confidence": confidence.iloc[:, 1].values

            })

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            download_csv(
                forecast_df,
                "sales_forecast.csv"
            )

            st.metric(
                "Total Forecasted Revenue",
                format_currency(forecast.sum())
            )

            st.caption(
                "Forecast values are model estimates, not guaranteed sales."
            )


# ============================================================
# 13. CATEGORY & REGION FORECASTING
# ============================================================

elif menu == "Category & Region Forecasting":

    st.title("📊 Category & Region Forecasting")

    st.markdown(
        "### Forecast sales separately for different business segments."
    )

    forecast_dimension = st.selectbox(
        "Select Forecasting Dimension",
        ["Category", "Region"]
    )

    available_groups = sorted(
        filtered_df[forecast_dimension].unique()
    )

    selected_groups = st.multiselect(
        f"Select {forecast_dimension}",
        available_groups,
        default=available_groups[:min(5, len(available_groups))]
    )

    periods = st.slider(
        "Forecast Period (Months)",
        1,
        12,
        3,
        key="group_periods"
    )

    if st.button(
        "Generate Segment Forecast",
        use_container_width=True
    ):

        results = []

        progress = st.progress(0)

        for i, group in enumerate(selected_groups):

            group_data = filtered_df[
                filtered_df[forecast_dimension] == group
            ]

            group_ts = prepare_time_series(group_data)

            if len(group_ts) < 24:

                st.warning(
                    f"{group}: Not enough monthly history. "
                    "At least 24 observations are recommended."
                )

                progress.progress(
                    (i + 1) / max(len(selected_groups), 1)
                )

                continue

            try:

                model = make_sarima_model(group_ts)

                forecast = model.get_forecast(
                    steps=periods
                )

                predictions = forecast.predicted_mean

                for date, value in predictions.items():

                    results.append({

                        forecast_dimension: group,

                        "Forecast Date": date,

                        "Forecasted Sales": value

                    })

            except Exception as e:

                st.warning(
                    f"Forecast failed for {group}: {e}"
                )

            progress.progress(
                (i + 1) / max(len(selected_groups), 1)
            )

        if results:

            result_df = pd.DataFrame(results)

            st.session_state["segment_forecast"] = result_df

            st.success(
                "Segment forecasting completed!"
            )

        else:

            st.warning(
                "No segment forecast could be generated."
            )

    if "segment_forecast" in st.session_state:

        result_df = st.session_state["segment_forecast"]

        st.subheader("Forecast Table")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        fig = px.line(
            result_df,
            x="Forecast Date",
            y="Forecasted Sales",
            color=forecast_dimension,
            markers=True,
            title=f"{forecast_dimension}-wise Sales Forecast"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        comparison = (
            result_df.groupby(forecast_dimension)
            ["Forecasted Sales"]
            .sum()
            .reset_index()
            .sort_values(
                "Forecasted Sales",
                ascending=False
            )
        )

        fig = px.bar(
            comparison,
            x=forecast_dimension,
            y="Forecasted Sales",
            color=forecast_dimension,
            title="Total Forecasted Sales by Segment"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        download_csv(
            result_df,
            "category_region_forecast.csv"
        )


# ============================================================
# 14. ANOMALY DETECTION
# ============================================================

elif menu == "Anomaly Detection":

    st.title("🚨 Sales Anomaly Detection")

    st.markdown(
        "### Identify unusual sales transactions using Isolation Forest."
    )

    contamination = st.slider(
        "Expected Anomaly Percentage",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    if st.button(
        "Detect Sales Anomalies",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing sales transactions..."
        ):

            try:

                anomaly_data = filtered_df.copy()

                features = anomaly_data[
                    ["Sales", "Month", "Quarter"]
                ].copy()

                if len(features) < 10:

                    st.error(
                        "At least 10 records are recommended "
                        "for anomaly detection."
                    )

                else:

                    scaler = StandardScaler()

                    scaled_features = scaler.fit_transform(
                        features
                    )

                    model = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_estimators=200
                    )

                    predictions = model.fit_predict(
                        scaled_features
                    )

                    anomaly_data["Anomaly"] = np.where(
                        predictions == -1,
                        "Anomaly",
                        "Normal"
                    )

                    anomaly_data["Anomaly Score"] = (
                        model.decision_function(
                            scaled_features
                        )
                    )

                    st.session_state["anomaly_results"] = (
                        anomaly_data
                    )

                    st.success(
                        "Anomaly detection completed!"
                    )

            except Exception as e:

                st.error(
                    f"Anomaly detection error: {e}"
                )

    if "anomaly_results" in st.session_state:

        anomaly_data = st.session_state[
            "anomaly_results"
        ]

        anomaly_count = (
            anomaly_data["Anomaly"] == "Anomaly"
        ).sum()

        normal_count = (
            anomaly_data["Anomaly"] == "Normal"
        ).sum()

        anomaly_percentage = (
            anomaly_count / len(anomaly_data) * 100
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Anomalies",
            f"{anomaly_count:,}"
        )

        c2.metric(
            "Normal Transactions",
            f"{normal_count:,}"
        )

        c3.metric(
            "Anomaly Percentage",
            f"{anomaly_percentage:.2f}%"
        )

        st.markdown("---")

        fig = px.scatter(
            anomaly_data,
            x="Order Date",
            y="Sales",
            color="Anomaly",
            size="Sales",
            hover_data=[
                "Category",
                "Region",
                "Product Name",
                "Anomaly Score"
            ],
            title="Normal vs Anomalous Sales Transactions"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        st.subheader("Detected Anomalies")

        anomalies = anomaly_data[
            anomaly_data["Anomaly"] == "Anomaly"
        ].sort_values(
            "Anomaly Score"
        )

        st.dataframe(
            anomalies,
            use_container_width=True
        )

        download_csv(
            anomaly_data,
            "sales_anomaly_results.csv"
        )


# ============================================================
# 15. PRODUCT DEMAND SEGMENTATION
# ============================================================

elif menu == "Product Demand Segmentation":

    st.title("🎯 Product Demand Segmentation")

    st.markdown(
        "### Group products according to their sales performance "
        "using K-Means Clustering."
    )

    n_clusters = st.slider(
        "Number of Product Segments",
        min_value=2,
        max_value=8,
        value=4
    )

    if st.button(
        "Run Product Segmentation",
        use_container_width=True
    ):

        with st.spinner(
            "Performing K-Means clustering..."
        ):

            try:

                product_data = (
                    filtered_df.groupby("Product Name")
                    .agg(

                        Total_Sales=("Sales", "sum"),

                        Average_Sales=("Sales", "mean"),

                        Transaction_Count=("Sales", "count"),

                        Active_Months=("Order Date", "nunique")

                    )
                    .reset_index()
                )

                product_data = product_data.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).dropna()

                if len(product_data) < n_clusters:

                    st.error(
                        "The number of products must be at least "
                        "equal to the selected clusters."
                    )

                else:

                    features = product_data[
                        [
                            "Total_Sales",
                            "Average_Sales",
                            "Transaction_Count",
                            "Active_Months"
                        ]
                    ]

                    scaler = StandardScaler()

                    scaled_features = scaler.fit_transform(
                        features
                    )

                    model = KMeans(
                        n_clusters=n_clusters,
                        random_state=42,
                        n_init=10
                    )

                    product_data["Cluster"] = (
                        model.fit_predict(
                            scaled_features
                        )
                    )

                    # Rank clusters by total revenue.
                    # Cluster IDs themselves have no business meaning.

                    cluster_order = (
                        product_data.groupby("Cluster")
                        ["Total_Sales"]
                        .mean()
                        .sort_values(
                            ascending=False
                        )
                    )

                    labels = {
                        cluster: f"Segment {rank + 1}"
                        for rank, cluster in enumerate(
                            cluster_order.index
                        )
                    }

                    product_data["Segment"] = (
                        product_data["Cluster"].map(labels)
                    )

                    st.session_state["product_segments"] = (
                        product_data
                    )

                    st.success(
                        "Product segmentation completed!"
                    )

            except Exception as e:

                st.error(
                    f"Clustering error: {e}"
                )

    if "product_segments" in st.session_state:

        product_data = st.session_state[
            "product_segments"
        ]

        st.subheader("Product Segmentation Results")

        cluster_summary = (
            product_data.groupby("Segment")
            .agg(

                Total_Products=("Product Name", "count"),

                Total_Sales=("Total_Sales", "sum"),

                Average_Sales=("Average_Sales", "mean")

            )
            .reset_index()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Products",
            f"{len(product_data):,}"
        )

        c2.metric(
            "Total Segments",
            f"{product_data['Segment'].nunique()}"
        )

        c3.metric(
            "Total Product Revenue",
            format_currency(
                product_data["Total_Sales"].sum()
            )
        )

        st.markdown("---")

        left, right = st.columns(2)

        with left:

            fig = px.scatter(
                product_data,
                x="Total_Sales",
                y="Average_Sales",
                color="Segment",
                hover_name="Product Name",
                size="Transaction_Count",
                title="Product Segmentation"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        with right:

            fig = px.bar(
                cluster_summary,
                x="Segment",
                y="Total_Sales",
                color="Segment",
                title="Revenue by Product Segment"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        st.subheader("Segment Summary")

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        st.subheader("Product-level Classification")

        st.dataframe(
            product_data.sort_values(
                "Total_Sales",
                ascending=False
            ),
            use_container_width=True
        )

        download_csv(
            product_data,
            "product_demand_segments.csv"
        )


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:15px;">

        <h4 style="color:#38bdf8;">
            📊 Sales Forecasting & Intelligence Dashboard
        </h4>

        <p>
            Built using Python | Streamlit | Pandas |
            Plotly | Scikit-learn | Statsmodels
        </p>

        <p>
            Data Analytics • Machine Learning • Time Series Forecasting
        </p>

    </div>
    """,
    unsafe_allow_html=True
)
