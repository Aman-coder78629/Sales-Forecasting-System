
# ============================================================
# SALES FORECASTING & INTELLIGENCE DASHBOARD
# Complete Integrated Project - 7 Modules
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    border-radius: 9px;
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

        df = pd.read_excel(
            io.BytesIO(file_bytes)
        )

    elif file_name.endswith(".zip"):

        with zipfile.ZipFile(
            io.BytesIO(file_bytes)
        ) as z:

            csv_files = [
                f for f in z.namelist()
                if f.lower().endswith(".csv")
                and not f.startswith("__MACOSX/")
            ]

            if not csv_files:
                raise ValueError(
                    "No CSV file found inside ZIP."
                )

            with z.open(csv_files[0]) as file:

                try:
                    df = pd.read_csv(file)

                except UnicodeDecodeError:

                    file.seek(0)

                    df = pd.read_csv(
                        file,
                        encoding="latin1"
                    )

    elif file_name.endswith(".csv"):

        try:

            df = pd.read_csv(
                io.BytesIO(file_bytes)
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1"
            )

    else:

        raise ValueError(
            "Upload CSV, ZIP, XLSX or XLS."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

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

        # Date

        "orderdate": "Order Date",
        "date": "Order Date",
        "salesdate": "Order Date",
        "transactiondate": "Order Date",

        # Sales

        "sales": "Sales",
        "revenue": "Sales",
        "totalsales": "Sales",
        "amount": "Sales",
        "weekly_sales": "Sales",
        "weeklysales": "Sales",

        # Category

        "category": "Category",
        "productcategory": "Category",
        "family": "Category",

        # Region / Store

        "region": "Region",
        "store": "Region",
        "storenbr": "Region",
        "storeid": "Region",

        # Product

        "productname": "Product Name",
        "product": "Product Name",
        "item": "Product Name",

        # Other columns

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

    required_columns = [
        "Order Date",
        "Sales"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Date conversion

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

    # Sales conversion

    df["Sales"] = (
        df["Sales"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("₹", "", regex=False)
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

    df["Month Name"] = (
        df["Order Date"].dt.month_name()
    )

    df["Quarter"] = (
        df["Order Date"].dt.quarter
    )

    df["Year-Month"] = (
        df["Order Date"].dt.to_period("M").astype(str)
    )

    return df.sort_values("Order Date")


# ============================================================
# 6. DATA UPLOAD
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

        with st.expander(
            "📁 Check Dataset Information"
        ):

            st.write("Original Columns")

            st.write(
                raw_data.columns.tolist()
            )

            st.write("Processed Columns")

            st.write(
                df.columns.tolist()
            )

            st.write(
                "Dataset Shape:",
                df.shape
            )

            st.dataframe(
                df.head(),
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Dataset loading error: {e}"
        )

        st.stop()

else:

    st.title(
        "📊 Sales Intelligence Dashboard"
    )

    st.warning(
        "Please upload your sales dataset using the sidebar."
    )

    st.info(
        "Required columns: Order Date and Sales."
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

if isinstance(date_range, (tuple, list)):

    if len(date_range) == 2:

        filtered_df = filtered_df[
            filtered_df["Order Date"].dt.date.between(
                date_range[0],
                date_range[1]
            )
        ]

    elif len(date_range) == 1:

        filtered_df = filtered_df[
            filtered_df["Order Date"].dt.date == date_range[0]
        ]

regions = sorted(
    filtered_df["Region"].unique()
)

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

filtered_df = filtered_df[
    filtered_df["Region"].isin(selected_regions)
]

categories = sorted(
    filtered_df["Category"].unique()
)

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
        "No records found for the selected filters."
    )

    st.stop()


# ============================================================
# 8. COMMON FUNCTIONS
# ============================================================

def format_currency(value):

    return f"${value:,.2f}"


def monthly_sales(data):

    monthly = (
        data.groupby("Year-Month")["Sales"]
        .sum()
        .reset_index()
    )

    monthly["Order Date"] = pd.to_datetime(
        monthly["Year-Month"]
    )

    return monthly.sort_values("Order Date")


def prepare_time_series(data):

    monthly = monthly_sales(data)

    ts = monthly.set_index(
        "Order Date"
    )["Sales"]

    ts = ts.asfreq("MS").fillna(0)

    return ts


def make_sarima_model(series):

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(
        disp=False,
        maxiter=200
    )


def generate_forecast(data, periods):

    ts = prepare_time_series(data)

    if len(ts) < 24:

        raise ValueError(
            "At least 24 monthly observations are recommended."
        )

    model = make_sarima_model(ts)

    forecast_result = model.get_forecast(
        steps=periods
    )

    forecast = forecast_result.predicted_mean

    confidence = forecast_result.conf_int()

    return (
        ts,
        forecast,
        confidence,
        model
    )


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


def create_forecast_chart(
    historical,
    forecast,
    confidence
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical.values,
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

            line=dict(
                color="rgba(255,255,255,0)"
            ),

            name="Confidence Interval"
        )
    )

    return plot_layout(
        fig,
        "Historical vs Forecasted Sales"
    )


# ============================================================
# 9. MODULE 1: EXECUTIVE DASHBOARD
# ============================================================

if menu == "Executive Dashboard":

    st.title(
        "📊 Sales Intelligence Dashboard"
    )

    st.markdown(
        "### Business Performance Overview"
    )

    st.caption(
        "Monitor revenue, orders, products, "
        "categories and regional performance."
    )

    total_sales = filtered_df["Sales"].sum()

    valid_orders = filtered_df[
        filtered_df["Order ID"] != "Unknown"
    ]

    if not valid_orders.empty:

        total_orders = valid_orders["Order ID"].nunique()

    else:

        total_orders = len(filtered_df)

    avg_sales = filtered_df["Sales"].mean()

    total_products = (
        filtered_df["Product Name"].nunique()
    )

    # KPI Cards

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

    # Monthly Trend

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

    # Category Sales

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

    # Region Sales

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

    # Top Products

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

    # Monthly Category Comparison

    st.subheader(
        "Monthly Category Performance"
    )

    category_monthly = (
        filtered_df.groupby(
            ["Year-Month", "Category"]
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        category_monthly,
        x="Year-Month",
        y="Sales",
        color="Category",
        markers=True
    )

    st.plotly_chart(
        plot_layout(fig),
        use_container_width=True
    )

    # Transactions

    st.subheader(
        "Recent Transactions"
    )

    st.dataframe(
        filtered_df.sort_values(
            "Order Date",
            ascending=False
        ).head(20),
        use_container_width=True
    )


# ============================================================
# 10. MODULE 2: DATA LOADING & EDA
# ============================================================

elif menu == "Data Loading & EDA":

    st.title(
        "📁 Data Loading & Exploratory Data Analysis"
    )

    st.markdown(
        "### Explore dataset structure, quality and sales patterns."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Rows",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Total Columns",
        filtered_df.shape[1]
    )

    c3.metric(
        "Missing Values",
        filtered_df.isnull().sum().sum()
    )

    c4.metric(
        "Duplicate Records",
        filtered_df.duplicated().sum()
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dataset Preview",
            "Data Quality",
            "Sales Analysis",
            "Correlation Analysis"
        ]
    )

    # Dataset Preview

    with tab1:

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
        )

        st.subheader(
            "Statistical Summary"
        )

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

    # Data Quality

    with tab2:

        st.subheader(
            "Missing Value Analysis"
        )

        missing = pd.DataFrame({

            "Column": filtered_df.columns,

            "Missing Values":
                filtered_df.isnull().sum().values,

            "Missing Percentage": (
                filtered_df.isnull().mean().values * 100
            ).round(2)

        })

        st.dataframe(
            missing,
            use_container_width=True
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

        st.subheader(
            "Duplicate Records"
        )

        duplicate_data = filtered_df[
            filtered_df.duplicated(keep=False)
        ]

        st.dataframe(
            duplicate_data.head(100),
            use_container_width=True
        )

    # Sales Analysis

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
            .sort_values(
                "Sales",
                ascending=False
            )
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

        # Region sales

        region_sales = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            region_sales,
            names="Region",
            values="Sales",
            title="Regional Revenue Distribution"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

    # Correlation Analysis

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
                "At least two numeric columns are required."
            )


# ============================================================
# 11. MODULE 3: TIME SERIES ANALYSIS
# ============================================================

elif menu == "Time Series Analysis":

    st.title(
        "📈 Time Series Analysis"
    )

    st.markdown(
        "### Analyze sales trends, seasonality and autocorrelation."
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

    # Monthly Sales

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

    # Rolling Average

    st.subheader(
        "Rolling Average Analysis"
    )

    window = st.slider(
        "Select Rolling Window",
        2,
        min(12, max(2, len(ts))),
        min(3, max(2, len(ts))),
        key="rolling_window"
    )

    rolling = pd.DataFrame({

        "Actual Sales": ts,

        "Rolling Average": ts.rolling(
            window=window
        ).mean()

    })

    fig = px.line(
        rolling,
        title="Actual Sales vs Rolling Average"
    )

    st.plotly_chart(
        plot_layout(fig),
        use_container_width=True
    )

    # Seasonal Decomposition

    st.subheader(
        "Seasonal Decomposition"
    )

    if len(ts) >= 24:

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

            for col in components.columns:

                fig = px.line(
                    components.reset_index(),
                    x="Order Date",
                    y=col,
                    title=f"{col} Component"
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                f"Decomposition error: {e}"
            )

    else:

        st.warning(
            "At least 24 monthly observations are recommended "
            "for seasonal decomposition."
        )

    # ACF and PACF

    st.subheader(
        "Autocorrelation Analysis"
    )

    if len(ts) >= 5:

        max_lags = min(
            24,
            len(ts) // 2 - 1
        )

        if max_lags >= 1:

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

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

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

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
            "More observations are required for ACF/PACF."
        )


# ============================================================
# 12. MODULE 4: SALES FORECASTING
# ============================================================

elif menu == "Sales Forecasting":

    st.title(
        "🔮 Sales Forecasting Model"
    )

    st.markdown(
        "### Predict future sales using seasonal SARIMAX."
    )

    ts = prepare_time_series(filtered_df)

    st.metric(
        "Available Historical Months",
        len(ts)
    )

    periods = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=12,
        value=3
    )

    if len(ts) < 24:

        st.error(
            "At least 24 monthly observations are recommended "
            "for seasonal SARIMAX."
        )

    else:

        if st.button(
            "🚀 Generate Sales Forecast",
            use_container_width=True
        ):

            with st.spinner(
                "Training SARIMAX model..."
            ):

                try:

                    # Chronological holdout evaluation

                    test_size = min(
                        3,
                        max(1, len(ts) // 5)
                    )

                    train = ts.iloc[:-test_size]
                    test = ts.iloc[-test_size:]

                    evaluation = None

                    if len(train) >= 24:

                        evaluation_model = (
                            make_sarima_model(train)
                        )

                        test_predictions = (
                            evaluation_model.get_forecast(
                                steps=len(test)
                            ).predicted_mean
                        )

                        mae = mean_absolute_error(
                            test.values,
                            test_predictions.values
                        )

                        rmse = np.sqrt(
                            mean_squared_error(
                                test.values,
                                test_predictions.values
                            )
                        )

                        evaluation = {
                            "MAE": mae,
                            "RMSE": rmse
                        }

                    # Final model using all available data

                    model = make_sarima_model(ts)

                    forecast_result = model.get_forecast(
                        steps=periods
                    )

                    forecast = (
                        forecast_result.predicted_mean
                    )

                    confidence = (
                        forecast_result.conf_int()
                    )

                    st.session_state["sales_forecast"] = (
                        ts,
                        forecast,
                        confidence
                    )

                    st.session_state["sales_evaluation"] = (
                        evaluation
                    )

                    st.success(
                        "Sales forecast generated successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Forecasting error: {e}"
                    )

        # Display results

        if "sales_forecast" in st.session_state:

            historical, forecast, confidence = (
                st.session_state["sales_forecast"]
            )

            evaluation = st.session_state.get(
                "sales_evaluation"
            )

            st.subheader(
                "Model Evaluation"
            )

            if evaluation is not None:

                c1, c2 = st.columns(2)

                c1.metric(
                    "Mean Absolute Error",
                    format_currency(evaluation["MAE"])
                )

                c2.metric(
                    "Root Mean Squared Error",
                    format_currency(evaluation["RMSE"])
                )

            else:

                st.info(
                    "Forecast generated. A separate holdout "
                    "evaluation needs at least 24 training months "
                    "after reserving test observations."
                )

            # Forecast chart

            fig = create_forecast_chart(
                historical,
                forecast,
                confidence
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Forecast table

            forecast_df = pd.DataFrame({

                "Forecast Date": forecast.index,

                "Forecasted Sales": forecast.values,

                "Lower Confidence": confidence.iloc[:, 0].values,

                "Upper Confidence": confidence.iloc[:, 1].values

            })

            st.subheader(
                "Forecast Results"
            )

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            st.metric(
                "Total Forecasted Revenue",
                format_currency(forecast.sum())
            )

            download_csv(
                forecast_df,
                "sales_forecast.csv"
            )

            st.caption(
                "Forecasts are estimates and confidence intervals "
                "represent model uncertainty."
            )


# ============================================================
# 13. MODULE 5: CATEGORY & REGION FORECASTING
# ============================================================

elif menu == "Category & Region Forecasting":

    st.title(
        "📊 Category & Region Forecasting"
    )

    st.markdown(
        "### Predict future sales for individual business segments."
    )

    dimension = st.selectbox(
        "Select Forecasting Dimension",
        ["Category", "Region"]
    )

    available_groups = sorted(
        filtered_df[dimension].unique()
    )

    selected_groups = st.multiselect(
        f"Select {dimension}",
        available_groups,
        default=available_groups[
            :min(5, len(available_groups))
        ]
    )

    periods = st.slider(
        "Forecast Period (Months)",
        1,
        12,
        3,
        key="segment_periods"
    )

    if st.button(
        "Generate Segment Forecast",
        use_container_width=True
    ):

        results = []

        progress = st.progress(0)

        if not selected_groups:

            st.warning(
                "Please select at least one segment."
            )

        else:

            for i, group in enumerate(selected_groups):

                group_data = filtered_df[
                    filtered_df[dimension] == group
                ]

                group_ts = prepare_time_series(
                    group_data
                )

                if len(group_ts) < 24:

                    st.warning(
                        f"{group}: Insufficient monthly history."
                    )

                    progress.progress(
                        (i + 1) / len(selected_groups)
                    )

                    continue

                try:

                    model = make_sarima_model(
                        group_ts
                    )

                    predictions = model.get_forecast(
                        steps=periods
                    ).predicted_mean

                    for date, value in predictions.items():

                        results.append({

                            dimension: group,

                            "Forecast Date": date,

                            "Forecasted Sales": value

                        })

                except Exception as e:

                    st.warning(
                        f"Forecast failed for {group}: {e}"
                    )

                progress.progress(
                    (i + 1) / len(selected_groups)
                )

            if results:

                result_df = pd.DataFrame(
                    results
                )

                st.session_state["segment_forecast"] = (
                    result_df
                )

                st.success(
                    "Segment forecasting completed!"
                )

            else:

                st.warning(
                    "No forecasts were generated."
                )

    # Display segment forecast

    if "segment_forecast" in st.session_state:

        result_df = st.session_state[
            "segment_forecast"
        ]

        st.subheader(
            "Category / Region Forecast Results"
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        fig = px.line(
            result_df,
            x="Forecast Date",
            y="Forecasted Sales",
            color=dimension,
            markers=True,
            title=f"{dimension}-wise Sales Forecast"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        comparison = (
            result_df.groupby(dimension)
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
            x=dimension,
            y="Forecasted Sales",
            color=dimension,
            title="Forecasted Revenue Comparison"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        download_csv(
            result_df,
            "segment_forecast.csv"
        )


# ============================================================
# 14. MODULE 6: ANOMALY DETECTION
# ============================================================

elif menu == "Anomaly Detection":

    st.title(
        "🚨 Sales Anomaly Detection"
    )

    st.markdown(
        "### Detect unusual sales transactions using Isolation Forest."
    )

    contamination = st.slider(
        "Expected Anomaly Percentage",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    if st.button(
        "🔍 Detect Sales Anomalies",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing transactions..."
        ):

            try:

                anomaly_data = filtered_df.copy()

                # Feature engineering

                features = anomaly_data[
                    [
                        "Sales",
                        "Month",
                        "Quarter"
                    ]
                ].copy()

                if len(features) < 10:

                    st.error(
                        "At least 10 transactions are recommended."
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

    # Results

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
            hover_data=[
                "Category",
                "Region",
                "Product Name",
                "Anomaly Score"
            ],
            title="Normal vs Anomalous Sales"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        # Anomaly Distribution

        anomaly_summary = (
            anomaly_data["Anomaly"]
            .value_counts()
            .reset_index()
        )

        anomaly_summary.columns = [
            "Status",
            "Count"
        ]

        fig = px.pie(
            anomaly_summary,
            names="Status",
            values="Count",
            hole=0.5,
            title="Anomaly Distribution"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        # Detected records

        st.subheader(
            "Detected Anomalous Transactions"
        )

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

        st.caption(
            "Anomaly detection identifies unusual statistical patterns; "
            "an anomaly is not necessarily an error or fraud."
        )


# ============================================================
# 15. MODULE 7: PRODUCT DEMAND SEGMENTATION
# ============================================================

elif menu == "Product Demand Segmentation":

    st.title(
        "🎯 Product Demand Segmentation"
    )

    st.markdown(
        "### Classify products using K-Means Clustering."
    )

    n_clusters = st.slider(
        "Number of Product Segments",
        min_value=2,
        max_value=8,
        value=4
    )

    if st.button(
        "🚀 Run Product Segmentation",
        use_container_width=True
    ):

        with st.spinner(
            "Performing K-Means clustering..."
        ):

            try:

                # Aggregate product-level features

                product_data = (
                    filtered_df.groupby("Product Name")
                    .agg(

                        Total_Sales=(
                            "Sales",
                            "sum"
                        ),

                        Average_Sales=(
                            "Sales",
                            "mean"
                        ),

                        Transaction_Count=(
                            "Sales",
                            "count"
                        ),

                        Active_Months=(
                            "Year-Month",
                            "nunique"
                        )

                    )
                    .reset_index()
                )

                product_data = product_data.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).dropna()

                if len(product_data) < n_clusters:

                    st.error(
                        "Number of products must be greater "
                        "than or equal to the selected clusters."
                    )

                else:

                    # Clustering features

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

                    # Rank clusters by average total revenue

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
                        product_data["Cluster"].map(
                            labels
                        )
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

    # Display results

    if "product_segments" in st.session_state:

        product_data = st.session_state[
            "product_segments"
        ]

        cluster_summary = (
            product_data.groupby("Segment")
            .agg(

                Total_Products=(
                    "Product Name",
                    "count"
                ),

                Total_Sales=(
                    "Total_Sales",
                    "sum"
                ),

                Average_Sales=(
                    "Average_Sales",
                    "mean"
                ),

                Total_Transactions=(
                    "Transaction_Count",
                    "sum"
                )

            )
            .reset_index()
        )

        st.markdown("---")

        # KPI Cards

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Products",
            f"{len(product_data):,}"
        )

        c2.metric(
            "Total Segments",
            product_data["Segment"].nunique()
        )

        c3.metric(
            "Total Product Revenue",
            format_currency(
                product_data["Total_Sales"].sum()
            )
        )

        st.markdown("---")

        left, right = st.columns(2)

        # Scatter Plot

        with left:

            fig = px.scatter(
                product_data,
                x="Total_Sales",
                y="Average_Sales",
                color="Segment",
                size="Transaction_Count",
                hover_name="Product Name",
                title="Product Demand Segmentation"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        # Revenue by Segment

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

        # Product Count

        fig = px.pie(
            cluster_summary,
            names="Segment",
            values="Total_Products",
            hole=0.5,
            title="Product Distribution by Segment"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True
        )

        # Summary

        st.subheader(
            "Product Segment Summary"
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        # Product-level Results

        st.subheader(
            "Product-level Classification"
        )

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

        download_csv(
            cluster_summary,
            "product_segment_summary.csv"
        )


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        padding:15px;
    ">

        <h4 style="color:#38bdf8;">
            📊 Sales Forecasting & Intelligence Dashboard
        </h4>

        <p>
            Built using Python | Streamlit | Pandas |
            NumPy | Plotly | Scikit-learn | Statsmodels
        </p>

        <p>
            Data Analytics • Machine Learning •
            Time Series Forecasting
        </p>

    </div>
    """,
    unsafe_allow_html=True
)
