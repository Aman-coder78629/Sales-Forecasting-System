

# ============================================================
# SALES FORECASTING & INTELLIGENCE DASHBOARD
# Based on all 6 Jupyter Notebooks
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
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


# ============================================================
# 4. DATA LOADING AND DATA PREPROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):

    file_name = file_name.lower()

    # Excel files
    if file_name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(file_bytes))

    elif file_name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))

    # ZIP files
    elif file_name.endswith(".zip"):

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:

            csv_files = [
                f for f in z.namelist()
                if f.lower().endswith(".csv")
                and not f.startswith("__MACOSX/")
            ]

            if not csv_files:
                raise ValueError(
                    "ZIP file ke andar koi CSV file nahi mili."
                )

            with z.open(csv_files[0]) as csv_file:

                df = pd.read_csv(csv_file)

    # Normal CSV files
    elif file_name.endswith(".csv"):

        df = pd.read_csv(io.BytesIO(file_bytes))

    else:

        raise ValueError(
            "Unsupported file format. Upload CSV, ZIP or Excel."
        )

    return df


def preprocess_data(data):

    df = data.copy()

    # --------------------------------------------------------
    # STEP 1: CLEAN COLUMN NAMES
    # --------------------------------------------------------

    df.columns = [
        re.sub(r"[^a-z0-9]", "", str(col).lower())
        for col in df.columns
    ]

    # --------------------------------------------------------
    # STEP 2: AUTOMATIC COLUMN MAPPING
    # --------------------------------------------------------

    column_mapping = {

        # Date columns
        "orderdate": "Order Date",
        "date": "Order Date",
        "salesdate": "Order Date",
        "transactiondate": "Order Date",

        # Sales columns
        "sales": "Sales",
        "revenue": "Sales",
        "totalsales": "Sales",
        "amount": "Sales",

        # Category columns
        "category": "Category",
        "productcategory": "Category",
        "family": "Category",

        # Region / Store columns
        "region": "Region",
        "store": "Region",
        "storenbr": "Region",
        "storeid": "Region",

        # Product columns
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

    # --------------------------------------------------------
    # STEP 3: VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    if "Order Date" not in df.columns:

        raise ValueError(
            "Date column not found. Available columns: "
            + str(df.columns.tolist())
        )

    if "Sales" not in df.columns:

        raise ValueError(
            "Sales column not found. Available columns: "
            + str(df.columns.tolist())
        )

    # --------------------------------------------------------
    # STEP 4: DATE CONVERSION
    # --------------------------------------------------------

    dates = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    # Try day-first parsing if most dates failed
    if dates.isna().mean() > 0.5:

        dates = pd.to_datetime(
            df["Order Date"],
            errors="coerce",
            dayfirst=True
        )

    df["Order Date"] = dates

    # --------------------------------------------------------
    # STEP 5: SALES CONVERSION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # STEP 6: REMOVE INVALID RECORDS
    # --------------------------------------------------------

    df.dropna(
        subset=["Order Date", "Sales"],
        inplace=True
    )

    if df.empty:

        raise ValueError(
            "Date aur Sales conversion ke baad dataset empty hai. "
            "Please check your dataset values."
        )

    # --------------------------------------------------------
    # STEP 7: OPTIONAL COLUMN HANDLING
    # --------------------------------------------------------

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

    # If Product Name is missing but Category exists,
    # use Category as a fallback for product-level display.
    if "Product Name" not in df.columns:

        if "Category" in df.columns:

            df["Product Name"] = df["Category"]

    for col in optional_columns:

        if col not in df.columns:

            df[col] = "Unknown"

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    # --------------------------------------------------------
    # STEP 8: FEATURE ENGINEERING
    # --------------------------------------------------------

    df["Year"] = df["Order Date"].dt.year

    df["Month"] = df["Order Date"].dt.month

    df["Month Name"] = (
        df["Order Date"].dt.month_name()
    )

    df["Quarter"] = (
        df["Order Date"].dt.quarter
    )

    return df.sort_values("Order Date")



# ============================================================
# 5. UPLOAD DATASET
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Superstore Dataset",
    type=["csv", "zip", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:

        raw_data = load_data(
            uploaded_file.getvalue(),
            uploaded_file.name
        )

        df = preprocess_data(raw_data)
        st.sidebar.success("Dataset loaded successfully!")

with st.expander("Check Dataset Information"):

    st.write("Original Columns:")
    st.write(raw_data.columns.tolist())

    st.write("Processed Columns:")
    st.write(df.columns.tolist())

    st.write("Dataset Shape:", df.shape)

    st.dataframe(df.head())
        

        st.sidebar.success("Dataset loaded successfully!")

    except Exception as e:

        st.error(f"Dataset loading error: {e}")
        st.stop()

else:

    st.warning(
        "Please upload your train.csv.zip dataset "
        "using the sidebar to start the dashboard."
    )

    st.info(
        "Required columns: Order Date and Sales. "
        "Category, Region, Sub-Category and Product Name "
        "are required for their respective detailed analyses."
    )

    st.stop()


# ============================================================
# 6. GLOBAL FILTERS
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

regions = sorted(filtered_df["Region"].unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions
)

if selected_regions:

    filtered_df = filtered_df[
        filtered_df["Region"].isin(selected_regions)
    ]

categories = sorted(filtered_df["Category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

if selected_categories:

    filtered_df = filtered_df[
        filtered_df["Category"].isin(selected_categories)
    ]

if filtered_df.empty:

    st.warning("No records found for the selected filters.")
    st.stop()


# ============================================================
# 7. COMMON FUNCTIONS
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

    monthly.rename(
        columns={"Sales": "Sales"},
        inplace=True
    )

    return monthly


def format_currency(value):

    return f"${value:,.2f}"


def make_sarima_model(series):

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)

    return result


def forecast_sales(data, periods=3):

    monthly = monthly_sales(data)

    ts = monthly.set_index("Order Date")["Sales"]

    ts = ts.asfreq("ME").fillna(0)

    if len(ts) < 24:

        raise ValueError(
            "At least 24 monthly observations are recommended "
            "for the seasonal SARIMA model."
        )

    result = make_sarima_model(ts)

    forecast = result.get_forecast(
        steps=periods
    )

    forecast_values = forecast.predicted_mean

    confidence = forecast.conf_int()

    return (
        ts,
        forecast_values,
        confidence,
        result
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

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


# ============================================================
# 8. EXECUTIVE DASHBOARD
# ============================================================

if menu == "Executive Dashboard":

    st.title("📊 Sales Intelligence Dashboard")

    st.markdown(
        "### Business Performance Overview"
    )

    st.caption(
        "Monitor revenue, transactions, product categories "
        "and regional sales performance."
    )

    total_sales = filtered_df["Sales"].sum()

    total_orders = filtered_df["Order ID"].nunique()

    avg_sales = filtered_df["Sales"].mean()

    total_products = filtered_df["Product Name"].nunique()

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
# 9. DATA LOADING & EXPLORATORY DATA ANALYSIS
# ============================================================

elif menu == "Data Loading & EDA":

    st.title("📁 Data Loading & Exploratory Data Analysis")

    st.markdown(
        "### Understand the structure and quality of your sales data."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Rows", f"{len(filtered_df):,}")

    c2.metric("Total Columns", f"{filtered_df.shape[1]}")

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
            filtered_df.describe(include="all").T,
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
            "Missing Values": filtered_df.isnull().sum().values,
            "Missing Percentage": (
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

        fig = px.bar(
            filtered_df.groupby(
                "Sub-Category"
            )["Sales"].sum().reset_index().sort_values(
                "Sales",
                ascending=False
            ),
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
# 10. TIME SERIES ANALYSIS
# ============================================================

elif menu == "Time Series Analysis":

    st.title("📈 Time Series Analysis")

    st.markdown(
        "### Monthly sales trends, seasonality and decomposition."
    )

    monthly = monthly_sales(filtered_df)

    ts = monthly.set_index(
        "Order Date"
    )["Sales"]

    ts = ts.asfreq("ME").fillna(0)

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

    else:

        st.warning(
            "At least 24 monthly observations are needed "
            "for seasonal decomposition."
        )

    st.subheader("Autocorrelation Analysis")

    if len(ts) >= 5:

        max_lags = min(24, len(ts) // 2 - 1)

        fig, ax = plt.subplots(figsize=(12, 4))

        plot_acf(
            ts,
            lags=max_lags,
            ax=ax
        )

        ax.set_title("Autocorrelation Function (ACF)")

        st.pyplot(fig)

        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))

        plot_pacf(
            ts,
            lags=max_lags,
            ax=ax,
            method="ywm"
        )

        ax.set_title("Partial Autocorrelation Function (PACF)")

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# 11. SALES FORECASTING MODEL
# ============================================================

elif menu == "Sales Forecasting":

    st.title("🔮 Sales Forecasting Model")

    st.markdown(
        "### Predict future sales using seasonal SARIMAX."
    )

    monthly = monthly_sales(filtered_df)

    ts = monthly.set_index(
        "Order Date"
    )["Sales"].asfreq("ME").fillna(0)

    if len(ts) < 24:

        st.error(
            "The seasonal SARIMAX model requires at least "
    )
   
