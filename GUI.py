import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
import pickle
import re
import io
import base64
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes
try:
    import sys
    import os
    
    # Import your actual model class
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    import xgboost as xgb
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from scipy.stats import randint, uniform
    import pickle as pkl
    
    MODEL_AVAILABLE = True
    
    # --- KELAS PREDIKSI BARU (dari XGBOOST_FINAL_BANGET.ipynb) ---
    class SalesPrediction:
        """
        Model prediksi penjualan dengan XGBoost
        Menggunakan fitur: lag_1, lag_2, lag_3, lag_7, roll_mean_7, day_of_week
        """
        def __init__(self):
            self.HORIZON = 7  # Forecasting 7 hari ke depan
            self.LAGS = [1, 2, 3, 7]
            
            # ✅ BEST PARAMS langsung dari output RandomizedSearchCV notebook
            # (Best CV MAE: 264,283.98)
            self.xgb_params = dict(
                n_estimators=398,
                learning_rate=0.011747439807159633,
                max_depth=7,
                subsample=0.7221455441377573,
                colsample_bytree=0.9594216754108317,
                reg_alpha=0.9736638367553173,
                reg_lambda=1.2098024368436642,
                min_child_weight=9,
                gamma=0.30321452982979497,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
            
            self.model = None
            self.feature_cols = None
            self.y_series = None
            self.pos_map = None
            self.last_date = None
            
        def prepare_data(self, df):
            """
            Prepare data from uploaded file
            """
            df = df.copy()
            
            # Auto-detect date column
            date_col = None
            possible_date_names = ['tanggal', 'date', 'tgl', 'tanggal_penjualan', 'tanggal penjualan', 'sales_date', 'order_date']
            
            for col in df.columns:
                col_normalized = col.lower().replace('_', ' ').strip()
                if col_normalized in [name.replace('_', ' ') for name in possible_date_names]:
                    date_col = col
                    break
            
            if not date_col:
                for col in df.columns:
                    col_lower = col.lower().replace('_', ' ')
                    if any(date_name.replace('_', ' ') in col_lower for date_name in possible_date_names):
                        date_col = col
                        break
            
            if not date_col:
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    date_col = datetime_cols[0]
            
            if not date_col:
                available_cols = ", ".join(df.columns.tolist())
                raise KeyError(f"Kolom tanggal tidak ditemukan. Kolom yang tersedia: {available_cols}")
            
            # Convert date
            try:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            except:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                except:
                    raise ValueError(f"Tidak dapat mengkonversi kolom '{date_col}' menjadi format tanggal")
            
            df = df.rename(columns={date_col: 'tanggal_penjualan'})
            
            # Find sales column
            sales_col = None
            for col in df.columns:
                col_normalized = col.lower().replace('_', ' ').strip()
                if col_normalized == 'total penjualan':
                    sales_col = col
                    break
            
            if not sales_col:
                possible_sales_names = ['penjualan', 'total', 'sales', 'revenue', 'omzet', 'total_sales', 'total sales']
                for col in df.columns:
                    if col.lower().replace('_', ' ').strip() in possible_sales_names:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            sales_col = col
                            break
            
            if not sales_col:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sales_col = numeric_cols[-1]
                    st.info(f"ℹ️ Menggunakan kolom '{sales_col}' sebagai total penjualan")
            
            if sales_col != 'total_penjualan':
                df = df.rename(columns={sales_col: 'total_penjualan'})
            
            # Clean data
            df = df.dropna(subset=['tanggal_penjualan', 'total_penjualan'])
            df = df[df['tanggal_penjualan'].notna()]
            df = df.sort_values('tanggal_penjualan')
            df = df.reset_index(drop=True)
            
            # Set date as index
            df = df.set_index('tanggal_penjualan')
            
            return df
        
        def create_features(self, df):
            """
            Create features: lag_1, lag_2, lag_3, lag_7, roll_mean_7, day_of_week
            """
            df_feat = df.copy()
            TARGET_COL = 'total_penjualan'
            
            # Lag features
            for l in self.LAGS:
                df_feat[f'lag_{l}'] = df_feat[TARGET_COL].shift(l)
            
            # Rolling mean 7
            df_feat['roll_mean_7'] = df_feat[TARGET_COL].rolling(7).mean()
            
            # Day of week (0=Monday, 6=Sunday)
            df_feat['day_of_week'] = df_feat.index.dayofweek
            
            # Target (next day)
            df_feat['y_t+1'] = df_feat[TARGET_COL].shift(-1)
            
            # Drop rows with NaN
            df_feat = df_feat.dropna()
            
            return df_feat
        
        def train(self, df):
            """
            Train XGBoost model
            """
            # Prepare data
            df = self.prepare_data(df)
            df_feat = self.create_features(df)
            
            # Feature columns
            self.feature_cols = [f'lag_{l}' for l in self.LAGS] + ['roll_mean_7', 'day_of_week']
            
            # Split train/test (80/20 split - SESUAI COLAB!)
            split_idx = int(len(df_feat) * 0.8)
            train_data = df_feat.iloc[:split_idx].copy()
            test_data = df_feat.iloc[split_idx:].copy()
            
            split_date = test_data.index[0]  # batas train/test
            
            X_train = train_data[self.feature_cols].values
            y_train = train_data['y_t+1'].values
            
            # ✅ Langsung pakai best params dari notebook — hasil IDENTIK
            self.model = XGBRegressor(**self.xgb_params)
            self.model.fit(X_train, y_train)
            
            # Store data for forecasting
            self.y_series = df['total_penjualan'].values
            self.pos_map = {t: i for i, t in enumerate(df.index)}
            self.last_date = df.index[-1]
            
            return df, df_feat, split_date, test_data
        
        def make_features_from_history(self, history, t_future):
            """
            Create features from history for prediction
            Args:
                history: list of past sales values
                t_future: timestamp for future prediction
            Returns:
                features array
            """
            feats = [
                history[-1],  # lag_1
                history[-2],  # lag_2
                history[-3],  # lag_3
                history[-7],  # lag_7
                float(np.mean(history[-7:])),  # roll_mean_7
                int(t_future.dayofweek)  # day_of_week
            ]
            return np.array(feats).reshape(1, -1)
        
        def recursive_forecast_7days(self, start_tstamp):
            """
            Forecast 7 days ahead using recursive strategy
            """
            i = self.pos_map[start_tstamp]
            history = list(self.y_series[:i+1])
            preds = []
            
            for s in range(1, self.HORIZON + 1):
                t_future = start_tstamp + pd.Timedelta(days=s)
                Xf = self.make_features_from_history(history, t_future)
                yhat = self.model.predict(Xf)[0]
                preds.append(float(yhat))
                history.append(float(yhat))  # Autoregressive
            
            return np.array(preds)
        
        def forecast_future_7days(self):
            """
            Forecast 7 days from last date in dataset
            """
            if self.model is None:
                raise ValueError("Model belum dilatih. Panggil train() terlebih dahulu")
            
            forecast_start = self.last_date
            future_preds = self.recursive_forecast_7days(forecast_start)
            
            # Create dates for forecast
            forecast_dates = pd.date_range(
                start=forecast_start + pd.Timedelta(days=1),
                periods=self.HORIZON,
                freq='D'
            )
            
            return forecast_dates, future_preds

except ImportError as e:
    print(f"⚠️ Model dependencies not available: {e}")
    MODEL_AVAILABLE = False
    
    class SalesPrediction:
        def __init__(self):
            pass
        
        def train(self, df):
            raise NotImplementedError("XGBoost tidak tersedia")
        
        def forecast_future_7days(self):
            raise NotImplementedError("XGBoost tidak tersedia")

# Custom CSS (SAMA SEPERTI ASLI)
st.set_page_config(
    page_title="Pawonkoe - Sales Prediction & Sentiment Analysis",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* ===== FORCE LIGHT MODE + BACKGROUND HIJAU PASTEL ===== */
    html, body, [class*="css"], .stApp {
        background-color: #D6EDD9 !important;
        color: #1a1a1a !important;
    }

    /* Main content block */
    .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"] {
        background-color: #D6EDD9 !important;
    }

    /* Sidebar hijau lebih dalam sedikit */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #B8DCC0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }

    /* ===== HEADER ===== */
    .main-header {
        font-size: 2.5rem;
        color: #E63946 !important;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #E63946;
        padding-bottom: 1rem;
        background-color: transparent !important;
    }

    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        margin: 0.5rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white !important;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #f0f0f0 !important;
        margin-top: 0.5rem;
    }

    /* ===== SIDEBAR SECTION ===== */
    .sidebar-section {
        background-color: #d0e8d5 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #E63946;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #b8dcc0 !important;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        color: #1a1a1a !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E63946 !important;
        color: white !important;
    }
    [data-testid="stTabContent"] {
        background-color: #D6EDD9 !important;
    }

    /* ===== INPUT KUNING PASTEL ===== */
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #FFFACD !important;
        border: 1.5px solid #D4C84A !important;
        border-radius: 6px !important;
        color: #1a1a1a !important;
    }
    /* Selectbox */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div {
        background-color: #FFFACD !important;
        border: 1.5px solid #D4C84A !important;
        color: #1a1a1a !important;
    }
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #FFFACD !important;
        border: 1.5px solid #D4C84A !important;
    }
    /* Textarea */
    .stTextArea > div > div > textarea {
        background-color: #FFFACD !important;
        border: 1.5px solid #D4C84A !important;
        color: #1a1a1a !important;
    }
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #FFFACD !important;
        border: 1.5px solid #D4C84A !important;
        color: #1a1a1a !important;
    }
    /* File uploader */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section {
        background-color: #FFFACD !important;
        border: 2px dashed #D4C84A !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #1a1a1a !important;
    }
    /* Checkbox */
    .stCheckbox > label {
        color: #1a1a1a !important;
    }

    /* ===== DATAFRAME / TABLE ===== */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border-radius: 8px;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] > div:first-child {
        background-color: #c8e6cc !important;
        color: #1a1a1a !important;
        border-radius: 6px !important;
    }

    /* ===== METRIC WIDGET NATIVE ===== */
    [data-testid="stMetric"] {
        background-color: #c8e6cc !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricValue"] > div {
        color: #1a1a1a !important;
    }

    /* ===== SUCCESS / INFO / WARNING BOXES ===== */
    [data-testid="stAlert"] {
        background-color: #e8f5e9 !important;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: #E63946 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #c1121f !important;
    }

    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }

    /* ===== TABEL / DATAFRAME - KUNING SOFT ===== */
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] iframe {
        background-color: #FEFCE8 !important;
        border-radius: 8px !important;
        border: 1px solid #EAD94C !important;
    }
    /* Header tabel */
    [data-testid="stDataFrame"] th,
    .dvn-scroller th {
        background-color: #FFF176 !important;
        color: #1a1a1a !important;
        font-weight: bold !important;
    }
    /* Row tabel */
    [data-testid="stDataFrame"] td,
    .dvn-scroller td {
        background-color: #FEFCE8 !important;
        color: #1a1a1a !important;
    }
    /* Row hover */
    [data-testid="stDataFrame"] tr:hover td {
        background-color: #FFF9C4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DASHBOARD CLASS ====================
class StreamlitDashboard:
    def __init__(self):
        self.sales_data = None
        self.sentiment_data = None
        self.predictor = SalesPrediction() if MODEL_AVAILABLE else None
        
    def load_data(self, uploaded_file, data_type):
        """Load CSV or Excel file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ {data_type} berhasil dimuat: {len(df)} baris")
            return df
        except Exception as e:
            st.error(f"❌ Error loading {data_type}: {str(e)}")
            return None
    
    def create_dashboard_metrics(self, sales_df, sentiment_df=None):
        """Create metrics section dengan style card dari peruin.py"""
        if sales_df is None:
            return
        
        # Detect sales column
        sales_col = None
        for col in sales_df.columns:
            col_normalized = col.lower().replace('_', ' ').strip()
            if 'total' in col_normalized and 'penjualan' in col_normalized:
                sales_col = col
                break
        if not sales_col:
            numeric_cols = sales_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sales_col = numeric_cols[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_records = len(sales_df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_records:,}</div>
                <div class="metric-label">📅 Total Records Penjualan</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if sentiment_df is not None:
                sentiment_count = len(sentiment_df)
                icon = "💬"
                display_sent = f"{sentiment_count:,}"
                label_sent = "Data Sentimen"
            else:
                icon = "⏳"
                display_sent = "-"
                label_sent = "Sentimen Belum Upload"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{icon} {display_sent}</div>
                <div class="metric-label">{label_sent}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("&nbsp;", unsafe_allow_html=True)
    
    def create_sales_forecast_visualization(self, sales_df):
        """
        UPDATED: Visualisasi forecasting 7 hari menggunakan model XGBoost baru
        """
        if sales_df is None:
            st.info("📥 Upload data penjualan untuk melihat prediksi")
            return
        
        st.subheader("📈 Prediksi Penjualan 7 Hari Ke Depan")
        
        if not MODEL_AVAILABLE:
            st.warning("⚠️ XGBoost tidak tersedia. Install dengan: pip install xgboost")
            return
        
        try:
            with st.spinner("🔄 Melatih model XGBoost..."):
                # Langsung retrain — TIDAK pakai cache agar tidak ada model lama tersimpan
                # random_state=42 menjamin hasil selalu sama
                predictor_obj = SalesPrediction()
                df, df_feat, split_date, test_data = predictor_obj.train(sales_df)
                forecast_dates, forecast_values = predictor_obj.forecast_future_7days()
                self.predictor = predictor_obj
                
                # IMPORTANT: Ikuti persis logika Colab!
                # 1. Actual zoom: last 120 days dari df ASLI (sesuai notebook: df.iloc[-120:])
                window_start = df.index.max() - pd.Timedelta(days=119)
                actual_zoom = df.loc[window_start:, 'total_penjualan']
                
                # 2. Predicted Test: 1-step ahead prediction
                X_test = test_data[self.predictor.feature_cols].values
                y_pred_test = self.predictor.model.predict(X_test)
                
                # IMPORTANT: pred_test_dates harus di-shift +1 untuk align dengan target (t+1)
                pred_test_dates = test_data.index + pd.Timedelta(days=1)
                pred_test_series = pd.Series(y_pred_test, index=pred_test_dates)
                
                # Filter pred_zoom untuk window yang sama dengan actual
                pred_zoom = pred_test_series.loc[pred_test_series.index >= window_start]
                
                # 3. Forecast 7 days
                forecast_df = pd.DataFrame({
                    'tanggal_penjualan': forecast_dates,
                    'total_penjualan': forecast_values
                })
                forecast_df = forecast_df.set_index('tanggal_penjualan')
                
                # ── Bangun grafik identik dengan notebook ──────────────
                # Notebook pakai: axvspan train/test, axvline split biru dotted,
                # actual biru solid, prediksi oranye dashed+marker, forecast hijau+marker
                fig = go.Figure()

                zmin = actual_zoom.index.min()
                zmax = forecast_df.index.max()

                # 1. Shading TRAIN AREA (biru muda, seperti axvspan notebook)
                fig.add_shape(type="rect",
                    x0=str(zmin.date()), x1=str(split_date.date()),
                    y0=0, y1=1, yref="paper", xref="x",
                    fillcolor="rgba(173,216,230,0.35)", line_width=0, layer="below")

                # 2. Shading TEST AREA (biru lebih muda)
                fig.add_shape(type="rect",
                    x0=str(split_date.date()), x1=str(zmax.date()),
                    y0=0, y1=1, yref="paper", xref="x",
                    fillcolor="rgba(173,216,230,0.15)", line_width=0, layer="below")

                # 3. Garis Split Train/Test — biru dotted tipis (persis notebook)
                split_date_str = split_date.strftime('%Y-%m-%d')
                fig.add_shape(type="line",
                    x0=split_date_str, x1=split_date_str,
                    y0=0, y1=1, yref="paper",
                    line=dict(color='steelblue', width=1.5, dash='dot'))

                # 4. Garis Actual — biru solid (label: "Actual")
                fig.add_trace(go.Scatter(
                    x=actual_zoom.index, y=actual_zoom.values,
                    mode='lines', name='Actual',
                    line=dict(color='#1f77b4', width=1.8)))

                # 5. Garis Prediksi Test — oranye dashed + marker kecil
                #    (label: "Prediksi (t+1) - Test", persis notebook)
                fig.add_trace(go.Scatter(
                    x=pred_zoom.index, y=pred_zoom.values,
                    mode='lines+markers', name='Prediksi (t+1) - Test',
                    line=dict(color='#ff7f0e', width=1.8, dash='dash'),
                    marker=dict(size=4, color='#ff7f0e')))

                # 6. Forecast 7 hari — hijau solid + marker (label: "Forecast Next 7 Days")
                fig.add_trace(go.Scatter(
                    x=forecast_df.index, y=forecast_df['total_penjualan'],
                    mode='lines+markers', name='Forecast Next 7 Days',
                    line=dict(color='#2ca02c', width=2.5),
                    marker=dict(size=8, color='#2ca02c')))

                # 7. Layout — sesuai gaya notebook (title, axis label, grid)
                fig.update_layout(
                    title=dict(text='Forecastig 7 Hari', x=0.5, xanchor='center',
                               font=dict(size=14)),
                    xaxis_title='Tanggal',
                    yaxis_title='total penjualan',
                    hovermode='x unified',
                    height=480,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=11),
                    legend=dict(
                        orientation="v", yanchor="top", y=0.99,
                        xanchor="left", x=0.01,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="gray", borderwidth=1),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray',
                               showline=True, linewidth=1, linecolor='black'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray',
                               showline=True, linewidth=1, linecolor='black'))

                # 8. Legend dummy untuk Train Area & Test Area & Split line
                #    (meniru legend box notebook)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color='rgba(173,216,230,0.6)',
                                symbol='square'),
                    name='Train Area', showlegend=True))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=10, color='rgba(173,216,230,0.25)',
                                symbol='square'),
                    name='Test Area', showlegend=True))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='steelblue', width=1.5, dash='dot'),
                    name='Split Train/Test', showlegend=True))

                st.plotly_chart(fig, use_container_width=True)
                
                # ── Ringkasan Prediksi 7 Hari (card style seperti peruin.py) ──
                st.markdown("### 🔮 Ringkasan Prediksi 7 Hari Ke Depan")
                
                avg_forecast  = np.mean(forecast_values)
                total_forecast = np.sum(forecast_values)
                max_forecast  = np.max(forecast_values)
                min_forecast  = np.min(forecast_values)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">Rp {avg_forecast:,.0f}</div>
                        <div class="metric-label">📊 Rata-rata Harian</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">Rp {total_forecast:,.0f}</div>
                        <div class="metric-label">💵 Total 7 Hari</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">Rp {max_forecast:,.0f}</div>
                        <div class="metric-label">📈 Prediksi Tertinggi</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">Rp {min_forecast:,.0f}</div>
                        <div class="metric-label">📉 Prediksi Terendah</div>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display forecast table (kuning soft via CSS)
                st.markdown("### 📋 Tabel Prediksi 7 Hari Ke Depan")
                
                forecast_table = pd.DataFrame({
                    'Tanggal': forecast_dates.strftime('%d-%m-%Y'),
                    'Hari': forecast_dates.strftime('%A'),
                    'Prediksi Penjualan': [f"Rp {val:,.0f}" for val in forecast_values]
                })
                
                st.dataframe(forecast_table, hide_index=True, use_container_width=True)
                
                # ── Helper metric ──────────────────────────────────────────
                def _rmse(yt, yp):
                    return np.sqrt(mean_squared_error(yt, yp))
                def _mape(yt, yp, eps=1e-9):
                    return np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100

                # ── Model performance: card style seperti peruin.py ─────────────
                st.markdown("### 📊 Performa Model pada Test Set")
                y_true_test = test_data['y_t+1'].values
                mae  = mean_absolute_error(y_true_test, y_pred_test)
                rmse = _rmse(y_true_test, y_pred_test)
                mape = _mape(y_true_test, y_pred_test)
                
                # Hitung R² score
                ss_res = np.sum((y_true_test - y_pred_test) ** 2)
                ss_tot = np.sum((y_true_test - np.mean(y_true_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">{r2:.4f}</div>
                        <div class="metric-label">🎯 R² Score (Test)</div>
                        <div style="color:#c8f7d4;font-size:0.85rem;margin-top:4px;">↑ {r2*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">Rp {mae:,.0f}</div>
                        <div class="metric-label">📉 MAE (1-step)</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">Rp {rmse:,.0f}</div>
                        <div class="metric-label">📐 RMSE (1-step)</div>
                    </div>""", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.6rem;">{mape:.2f}%</div>
                        <div class="metric-label">📏 MAPE (1-step)</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Evaluasi recursive 7-step PER HORIZON ──────────────────
                st.markdown("### 📊 Evaluasi Recursive 7-Step per Horizon")

                y_series_full = df['total_penjualan'].values
                idx_all_full  = df.index
                pos_map_full  = {t: i for i, t in enumerate(idx_all_full)}

                idx_test_all  = test_data.index
                HORIZON       = self.predictor.HORIZON
                max_i_allowed = len(df) - 1 - HORIZON
                idx_test_valid = [t for t in idx_test_all
                                  if pos_map_full[t] <= max_i_allowed]

                def _actual_future(tstamp):
                    i = pos_map_full[tstamp]
                    return y_series_full[i+1:i+1+HORIZON]

                y_true_7 = np.vstack([_actual_future(t) for t in idx_test_valid])
                y_pred_rec_7 = np.vstack([
                    self.predictor.recursive_forecast_7days(t) for t in idx_test_valid
                ])

                rows_eval = []
                for h in range(HORIZON):
                    rows_eval.append({
                        "Horizon": h + 1,
                        "MAE":    round(mean_absolute_error(y_true_7[:, h], y_pred_rec_7[:, h]), 2),
                        "RMSE":   round(_rmse(y_true_7[:, h], y_pred_rec_7[:, h]), 2),
                        "MAPE_%": round(_mape(y_true_7[:, h], y_pred_rec_7[:, h]), 4),
                    })
                eval_7_df = pd.DataFrame(rows_eval)
                st.dataframe(eval_7_df, hide_index=True, use_container_width=True)

                # Rata-rata keseluruhan horizon — card style
                mean_row = eval_7_df[["MAE","RMSE","MAPE_%"]].mean()
                
                st.markdown("#### 📈 Rata-rata Performa H1–H7")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">Rp {mean_row['MAE']:,.0f}</div>
                        <div class="metric-label">📉 Avg MAE (H1–H7)</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">Rp {mean_row['RMSE']:,.0f}</div>
                        <div class="metric-label">📐 Avg RMSE (H1–H7)</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size:1.4rem;">{mean_row['MAPE_%']:.2f}%</div>
                        <div class="metric-label">📏 Avg MAPE (H1–H7)</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Tabel Prediksi H1 (Test) + Error ───────────────────────
                st.markdown("### 📋 Tabel Prediksi H1 (Test) + Error")
                pred_test_dates_tbl = test_data.index + pd.Timedelta(days=1)
                pred_h1_df = pd.DataFrame({
                    "actual_y (t+1)": y_true_test,
                    "pred_y (t+1)":   y_pred_test,
                }, index=pred_test_dates_tbl)
                pred_h1_df.index.name = "date_target (t+1)"
                pred_h1_df["AE"]    = np.abs(pred_h1_df["actual_y (t+1)"] - pred_h1_df["pred_y (t+1)"])
                pred_h1_df["APE_%"] = (pred_h1_df["AE"] / (np.abs(pred_h1_df["actual_y (t+1)"]) + 1e-9)) * 100
                pred_h1_df = pred_h1_df.round(2)
                st.dataframe(pred_h1_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error dalam prediksi: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # ==================== SENTIMENT ANALYSIS (LENGKAP DARI PERUIN.PY) ====================
    def create_sentiment_analysis(self, sentiment_data):
        """Create comprehensive sentiment analysis dengan wordcloud"""
        st.subheader("💬 Analisis Sentimen Komprehensif")
        
        if sentiment_data is not None:
            # Find text column
            text_cols = sentiment_data.select_dtypes(include=['object']).columns
            
            if len(text_cols) == 0:
                st.error("Tidak ada kolom teks yang ditemukan")
                return
            
            # Auto-detect text column
            text_col = None
            for col in text_cols:
                if any(x in col.lower() for x in ['ulasan', 'review', 'text', 'komentar', 'comment']):
                    text_col = col
                    break
            
            if not text_col:
                text_col = st.selectbox("Pilih kolom teks untuk analisis:", text_cols)
            else:
                st.info(f"📝 Menggunakan kolom: **{text_col}**")
            
            texts = sentiment_data[text_col].dropna().astype(str).tolist()
            
            if len(texts) == 0:
                st.warning("Tidak ada teks yang valid untuk dianalisis")
                return
            
            # Affix removal option
            with st.expander("⚙️ Pengaturan Analisis"):
                remove_affixes_flag = st.checkbox("Hilangkan imbuhan (me-, di-, -kan, dll)", value=True,
                                                 help="Menghilangkan awalan dan akhiran Indonesia untuk analisis yang lebih baik")
            
            # Get stopwords
            indonesian_stopwords = self.get_comprehensive_indonesian_stopwords()
            
            # Simple sentiment analysis
            positive_keywords = ['bagus', 'enak', 'lezat', 'recommended', 'puas', 'mantap', 
                               'suka', 'senang', 'terbaik', 'sempurna', 'nikmat', 'juara',
                               'top', 'amazing', 'excellent', 'love', 'best', 'delicious']
            negative_keywords = ['buruk', 'mengecewakan', 'tidak suka', 'jelek', 'kecewa', 
                               'hambar', 'kurang', 'tidak enak', 'payah', 'rusak',
                               'bad', 'worst', 'terrible', 'hate', 'disappointed']
            
            sentiments = []
            sentiment_scores = []
            all_words = {'positive': [], 'neutral': [], 'negative': []}
            
            for text in texts:
                text_lower = text.lower()
                
                pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
                neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
                
                # Clean text for word collection
                clean_text = self.clean_text_for_wordcloud(text, indonesian_stopwords, remove_affixes_flag)
                words = clean_text.split()
                
                if pos_count > neg_count:
                    sentiments.append('positive')
                    sentiment_scores.append(0.7 + 0.2 * np.random.random())
                    all_words['positive'].extend(words)
                elif neg_count > pos_count:
                    sentiments.append('negative')
                    sentiment_scores.append(0.1 + 0.2 * np.random.random())
                    all_words['negative'].extend(words)
                else:
                    sentiments.append('neutral')
                    sentiment_scores.append(0.4 + 0.2 * np.random.random())
                    all_words['neutral'].extend(words)
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "😊 Positive", "😐 Neutral", "😞 Negative", "📈 Top Words"
            ])
            
            with tab1:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Sentiment distribution pie chart
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=[s.capitalize() for s in sentiment_counts.index],
                        title="Distribusi Sentimen Keseluruhan",
                        color_discrete_map={
                            'Positive': '#27AE60',
                            'Neutral': '#3498DB',
                            'Negative': '#E74C3C'
                        },
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.markdown("### 📈 Statistik Sentimen")
                    
                    total = len(sentiments)
                    positive_pct = (sentiments.count('positive') / total * 100)
                    neutral_pct = (sentiments.count('neutral') / total * 100)
                    negative_pct = (sentiments.count('negative') / total * 100)
                    
                    st.metric("Total Ulasan", f"{total:,}")
                    st.metric("Sentimen Positif", f"{positive_pct:.1f}%", 
                             f"{sentiments.count('positive')} ulasan")
                    st.metric("Sentimen Neutral", f"{neutral_pct:.1f}%",
                             f"{sentiments.count('neutral')} ulasan")
                    st.metric("Sentimen Negatif", f"{negative_pct:.1f}%",
                             f"{sentiments.count('negative')} ulasan")
                
                # Sentiment timeline
                if len(texts) > 10:
                    st.subheader("📅 Trend Sentimen dari Waktu ke Waktu")
                    
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.Timedelta(days=len(texts))
                    dates = pd.date_range(start=start_date, periods=len(texts), freq='D')
                    
                    timeline_df = pd.DataFrame({
                        'Tanggal': dates,
                        'Sentiment': sentiments,
                        'Score': sentiment_scores
                    })
                    
                    fig_timeline = go.Figure()
                    
                    for sentiment, color in [('positive', '#27AE60'), ('neutral', '#3498DB'), ('negative', '#E74C3C')]:
                        mask = timeline_df['Sentiment'] == sentiment
                        if mask.any():
                            fig_timeline.add_trace(go.Scatter(
                                x=timeline_df[mask]['Tanggal'],
                                y=timeline_df[mask]['Score'],
                                mode='markers',
                                name=sentiment.capitalize(),
                                marker=dict(color=color, size=8, opacity=0.6)
                            ))
                    
                    # Add trend line
                    timeline_df['Rolling_Avg'] = pd.Series(sentiment_scores).rolling(
                        window=min(7, len(texts)//3), min_periods=1
                    ).mean()
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_df['Tanggal'],
                        y=timeline_df['Rolling_Avg'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='black', width=2, dash='dash')
                    ))
                    
                    fig_timeline.update_layout(
                        height=350,
                        xaxis_title='Tanggal',
                        yaxis_title='Skor Sentimen',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Individual sentiment tabs with wordclouds
            color_schemes = {
                'positive': ['#2E8B57', '#32CD32', '#90EE90', '#98FB98', '#F0FFF0'],
                'negative': ['#8B0000', '#DC143C', '#FFB6C1', '#FFC0CB', '#FFF0F5'],
                'neutral': ['#4682B4', '#87CEEB', '#E0F6FF', '#F0F8FF', '#F8F8FF']
            }
            
            for tab, sentiment_name, sentiment_key in [
                (tab2, "Positive", "positive"),
                (tab3, "Neutral", "neutral"),
                (tab4, "Negative", "negative")
            ]:
                with tab:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Word cloud with improved styling
                        if all_words[sentiment_key]:
                            word_text = ' '.join(all_words[sentiment_key])
                            
                            colormap = LinearSegmentedColormap.from_list(
                                sentiment_key, color_schemes[sentiment_key]
                            )
                            
                            wordcloud = WordCloud(
                                width=800, height=500,
                                background_color='white',
                                colormap=colormap,
                                max_words=100,
                                collocations=False,
                                relative_scaling=0.5,
                                min_font_size=10
                            ).generate(word_text)
                            
                            fig_wc, ax = plt.subplots(figsize=(12, 8))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'Word Cloud - Sentimen {sentiment_name.capitalize()}', 
                                       fontsize=16, fontweight='bold', pad=15)
                            st.pyplot(fig_wc)
                            plt.close()
                        else:
                            st.info(f"Tidak ada data untuk sentimen {sentiment_name}")
                    
                    with col2:
                        # Top words chart
                        if all_words[sentiment_key]:
                            word_counts = Counter(all_words[sentiment_key])
                            top_words = word_counts.most_common(10)
                            
                            if top_words:
                                words, counts = zip(*top_words)
                                
                                color_map = {
                                    'positive': '#27AE60',
                                    'neutral': '#3498DB',
                                    'negative': '#E74C3C'
                                }
                                
                                fig_bar = go.Figure(go.Bar(
                                    y=list(reversed(words)),
                                    x=list(reversed(counts)),
                                    orientation='h',
                                    marker_color=color_map[sentiment_key],
                                    opacity=0.8
                                ))
                                
                                fig_bar.update_layout(
                                    title=f"Top 10 Kata - {sentiment_name.capitalize()}",
                                    height=500,
                                    xaxis_title='Frekuensi',
                                    yaxis_title='',
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Top Words Analysis Tab
            with tab5:
                st.subheader("📊 Analisis Kata Terpopuler")
                
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                colors = ['green', 'blue', 'red']
                
                for idx, (sentiment, words) in enumerate(all_words.items()):
                    ax = axes[idx]
                    
                    if words:
                        word_counts = Counter(words)
                        top_words = word_counts.most_common(15)
                        
                        if top_words:
                            words_list, counts_list = zip(*top_words)
                            
                            bars = ax.barh(range(len(words_list)), counts_list, 
                                         color=colors[idx], alpha=0.7)
                            ax.set_yticks(range(len(words_list)))
                            ax.set_yticklabels(words_list)
                            ax.invert_yaxis()
                            
                            for i, (bar, count) in enumerate(zip(bars, counts_list)):
                                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                       str(count), va='center', fontsize=9)
                            
                            ax.set_xlabel('Frequency')
                            ax.set_title(f'Top 15 Words - {sentiment.capitalize()}', 
                                       fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, f'No {sentiment} words', 
                               ha='center', va='center',
                               transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'Top 15 Words - {sentiment.capitalize()}', 
                                   fontweight='bold')
                
                plt.suptitle('Most Frequent Words by Sentiment Category', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        else:
            st.info("📥 Upload data sentimen untuk melihat analisis")
    
    # Helper functions for sentiment analysis
    def get_comprehensive_indonesian_stopwords(self):
        """Comprehensive list of Indonesian stopwords including affixes and conjunctions"""
        basic_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'oleh',
            'karena', 'sebagai', 'juga', 'akan', 'telah', 'sudah', 'masih', 'dapat', 'bisa',
            'ini', 'itu', 'adalah', 'ada', 'atau', 'tidak', 'ya', 'saya', 'kamu', 'dia',
            'kita', 'kami', 'mereka', 'satu', 'dua', 'tiga', 'banyak', 'sedikit', 'semua',
            'setiap', 'sama', 'lain', 'sendiri','ng','kan'
        }

        conjunctions = {
            'jika', 'kalau', 'maka', 'tetapi', 'tapi', 'namun', 'walaupun', 'meskipun',
            'setelah', 'sebelum', 'ketika', 'saat', 'waktu', 'bila', 'apabila', 'andai',
            'agar', 'supaya', 'hingga', 'sampai', 'sejak', 'semenjak', 'sedangkan', 'sementara'
        }

        question_words = {
            'tempat', 'dimana', 'kemana', 'darimana', 'bagaimana', 'mengapa', 'kenapa',
            'berapa', 'kapan', 'siapa', 'apa', 'mana', 'gimana'
        }

        informal_particles = {
            'gak', 'ga', 'enggak', 'nggak', 'bgt', 'banget', 'aja', 'sih', 'kok',
            'lho', 'deh', 'dong', 'kan', 'nih', 'tuh', 'dah', 'tau', 'kalo', 'emang',
            'jadi', 'terus', 'trus', 'iya', 'yah', 'yg', 'udah', 'udh', 'blm', 'belum',
            'nya','henti','suai','rja', 'lah', 'nah', 'ring', 'sma', 'adaan', 
            'urutku', 'tama', 'hat', 'masan', 'pol', 'nasaran', 'packaging'
        }

        pronouns = {
            'aku', 'kau', 'engkau', 'ia', 'beliau', 'nya', 'mu', 'ku', 'anda',
            'mereka', 'kalian', 'kita', 'kami'
        }

        common_affixed = {
            'tersebut', 'demikian', 'sebagainya', 'lainnya', 'sendiri', 'sekarang',
            'kemudian', 'selain', 'selama', 'sebelumnya', 'sesudah', 'beberapa',
            'berikut', 'masing', 'melalui', 'melainkan', 'menjadi', 'mengenai',
            'menurut', 'misalnya', 'sekitar', 'seolah', 'seringkali', 'terhadap',
            'termasuk', 'terlalu', 'tersedia', 'mungkin', 'sekali', 'sangat','bel','asa'
        }

        prepositions = {
            'di', 'ke', 'dari', 'pada', 'dalam', 'kepada', 'terhadap', 'oleh',
            'tentang', 'antara', 'hingga', 'sampai', 'lewat', 'tanpa'
        }

        time_words = {
            'hari', 'tahun', 'bulan', 'minggu', 'jam', 'menit', 'detik',
            'pagi', 'siang', 'sore', 'malam', 'besok', 'kemarin', 'nanti'
        }

        all_stopwords = (
            basic_stopwords | conjunctions | question_words | informal_particles |
            pronouns | common_affixed | prepositions | time_words
        )

        return all_stopwords

    def remove_affixes(self, word):
        """Remove common Indonesian affixes to get the root word"""
        prefixes = ['me', 'mem', 'men', 'meng', 'meny', 'di', 'ter', 'ber', 'per', 'pe', 'ke', 'se']
        suffixes = ['nya', 'kan', 'an']

        original_word = word

        # Remove suffix first
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break

        # Remove prefix
        for prefix in sorted(prefixes, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break

        return word if len(word) > 2 else original_word

    def clean_text_for_wordcloud(self, text, stopwords, remove_affixes_flag=True):
        """Clean and prepare text for word cloud"""
        if pd.isna(text) or str(text).strip() == '':
            return ""

        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        words = text.split()
        filtered_words = []
        
        for word in words:
            if word in stopwords or len(word) <= 2 or word.isdigit():
                continue

            if remove_affixes_flag:
                word = self.remove_affixes(word)

            if len(word) > 2 and word not in stopwords:
                filtered_words.append(word)

        return ' '.join(filtered_words)

def main():
    dashboard = StreamlitDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🍜 Pawonkoe Dashboard - Sales Prediction & Sentiment Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🍜 PAWONKOE")
        st.markdown("**Analytics Dashboard**")
        st.markdown("---")
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📁 Upload Data")
        
        # File upload for sales data
        st.subheader("1️⃣ Data Penjualan")
        sales_file = st.file_uploader(
            "Upload file penjualan (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="sales_upload",
            help="File harus berisi kolom: tanggal, total_penjualan"
        )
        
        if sales_file:
            dashboard.sales_data = dashboard.load_data(sales_file, "Data Penjualan")
        
        # File upload for sentiment data
        st.subheader("2️⃣ Data Sentimen")
        sentiment_file = st.file_uploader(
            "Upload file sentimen (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="sentiment_upload",
            help="File harus berisi kolom teks ulasan/review"
        )
        
        if sentiment_file:
            dashboard.sentiment_data = dashboard.load_data(sentiment_file, "Data Sentimen")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing options
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Pengaturan")
        
        if MODEL_AVAILABLE:
            st.success("🤖 Model XGBoost Ready!")
        else:
            st.warning("⚠️ Mode Simulasi")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <small>
        **Pawonkoe Analytics Dashboard**<br>
        Powered by XGBoost & WordCloud<br>
        Version 5.0 - Updated Forecasting
        </small>
        """, unsafe_allow_html=True)
    
    # Main dashboard area
    if dashboard.sales_data is not None:
        # Dashboard metrics
        dashboard.create_dashboard_metrics(dashboard.sales_data, dashboard.sentiment_data)
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "📈 Prediksi Penjualan", 
            "💬 Analisis Sentimen", 
            "📊 Data Mentah"
        ])
        
        with tab1:
            dashboard.create_sales_forecast_visualization(dashboard.sales_data)
        
        with tab2:
            dashboard.create_sentiment_analysis(dashboard.sentiment_data)
        
        with tab3:
            st.subheader("📊 Data Mentah")
            
            if dashboard.sales_data is not None:
                st.markdown("### Data Penjualan")
                st.dataframe(dashboard.sales_data, height=400)
                
                csv = dashboard.sales_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='data_penjualan.csv',
                    mime='text/csv'
                )
            
            if dashboard.sentiment_data is not None:
                st.markdown("### Data Sentimen")
                st.dataframe(dashboard.sentiment_data, height=400)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h2>👋 Selamat Datang di Pawonkoe Analytics Dashboard</h2>
            <p style='font-size: 1.2rem; color: #666; margin: 2rem 0;'>
                Platform prediksi penjualan & analisis sentimen berbasis AI untuk bisnis kuliner Anda
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Fitur Prediksi Penjualan
            
            - 📊 XGBoost Machine Learning
            - 📈 Prediksi 7 hari ke depan
            - 🎯 Akurasi tinggi dengan 6 fitur optimal
            - 📉 Visualisasi interaktif
            - 💡 Recursive forecasting strategy
            """)
        
        with col2:
            st.markdown("""
            ### 💬 Fitur Analisis Sentimen
            
            - 🔍 Analisis otomatis ulasan pelanggan
            - ☁️ WordCloud dengan stopwords Indonesia
            - 🗑️ Penghilangan imbuhan otomatis
            - 📊 Top words analysis
            - 📈 Distribusi sentimen
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📝 Format Data yang Didukung
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Contoh Data Penjualan:**")
            sample_sales = pd.DataFrame({
                'tanggal penjualan': ['01/01/2024', '02/01/2024', '03/01/2024'],
                'total penjualan': [1275000, 1559000, 1001000]
            })
            st.dataframe(sample_sales, hide_index=True)
        
        with col2:
            st.markdown("**Contoh Data Sentimen:**")
            sample_sentiment = pd.DataFrame({
                'ulasan': [
                    'Makanannya enak banget, recommended!',
                    'Pelayanan kurang memuaskan',
                    'Harga standar, rasa biasa saja'
                ]
            })
            st.dataframe(sample_sentiment, hide_index=True)

if __name__ == "__main__":
    main()