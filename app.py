import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Deret Waktu Saham",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih baik
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Background dan kontainer utama */
    .stApp {
        background: radial-gradient(circle at 20% 20%, rgba(102, 126, 234, 0.08), transparent 30%),
                    radial-gradient(circle at 80% 0%, rgba(246, 211, 101, 0.12), transparent 30%),
                    #f6f7fb;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Styling untuk header utama */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #312e81 100%);
        padding: 2.4rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 40px rgba(17, 24, 39, 0.25);
        position: relative;
        overflow: hidden;
    }
    .main-header:before {
        content: "";
        position: absolute;
        top: -40px;
        right: -80px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(94, 234, 212, 0.3), transparent 50%);
        filter: blur(8px);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.7rem;
        letter-spacing: 0.5px;
    }
    .main-header p {
        margin: 0.35rem 0 0;
        color: #e5e7eb;
    }

    /* Highlight badges */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.08);
        color: #e0e7ff;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(6px);
    }

    /* Cards */
    .section-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 1.5rem;
    }
    .section-card h3 {
        margin-top: 0;
        margin-bottom: 0.35rem;
        color: #0f172a;
        font-weight: 700;
    }
    .section-card p {
        margin-top: 0;
        color: #475569;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.08);
    }
    div[data-testid="stMetric"] > label {
        color: #475569;
    }
    div[data-testid="stMetric"] > div {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p {
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stDateInput > div > div {
        background-color: #111827;
        border: 1px solid #334155;
        border-radius: 10px;
        color: #e5e7eb;
    }

    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #f8fafc 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #bfdbfe;
        margin: 0.5rem 0 0;
        color: #0f172a;
    }
    .info-box strong {
        color: #1d4ed8;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat dan mempersiapkan data
def load_data(ticker):
    # Ganti dengan path dataset Anda
    file_path = 'World-Stock-Prices-Dataset.csv'  # Pastikan path dataset benar
    df = pd.read_csv(file_path)

    # Mengubah kolom 'Date' menjadi datetime dan memastikan bahwa 'Date' tidak memiliki zona waktu
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

    # Memeriksa apakah ada nilai yang tidak terkonversi menjadi datetime (NaT)
    if df['Date'].isnull().any():
        st.error("Ada nilai yang tidak dapat dikonversi menjadi datetime di kolom 'Date'.")
        st.write(df[df['Date'].isnull()])  # Menampilkan baris yang gagal konversi
        return None

    # Memfilter data untuk hanya sesuai dengan ticker yang dipilih
    df_ticker = df[df['Ticker'] == ticker]

    # Memeriksa apakah data cukup untuk dekomposisi
    if len(df_ticker) < 2:
        st.error(f"Data yang tersedia untuk ticker {ticker} tidak cukup untuk analisis dekomposisi.")
        return None

    # Mengatur kolom 'Date' sebagai indeks dan memastikan indeksnya menjadi DatetimeIndex
    df_ticker.set_index('Date', inplace=True)

    # Memastikan bahwa indeks sudah dalam format DatetimeIndex
    if not isinstance(df_ticker.index, pd.DatetimeIndex):
        st.error("Indeks 'Date' tidak terkonversi menjadi DatetimeIndex dengan benar.")
        return None

    # Mengisi missing values dengan interpolasi linear
    df_ticker['Close'] = df_ticker['Close'].interpolate(method='linear')

    # Resampling data untuk rata-rata bulanan
    df_monthly = df_ticker['Close'].resample('M').mean()

    # Menetapkan frekuensi pada indeks setelah resampling
    df_monthly = df_monthly.asfreq('M')  # Menambahkan frekuensi bulanan

    # Memeriksa apakah data cukup untuk dekomposisi
    if len(df_monthly) < 2:
        st.error(f"Data yang tersedia untuk ticker {ticker} tidak cukup untuk analisis dekomposisi.")
        return None

    return df_monthly


# Fungsi untuk dekomposisi deret waktu
def decompose_data(df_monthly):
    try:
        decomposition = seasonal_decompose(df_monthly, model='additive')
        return decomposition
    except Exception as e:
        st.error(f"Terjadi kesalahan saat dekomposisi: {e}")
        return None

# Header utama dengan styling
st.markdown("""
    <div class="main-header">
        <span class="pill">Realtime Time Series • Plotly</span>
        <h1>📈 Analisis Deret Waktu Saham</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">Platform Analisis Time Series untuk Prediksi Harga Saham</p>
    </div>
""", unsafe_allow_html=True)

# Highlight cards di bawah header
highlight_col1, highlight_col2, highlight_col3 = st.columns(3)
with highlight_col1:
    st.markdown("""
    <div class="section-card" style="padding: 1.2rem;">
        <h4 style="margin:0;">Data Bersih</h4>
        <p style="margin:0; color:#475569;">Interpolasi otomatis untuk menjaga kontinuitas data.</p>
    </div>
    """, unsafe_allow_html=True)
with highlight_col2:
    st.markdown("""
    <div class="section-card" style="padding: 1.2rem;">
        <h4 style="margin:0;">Insight Trend</h4>
        <p style="margin:0; color:#475569;">Dekomposisi additive untuk tren dan musim.</p>
    </div>
    """, unsafe_allow_html=True)
with highlight_col3:
    st.markdown("""
    <div class="section-card" style="padding: 1.2rem;">
        <h4 style="margin:0;">Visual Interaktif</h4>
        <p style="margin:0; color:#475569;">Plotly chart dengan hover terintegrasi.</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar untuk input
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Analisis")
    st.markdown("---")
    
    # Input untuk memilih ticker saham
    ticker = st.selectbox(
        '📊 Pilih Ticker Saham',
        ['AAPL', 'AMZN', 'TSLA'],
        help="Pilih saham yang ingin dianalisis"
    )
    
    st.markdown("---")
    
    # Input untuk memilih rentang tanggal
    st.markdown("### 📅 Rentang Waktu")
    start_date = st.date_input(
        "Tanggal Mulai",
        pd.to_datetime('2018-01-01'),
        help="Pilih tanggal mulai analisis"
    )
    end_date = st.date_input(
        "Tanggal Akhir",
        pd.to_datetime('2022-12-31'),
        help="Pilih tanggal akhir analisis"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.info("Aplikasi ini menganalisis data harga saham menggunakan metode dekomposisi time series untuk mengidentifikasi trend, seasonality, dan residual.")

# Mengonversi start_date dan end_date menjadi objek datetime dengan waktu default
start_date_dt = datetime.combine(start_date, datetime.min.time())
end_date_dt = datetime.combine(end_date, datetime.min.time())

# Menambahkan zona waktu UTC pada start_date dan end_date
start_date_utc = pd.Timestamp(start_date_dt).tz_localize('UTC')
end_date_utc = pd.Timestamp(end_date_dt).tz_localize('UTC')

# Memuat data
df_monthly = load_data(ticker)

# Jika data berhasil dimuat
if df_monthly is not None:
    # Menyaring data sesuai dengan rentang tanggal yang dipilih
    df_filtered = df_monthly.loc[start_date_utc:end_date_utc]
    
    # Menampilkan statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Jumlah Data",
            value=len(df_filtered),
            help="Total data point dalam rentang waktu yang dipilih"
        )
    
    with col2:
        min_price = df_filtered.min()
        st.metric(
            label="📉 Harga Terendah",
            value=f"${min_price:.2f}",
            help="Harga penutupan terendah dalam periode"
        )
    
    with col3:
        max_price = df_filtered.max()
        st.metric(
            label="📈 Harga Tertinggi",
            value=f"${max_price:.2f}",
            help="Harga penutupan tertinggi dalam periode"
        )
    
    with col4:
        avg_price = df_filtered.mean()
        st.metric(
            label="📊 Rata-rata Harga",
            value=f"${avg_price:.2f}",
            help="Rata-rata harga penutupan dalam periode"
        )
    
    st.markdown("---")
    
    # Menampilkan grafik harga saham dengan Plotly
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Grafik Harga Penutupan Saham")
    st.markdown("Pergerakan harga penutupan per bulan untuk ticker terpilih.")
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered.values,
        mode='lines',
        name='Harga Penutupan',
        line=dict(color='#667eea', width=2),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.08)'
    ))
    
    fig_price.update_layout(
        title=f'Harga Penutupan Saham {ticker}',
        xaxis_title='Tanggal',
        yaxis_title='Harga Penutupan (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        margin=dict(t=50, r=10, l=10, b=10)
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Dekomposisi dan menampilkan hasilnya
    st.markdown("---")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Dekomposisi Time Series")
    st.markdown("Dekomposisi data menjadi komponen Observed, Trend, Seasonality, dan Residual")
    
    decomposition = decompose_data(df_filtered)

    # Jika dekomposisi berhasil
    if decomposition:
        # Menggunakan Plotly untuk dekomposisi yang lebih interaktif
        fig_decomp = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed (Data Asli)', 'Trend', 'Seasonality', 'Residual'),
            vertical_spacing=0.08,
            row_heights=[0.3, 0.3, 0.2, 0.2]
        )
        
        # Observed
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values,
                      mode='lines', name='Observed', line=dict(color='#667eea')),
            row=1, col=1
        )
        
        # Trend
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values,
                      mode='lines', name='Trend', line=dict(color='#f093fb')),
            row=2, col=1
        )
        
        # Seasonal
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values,
                      mode='lines', name='Seasonal', line=dict(color='#4facfe')),
            row=3, col=1
        )
        
        # Residual
        fig_decomp.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values,
                      mode='lines', name='Residual', line=dict(color='#f5576c')),
            row=4, col=1
        )
        
        fig_decomp.update_layout(
            height=800,
            showlegend=False,
            template='plotly_white',
            title_text=f"Dekomposisi Time Series - {ticker}"
        )
        
        fig_decomp.update_xaxes(title_text="Tanggal", row=4, col=1)
        fig_decomp.update_yaxes(title_text="Nilai", row=2, col=1)
        
        st.plotly_chart(fig_decomp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Menampilkan Rolling Mean dan Rolling Std
    st.markdown("---")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Analisis Rolling Statistics")
    st.markdown("Rolling Mean dan Rolling Standard Deviation dengan window 12 bulan")
        
    rolling_mean = df_filtered.rolling(window=12).mean()
    rolling_std = df_filtered.rolling(window=12).std()

    # Menampilkan grafik Rolling Mean dan Rolling Std dengan Plotly
    fig_rolling = go.Figure()
    
    fig_rolling.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered.values,
        mode='lines',
        name='Harga Penutupan',
        line=dict(color='#667eea', width=2)
    ))
    
    fig_rolling.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        mode='lines',
        name='Rolling Mean (12 bulan)',
        line=dict(color='#f5576c', width=2, dash='dash')
    ))
    
    fig_rolling.add_trace(go.Scatter(
        x=rolling_std.index,
        y=rolling_std.values,
        mode='lines',
        name='Rolling Std (12 bulan)',
        line=dict(color='#43e97b', width=2, dash='dot')
    ))
    
    fig_rolling.update_layout(
        title=f'Rolling Mean dan Rolling Std - {ticker}',
        xaxis_title='Tanggal',
        yaxis_title='Harga Penutupan (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, r=10, l=10, b=10)
    )
    
    st.plotly_chart(fig_rolling, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>📊 Analisis Deret Waktu Saham | Dibuat dengan Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
