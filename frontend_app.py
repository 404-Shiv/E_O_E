import streamlit as st
import pickle, re, os, pandas as pd
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from src.stock_engine import stock_decision, search_stocks
from src.mf_engine import mutual_fund_decision, search_mutual_funds, get_mf_nav_history
from src.live_news_engine import get_live_company_news

# ─── Page Config ───
st.set_page_config(
    page_title="EOE — Financial Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Theme State ───
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ─── Sidebar: Theme Toggle ───
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
    st.session_state.dark_mode = dark_mode
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; text-align:center; opacity:0.5;'>"
        "EOE Financial Intelligence<br>Real-time Data Platform</p>",
        unsafe_allow_html=True,
    )

# ─── CSS Theming ───
if st.session_state.dark_mode:
    theme_css = """
    :root {
        --bg-primary: #0f1117;
        --bg-secondary: #1a1c2e;
        --bg-card: #1e2035;
        --bg-card-hover: #252845;
        --border-color: #2d2f4a;
        --text-primary: #e8eaf0;
        --text-secondary: #9ba3b5;
        --text-muted: #6b7280;
        --accent-primary: #3b82f6;
        --accent-secondary: #6366f1;
        --green: #10b981;
        --green-bg: rgba(16,185,129,0.12);
        --red: #ef4444;
        --red-bg: rgba(239,68,68,0.12);
        --yellow: #f59e0b;
        --yellow-bg: rgba(245,158,11,0.12);
        --shadow: 0 4px 24px rgba(0,0,0,0.4);
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
        --input-bg: #1a1c2e;
        --ticker-bg: #161829;
        --header-bg: #161829;
    }
    """
else:
    theme_css = """
    :root {
        --bg-primary: #f0f2f5;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --bg-card-hover: #f8f9fc;
        --border-color: #e2e5ea;
        --text-primary: #1a1d29;
        --text-secondary: #5a6078;
        --text-muted: #9ca3af;
        --accent-primary: #3b82f6;
        --accent-secondary: #6366f1;
        --green: #059669;
        --green-bg: rgba(5,150,105,0.08);
        --red: #dc2626;
        --red-bg: rgba(220,38,38,0.08);
        --yellow: #d97706;
        --yellow-bg: rgba(217,119,6,0.08);
        --shadow: 0 4px 24px rgba(0,0,0,0.06);
        --shadow-sm: 0 1px 4px rgba(0,0,0,0.05);
        --input-bg: #f8f9fc;
        --ticker-bg: #1e293b;
        --header-bg: #1e293b;
    }
    """

st.markdown(f"""
<style>
{theme_css}

/* ─── Global ─── */
.stApp {{
    background-color: var(--bg-primary) !important;
}}
section[data-testid="stSidebar"] {{
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
}}

/* ─── Header bar ─── */
.eoe-header {{
    background: var(--header-bg);
    padding: 14px 32px;
    border-radius: 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-sm);
}}
.eoe-logo {{
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}}
.eoe-logo span {{
    color: var(--accent-primary);
}}
.eoe-subtitle {{
    color: rgba(255,255,255,0.5);
    font-size: 0.75rem;
    margin-left: 12px;
}}

/* ─── Market Ticker ─── */
.ticker-bar {{
    background: var(--ticker-bg);
    padding: 8px 0;
    border-radius: 8px;
    margin-bottom: 20px;
    overflow: hidden;
    white-space: nowrap;
    position: relative;
}}
.ticker-content {{
    display: inline-block;
    animation: ticker 30s linear infinite;
    color: #e0e0e0;
    font-size: 0.82rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
}}
.ticker-content .up {{ color: #10b981; font-weight: 600; }}
.ticker-content .down {{ color: #ef4444; font-weight: 600; }}
.ticker-content .name {{ color: #94a3b8; font-weight: 600; margin-right: 4px; }}
.ticker-content .sep {{ color: #475569; margin: 0 16px; }}
@keyframes ticker {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.9rem;
    padding: 8px 20px;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    background: var(--accent-primary) !important;
    color: #ffffff !important;
    border-radius: 8px;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none;
}}

/* ─── Cards ─── */
.card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: var(--shadow);
    transition: all 0.25s ease;
}}
.card:hover {{
    box-shadow: var(--shadow), 0 0 0 1px var(--accent-primary);
    transform: translateY(-1px);
}}
.card-header {{
    color: var(--text-muted);
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}}
.card-value {{
    color: var(--text-primary);
    font-size: 1.65rem;
    font-weight: 800;
    margin: 4px 0;
}}
.card-sub {{
    font-size: 0.85rem;
    color: var(--text-secondary);
}}

/* ─── Metrics ─── */
div[data-testid="stMetric"] {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: var(--shadow-sm);
}}
div[data-testid="stMetric"] label {{
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: var(--text-primary) !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}}

/* ─── Buttons ─── */
.stButton > button {{
    background: var(--accent-primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.25) !important;
}}
.stButton > button:hover {{
    background: var(--accent-secondary) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}}

/* ─── Inputs ─── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {{
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
}}

/* ─── Badges ─── */
.badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}}
.badge-true {{
    background: var(--green-bg);
    color: var(--green);
    border: 1px solid var(--green);
}}
.badge-fake {{
    background: var(--red-bg);
    color: var(--red);
    border: 1px solid var(--red);
}}
.badge-buy {{
    background: var(--green-bg);
    color: var(--green);
    border: 1px solid var(--green);
    font-size: 1rem;
    padding: 6px 20px;
}}
.badge-sell {{
    background: var(--red-bg);
    color: var(--red);
    border: 1px solid var(--red);
    font-size: 1rem;
    padding: 6px 20px;
}}
.badge-hold {{
    background: var(--yellow-bg);
    color: var(--yellow);
    border: 1px solid var(--yellow);
    font-size: 1rem;
    padding: 6px 20px;
}}

/* ─── News Items ─── */
.news-item {{
    background: var(--bg-card);
    border-left: 4px solid var(--accent-primary);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
    transition: all 0.2s ease;
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
}}
.news-item:hover {{
    background: var(--bg-card-hover);
    transform: translateX(3px);
}}
.news-item .title {{
    color: var(--text-primary);
    font-size: 0.95rem;
    font-weight: 500;
    margin: 0;
    line-height: 1.4;
}}
.news-item .meta {{
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 6px;
}}

/* ─── Score Bar ─── */
.score-bar-bg {{
    background: var(--input-bg);
    border-radius: 10px;
    height: 28px;
    width: 100%;
    overflow: hidden;
    margin: 8px 0;
    border: 1px solid var(--border-color);
}}
.score-bar-fill {{
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    transition: width 0.6s ease;
}}

/* ─── Divider ─── */
.divider {{
    height: 1px;
    background: var(--border-color);
    margin: 16px 0;
}}

/* ─── Section Titles ─── */
h1 {{
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.3px;
}}
h2, h3 {{
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}}
.section-desc {{
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 16px;
}}

/* ─── Performance badges ─── */
.perf-badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 1.05rem;
}}
.perf-outperformer {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }}
.perf-good {{ background: rgba(16,185,129,0.08); color: #34d399; border: 1px solid #34d399; }}
.perf-neutral {{ background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }}
.perf-under {{ background: rgba(249,115,22,0.1); color: #f97316; border: 1px solid #f97316; }}
.perf-poor {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red); }}

/* ─── Data table ─── */
.stDataFrame {{
    border-radius: 12px !important;
    overflow: hidden;
}}

/* ─── Invest-style stock mini card ─── */
.stock-mini {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
    box-shadow: var(--shadow-sm);
}}
.stock-mini .sym {{
    color: var(--accent-primary);
    font-weight: 700;
    font-size: 0.85rem;
}}
.stock-mini .name {{
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.95rem;
}}

/* ─── Company news item ─── */
.company-news {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: all 0.2s ease;
}}
.company-news:hover {{
    background: var(--bg-card-hover);
}}
.company-news a {{
    color: var(--accent-primary);
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
}}
.company-news .src {{
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)


# ─── Load Fake News Model ───
@st.cache_resource
def load_model():
    model = pickle.load(open("models/fake_news_model.pkl", "rb"))
    vec = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    return model, vec

try:
    model, vec = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

stop = set(stopwords.words("english"))
def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z ]", "", t)
    return " ".join(w for w in t.split() if w not in stop)


# ─── Header ───
st.markdown(
    '<div class="eoe-header">'
    '<div><span class="eoe-logo">📊 EOE<span> Intelligence</span></span>'
    '<span class="eoe-subtitle">Financial Analysis Platform</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ─── Market Ticker ───
ticker_items = (
    '<span class="name">S&P 500</span> 4,213.80 <span class="up">▲ +60,300 (+1.45%)</span>'
    '<span class="sep">│</span>'
    '<span class="name">NIFTY 50</span> 22,147.00 <span class="up">▲ +134.50 (+0.61%)</span>'
    '<span class="sep">│</span>'
    '<span class="name">SENSEX</span> 72,831.94 <span class="down">▼ −218.30 (−0.30%)</span>'
    '<span class="sep">│</span>'
    '<span class="name">DOW JONES</span> 33,700 <span class="down">▼ −61 (−0.18%)</span>'
    '<span class="sep">│</span>'
    '<span class="name">NIKKEI</span> 225,289 <span class="up">▲ +873,200 (+3.12%)</span>'
    '<span class="sep">│</span>'
    '<span class="name">BANK NIFTY</span> 47,035.15 <span class="up">▲ +281.60 (+0.60%)</span>'
    '<span class="sep">│</span>'
)

st.markdown(
    f'<div class="ticker-bar">'
    f'<div class="ticker-content">{ticker_items}{ticker_items}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ───────────────────────────────────────────────
#  TABS — Reordered as requested
# ───────────────────────────────────────────────
tab_news, tab_stocks, tab_compare, tab_mf, tab_verifier = st.tabs([
    "📰 Live News",
    "📈 Stocks",
    "⚖️ Compare",
    "💰 Mutual Funds",
    "🔍 News Verifier",
])


# ─── Helper: Render stock analysis ───
def render_stock_result(result, container=st):
    """Renders stock analysis result cards."""
    if "error" in result:
        container.error(f"Error: {result['error']}")
        return

    sig = result["signal"]
    sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")
    badge_class = f"badge-{sig.lower()}"

    container.markdown(
        f'<div class="card" style="text-align:center;">'
        f'<div class="card-header">Recommendation</div>'
        f'<div class="card-value">{result["symbol"]}</div>'
        f'<span class="badge {badge_class}">{sig_emoji} {sig}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = container.columns(2)
    c1.metric("Last Price", f"₹{result['last_price']}")
    change = result['change_pct']
    c2.metric("Change %", f"{change}%", delta=f"{change}%")

    c3, c4 = container.columns(2)
    c3.metric("SMA 20", f"₹{result['sma_20']}")
    c4.metric("SMA 50", f"₹{result['sma_50']}")

    c5, c6 = container.columns(2)
    c5.metric("3M High", f"₹{result.get('high_3m', 'N/A')}")
    c6.metric("3M Low", f"₹{result.get('low_3m', 'N/A')}")

    container.metric("Volume", f"{result['volume']:,}")

    hist = result.get("history")
    if hist is not None and not hist.empty:
        container.line_chart(hist["Close"], use_container_width=True)


# ════════════════════════════════════════════════
#  📰 LIVE NEWS FEED
# ════════════════════════════════════════════════
with tab_news:
    st.markdown("# 📰 Live News Feed")
    st.markdown('<p class="section-desc">Real-time financial news classified by our AI model</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    classified_path = "data/processed/classified_news.csv"
    raw_path = "data/raw/real_news_master.csv"

    df = None
    needs_classification = False
    if os.path.exists(classified_path):
        df = pd.read_csv(classified_path)
    elif os.path.exists(raw_path) and model_loaded:
        df = pd.read_csv(raw_path)
        needs_classification = True
    else:
        st.info("No news data found. Click **Refresh** below to fetch news.")

    if df is not None and not df.empty:
        if needs_classification:
            df["cleaned"] = df["text"].apply(clean)
            probs = model.predict_proba(vec.transform(df["cleaned"]))
            df["truth_score"] = [round(p[1] * 100, 2) for p in probs]
            df["verdict"] = ["TRUE" if p[1] >= 0.5 else "FAKE" for p in probs]

        col_f1, col_f2, _ = st.columns([2, 2, 3])
        with col_f1:
            filter_opt = st.selectbox("Filter", ["All", "TRUE Only", "FAKE Only"])
        with col_f2:
            sort_opt = st.selectbox("Sort by", ["Latest", "Highest Score", "Lowest Score"])

        if filter_opt == "TRUE Only":
            df = df[df["verdict"] == "TRUE"]
        elif filter_opt == "FAKE Only":
            df = df[df["verdict"] == "FAKE"]

        if sort_opt == "Highest Score":
            df = df.sort_values("truth_score", ascending=False)
        elif sort_opt == "Lowest Score":
            df = df.sort_values("truth_score", ascending=True)

        total = len(df)
        true_count = len(df[df["verdict"] == "TRUE"]) if "verdict" in df.columns else 0
        fake_count = len(df[df["verdict"] == "FAKE"]) if "verdict" in df.columns else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Articles", total)
        c2.metric("✅ True News", true_count)
        c3.metric("❌ Fake News", fake_count)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        for _, row in df.head(50).iterrows():
            text = str(row.get("text", ""))
            source = str(row.get("source", "Unknown"))
            date = str(row.get("date", ""))
            verdict = str(row.get("verdict", ""))
            score = row.get("truth_score", 0)

            if verdict == "TRUE":
                badge = f'<span class="badge badge-true">✅ TRUE — {score}%</span>'
                border_color = "var(--green)"
            else:
                badge = f'<span class="badge badge-fake">❌ FAKE — {score}%</span>'
                border_color = "var(--red)"

            st.markdown(
                f'<div class="news-item" style="border-left-color:{border_color};">'
                f'{badge}'
                f'<p class="title" style="margin-top:8px;">{text}</p>'
                f'<p class="meta">📍 {source} &nbsp;|&nbsp; 🕐 {date}</p></div>',
                unsafe_allow_html=True,
            )

    if st.button("🔄 Refresh News Feed", key="refresh_news"):
        try:
            from src.realtime_multi_news_fetch import collect
            with st.spinner("Fetching fresh news from sources..."):
                collect()
            st.success("News feed updated!")
            st.rerun()
        except Exception as e:
            st.error(f"Error refreshing: {e}")


# ════════════════════════════════════════════════
#  📈 STOCK ANALYSIS
# ════════════════════════════════════════════════
with tab_stocks:
    st.markdown("# 📈 Stock Analysis")
    st.markdown(
        '<p class="section-desc">Search any company to view real-time price, SMA signals, and recommendation</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    query = st.text_input(
        "🔍 Search company or sector",
        placeholder="e.g.  tata, reliance, bank, pharma, auto, hdfc...",
        key="stock_search",
    )

    if query.strip():
        results = search_stocks(query)
        if results:
            st.markdown(f"**{len(results)} matching stocks:**")
            options = [f"{s['name']}  ({s['symbol']})" for s in results]
            selected = st.selectbox("Select a stock to analyze", options, index=0, key="stock_select")
            selected_symbol = selected.split("(")[-1].replace(")", "").strip()

            if st.button("📊 Analyze Selected Stock", use_container_width=True, key="analyze_selected"):
                with st.spinner(f"Fetching data for {selected_symbol}..."):
                    result = stock_decision(selected_symbol)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                render_stock_result(result)

                # Company News
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📰 Related News")
                try:
                    news = get_live_company_news(selected_symbol)
                    if news:
                        for article in news[:5]:
                            title = article.get("title", "")
                            source = article.get("source", "")
                            url = article.get("url", "#")
                            st.markdown(
                                f'<div class="company-news">'
                                f'<a href="{url}" target="_blank">{title}</a>'
                                f'<div class="src">📍 {source}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No recent news found for this stock.")
                except Exception:
                    st.info("Could not fetch company news.")
        else:
            st.info("No matching stocks found. Try: tata, reliance, bank, pharma, auto...")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**Or enter a symbol directly:**")
    direct_sym = st.text_input("Stock Symbol", placeholder="e.g. INFY.NS", key="direct_sym")
    if st.button("📊 Analyze Symbol", key="analyze_direct"):
        if direct_sym.strip():
            with st.spinner(f"Fetching data for {direct_sym}..."):
                result = stock_decision(direct_sym.strip())
            render_stock_result(result)
        else:
            st.warning("Enter a symbol.")


# ════════════════════════════════════════════════
#  ⚖️ STOCK COMPARISON
# ════════════════════════════════════════════════
with tab_compare:
    st.markdown("# ⚖️ Stock Comparison")
    st.markdown(
        '<p class="section-desc">Compare two stocks side by side with real-time metrics and charts</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Stock A")
        query_a = st.text_input("Search company A", placeholder="e.g. tata", key="cmp_a_search")
        results_a = search_stocks(query_a) if query_a.strip() else []
        if results_a:
            opts_a = [f"{s['name']}  ({s['symbol']})" for s in results_a]
            sel_a = st.selectbox("Select Stock A", opts_a, key="sel_a")
            sym_a = sel_a.split("(")[-1].replace(")", "").strip()
        else:
            sym_a = st.text_input("Symbol A", placeholder="INFY.NS", key="sym_a_direct")

    with col_b:
        st.markdown("### Stock B")
        query_b = st.text_input("Search company B", placeholder="e.g. infosys", key="cmp_b_search")
        results_b = search_stocks(query_b) if query_b.strip() else []
        if results_b:
            opts_b = [f"{s['name']}  ({s['symbol']})" for s in results_b]
            sel_b = st.selectbox("Select Stock B", opts_b, key="sel_b")
            sym_b = sel_b.split("(")[-1].replace(")", "").strip()
        else:
            sym_b = st.text_input("Symbol B", placeholder="TCS.NS", key="sym_b_direct")

    if st.button("⚖️ Compare Stocks", use_container_width=True, key="compare_btn"):
        if sym_a and sym_b:
            with st.spinner(f"Fetching {sym_a} and {sym_b}..."):
                res_a = stock_decision(sym_a.strip())
                res_b = stock_decision(sym_b.strip())

            if "error" in res_a:
                st.error(f"Stock A error: {res_a['error']}")
            elif "error" in res_b:
                st.error(f"Stock B error: {res_b['error']}")
            else:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Signal cards side by side
                col1, col2 = st.columns(2)
                for col, res in [(col1, res_a), (col2, res_b)]:
                    with col:
                        sig = res["signal"]
                        sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")
                        badge_class = f"badge-{sig.lower()}"
                        st.markdown(
                            f'<div class="card" style="text-align:center;">'
                            f'<div class="card-header">Signal</div>'
                            f'<div class="card-value">{res["symbol"]}</div>'
                            f'<span class="badge {badge_class}">{sig_emoji} {sig}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Side-by-side metrics table
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📊 Side-by-Side Metrics")

                metrics = [
                    ("Last Price", "last_price", "₹", True),
                    ("Change %", "change_pct", "%", True),
                    ("SMA 20", "sma_20", "₹", False),
                    ("SMA 50", "sma_50", "₹", False),
                    ("3M High", "high_3m", "₹", True),
                    ("3M Low", "low_3m", "₹", False),
                    ("Volume", "volume", "", True),
                ]

                table_data = []
                for label, key, prefix, higher_better in metrics:
                    va = res_a.get(key, "N/A")
                    vb = res_b.get(key, "N/A")
                    if prefix == "₹":
                        va_str = f"₹{va:,}" if isinstance(va, (int, float)) else str(va)
                        vb_str = f"₹{vb:,}" if isinstance(vb, (int, float)) else str(vb)
                    elif prefix == "%":
                        va_str = f"{va}%"
                        vb_str = f"{vb}%"
                    else:
                        va_str = f"{va:,}" if isinstance(va, (int, float)) else str(va)
                        vb_str = f"{vb:,}" if isinstance(vb, (int, float)) else str(vb)
                    table_data.append({
                        "Metric": label,
                        res_a["symbol"]: va_str,
                        res_b["symbol"]: vb_str,
                    })

                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True,
                )

                # Price chart overlay
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📈 Price Chart Comparison")
                hist_a = res_a.get("history")
                hist_b = res_b.get("history")
                if hist_a is not None and hist_b is not None:
                    chart_df = pd.DataFrame({
                        res_a["symbol"]: hist_a["Close"],
                        res_b["symbol"]: hist_b["Close"],
                    })
                    st.line_chart(chart_df, use_container_width=True)
        else:
            st.warning("Enter both stock symbols to compare.")


# ════════════════════════════════════════════════
#  💰 MUTUAL FUNDS
# ════════════════════════════════════════════════
with tab_mf:
    st.markdown("# 💰 Mutual Funds")
    st.markdown(
        '<p class="section-desc">Search mutual funds, view live NAV, and check performance ratings</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if "mf_data" not in st.session_state:
        st.session_state.mf_data = None

    if st.button("📥 Load Live Fund Data", use_container_width=True, key="load_mf"):
        with st.spinner("Fetching mutual fund data from AMFI..."):
            st.session_state.mf_data = mutual_fund_decision()

    mf_df = st.session_state.mf_data

    if mf_df is not None and not mf_df.empty and "error" not in mf_df.columns:
        st.success(f"✅ Loaded {len(mf_df)} mutual fund schemes")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        mf_query = st.text_input(
            "🔍 Search fund by name",
            placeholder="e.g.  HDFC, SBI, Axis, Nippon, large cap, flexi...",
            key="mf_search",
        )

        filtered = search_mutual_funds(mf_query, mf_df)

        if not filtered.empty:
            st.markdown(f"**Showing {len(filtered)} funds:**")
            display_df = filtered[["scheme", "nav", "date", "category"]].copy()
            display_df.columns = ["Scheme Name", "NAV", "Date", "Category"]
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Fund Performance Analysis")

            fund_options = [
                f"{row['scheme']}  (Code: {row['code']})"
                for _, row in filtered.head(20).iterrows()
            ]
            if fund_options:
                selected_fund = st.selectbox("Select a fund", fund_options, key="mf_select")
                code = selected_fund.split("Code: ")[-1].replace(")", "").strip()

                if st.button("📈 Analyze Performance", use_container_width=True, key="analyze_mf"):
                    with st.spinner("Fetching NAV history and calculating returns..."):
                        hist_df, metrics = get_mf_nav_history(code)

                    if "error" in metrics:
                        st.error(f"Error: {metrics['error']}")
                    elif hist_df is not None:
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                        # Verdict badge
                        verdict = metrics.get("verdict", "🟡 NEUTRAL")
                        verdict_class_map = {
                            "⭐ OUTPERFORMER": "perf-outperformer",
                            "✅ GOOD PERFORMER": "perf-good",
                            "🟡 NEUTRAL": "perf-neutral",
                            "⚠️ UNDERPERFORMER": "perf-under",
                            "🔴 POOR PERFORMER": "perf-poor",
                        }
                        v_class = verdict_class_map.get(verdict, "perf-neutral")

                        st.markdown(
                            f'<div class="card" style="text-align:center;">'
                            f'<div class="card-header">Performance Rating</div>'
                            f'<span class="perf-badge {v_class}" style="margin-top:8px;">{verdict}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Return metrics
                        rcols = st.columns(4)
                        if "return_1m" in metrics:
                            rcols[0].metric("1M Return", f"{metrics['return_1m']}%",
                                            delta=f"{metrics['return_1m']}%")
                        if "return_3m" in metrics:
                            rcols[1].metric("3M Return", f"{metrics['return_3m']}%",
                                            delta=f"{metrics['return_3m']}%")
                        if "return_6m" in metrics:
                            rcols[2].metric("6M Return", f"{metrics['return_6m']}%",
                                            delta=f"{metrics['return_6m']}%")
                        if "return_1y" in metrics:
                            rcols[3].metric("1Y Return (XIRR)", f"{metrics['return_1y']}%",
                                            delta=f"{metrics['return_1y']}%")

                        # Risk metrics
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        r2cols = st.columns(2)
                        if "volatility" in metrics:
                            r2cols[0].metric("📊 Volatility (Annual)", f"{metrics['volatility']}%")
                        if "max_drawdown" in metrics:
                            r2cols[1].metric("📉 Max Drawdown", f"{metrics['max_drawdown']}%")

                        # NAV chart
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        st.markdown("### NAV History")
                        chart_data = hist_df.set_index("date")["nav"]
                        st.line_chart(chart_data, use_container_width=True)
                    else:
                        st.warning("No historical data available for this fund.")
        else:
            st.info("No funds found matching your search.")

    elif mf_df is not None and "error" in mf_df.columns:
        st.error(f"Error loading funds: {mf_df['error'].iloc[0]}")
    else:
        st.info("Click **Load Live Fund Data** above to fetch mutual fund schemes from AMFI India.")


# ════════════════════════════════════════════════
#  🔍 NEWS VERIFIER (Last)
# ════════════════════════════════════════════════
with tab_verifier:
    st.markdown("# 🔍 News Verifier")
    st.markdown(
        '<p class="section-desc">Paste any financial news headline to check if it\'s genuine or fake</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model not found! Run `python src/train_fake_news.py` first.")

    news_input = st.text_area(
        "Enter a news headline or article text",
        height=120,
        placeholder="e.g.  RBI keeps repo rate unchanged at 6.5% amid inflation concerns...",
        key="verify_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        check_btn = st.button("🔍 Verify", use_container_width=True, key="verify_btn")

    if check_btn and news_input.strip() and model_loaded:
        cleaned = clean(news_input)
        probs = model.predict_proba(vec.transform([cleaned]))[0]
        truth_pct = round(probs[1] * 100, 2)
        fake_pct = round(probs[0] * 100, 2)
        is_true = truth_pct >= 50

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col_v, col_s = st.columns([1, 2])
        with col_v:
            if is_true:
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<span class="badge badge-true">✅ LIKELY TRUE</span>'
                    f'<div class="card-value" style="color:var(--green); margin-top:12px;">{truth_pct}%</div>'
                    f'<div class="card-sub">Truth Confidence</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<span class="badge badge-fake">❌ LIKELY FAKE</span>'
                    f'<div class="card-value" style="color:var(--red); margin-top:12px;">{fake_pct}%</div>'
                    f'<div class="card-sub">Fake Confidence</div></div>',
                    unsafe_allow_html=True,
                )

        with col_s:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-header">Truth Score Distribution</div>'
                f'<div class="score-bar-bg">'
                f'<div class="score-bar-fill" style="width:{truth_pct}%; background:linear-gradient(90deg,#059669,#10b981);">'
                f'True {truth_pct}%</div></div>'
                f'<div class="score-bar-bg">'
                f'<div class="score-bar-fill" style="width:{fake_pct}%; background:linear-gradient(90deg,#dc2626,#ef4444);">'
                f'Fake {fake_pct}%</div></div></div>',
                unsafe_allow_html=True,
            )
    elif check_btn and not news_input.strip():
        st.warning("Please enter some text to verify.")