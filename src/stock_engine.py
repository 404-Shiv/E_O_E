import yfinance as yf

# ── Popular Indian stocks for NLP search ──
STOCK_DB = {
    "tata": [
        {"name": "Tata Consultancy Services", "symbol": "TCS.NS"},
        {"name": "Tata Motors", "symbol": "TATAMOTORS.NS"},
        {"name": "Tata Steel", "symbol": "TATASTEEL.NS"},
        {"name": "Tata Power", "symbol": "TATAPOWER.NS"},
        {"name": "Tata Consumer Products", "symbol": "TATACONSUM.NS"},
        {"name": "Tata Elxsi", "symbol": "TATAELXSI.NS"},
        {"name": "Tata Communications", "symbol": "TATACOMM.NS"},
        {"name": "Tata Chemicals", "symbol": "TATACHEM.NS"},
        {"name": "Tata Teleservices", "symbol": "TTML.NS"},
        {"name": "Indian Hotels (Tata)", "symbol": "INDHOTEL.NS"},
    ],
    "reliance": [
        {"name": "Reliance Industries", "symbol": "RELIANCE.NS"},
    ],
    "infosys": [
        {"name": "Infosys", "symbol": "INFY.NS"},
    ],
    "wipro": [
        {"name": "Wipro", "symbol": "WIPRO.NS"},
    ],
    "hdfc": [
        {"name": "HDFC Bank", "symbol": "HDFCBANK.NS"},
        {"name": "HDFC Life Insurance", "symbol": "HDFCLIFE.NS"},
        {"name": "HDFC AMC", "symbol": "HDFCAMC.NS"},
    ],
    "icici": [
        {"name": "ICICI Bank", "symbol": "ICICIBANK.NS"},
        {"name": "ICICI Prudential Life", "symbol": "ICICIPRULI.NS"},
        {"name": "ICICI Lombard", "symbol": "ICICIGI.NS"},
    ],
    "adani": [
        {"name": "Adani Enterprises", "symbol": "ADANIENT.NS"},
        {"name": "Adani Ports", "symbol": "ADANIPORTS.NS"},
        {"name": "Adani Green Energy", "symbol": "ADANIGREEN.NS"},
        {"name": "Adani Power", "symbol": "ADANIPOWER.NS"},
        {"name": "Adani Total Gas", "symbol": "ATGL.NS"},
        {"name": "Adani Wilmar", "symbol": "AWL.NS"},
    ],
    "bajaj": [
        {"name": "Bajaj Finance", "symbol": "BAJFINANCE.NS"},
        {"name": "Bajaj Finserv", "symbol": "BAJAJFINSV.NS"},
        {"name": "Bajaj Auto", "symbol": "BAJAJ-AUTO.NS"},
        {"name": "Bajaj Holdings", "symbol": "BAJAJHLDNG.NS"},
    ],
    "mahindra": [
        {"name": "Mahindra & Mahindra", "symbol": "M&M.NS"},
        {"name": "Tech Mahindra", "symbol": "TECHM.NS"},
        {"name": "Mahindra Finance", "symbol": "M&MFIN.NS"},
    ],
    "sbi": [
        {"name": "State Bank of India", "symbol": "SBIN.NS"},
        {"name": "SBI Life Insurance", "symbol": "SBILIFE.NS"},
        {"name": "SBI Cards", "symbol": "SBICARD.NS"},
    ],
    "kotak": [
        {"name": "Kotak Mahindra Bank", "symbol": "KOTAKBANK.NS"},
    ],
    "itc": [
        {"name": "ITC Limited", "symbol": "ITC.NS"},
    ],
    "bharti": [
        {"name": "Bharti Airtel", "symbol": "BHARTIARTL.NS"},
    ],
    "airtel": [
        {"name": "Bharti Airtel", "symbol": "BHARTIARTL.NS"},
    ],
    "sun": [
        {"name": "Sun Pharma", "symbol": "SUNPHARMA.NS"},
    ],
    "hul": [
        {"name": "Hindustan Unilever", "symbol": "HINDUNILVR.NS"},
    ],
    "asian": [
        {"name": "Asian Paints", "symbol": "ASIANPAINT.NS"},
    ],
    "maruti": [
        {"name": "Maruti Suzuki", "symbol": "MARUTI.NS"},
    ],
    "larsen": [
        {"name": "Larsen & Toubro", "symbol": "LT.NS"},
    ],
    "lt": [
        {"name": "Larsen & Toubro", "symbol": "LT.NS"},
    ],
    "axis": [
        {"name": "Axis Bank", "symbol": "AXISBANK.NS"},
    ],
    "ultratech": [
        {"name": "UltraTech Cement", "symbol": "ULTRACEMCO.NS"},
    ],
    "nestle": [
        {"name": "Nestle India", "symbol": "NESTLEIND.NS"},
    ],
    "titan": [
        {"name": "Titan Company", "symbol": "TITAN.NS"},
    ],
    "power": [
        {"name": "Power Grid Corp", "symbol": "POWERGRID.NS"},
        {"name": "NTPC", "symbol": "NTPC.NS"},
        {"name": "Tata Power", "symbol": "TATAPOWER.NS"},
        {"name": "Adani Power", "symbol": "ADANIPOWER.NS"},
    ],
    "bank": [
        {"name": "HDFC Bank", "symbol": "HDFCBANK.NS"},
        {"name": "ICICI Bank", "symbol": "ICICIBANK.NS"},
        {"name": "State Bank of India", "symbol": "SBIN.NS"},
        {"name": "Axis Bank", "symbol": "AXISBANK.NS"},
        {"name": "Kotak Mahindra Bank", "symbol": "KOTAKBANK.NS"},
        {"name": "IndusInd Bank", "symbol": "INDUSINDBK.NS"},
    ],
    "pharma": [
        {"name": "Sun Pharma", "symbol": "SUNPHARMA.NS"},
        {"name": "Dr Reddy's", "symbol": "DRREDDY.NS"},
        {"name": "Cipla", "symbol": "CIPLA.NS"},
        {"name": "Divis Labs", "symbol": "DIVISLAB.NS"},
    ],
    "auto": [
        {"name": "Maruti Suzuki", "symbol": "MARUTI.NS"},
        {"name": "Tata Motors", "symbol": "TATAMOTORS.NS"},
        {"name": "Mahindra & Mahindra", "symbol": "M&M.NS"},
        {"name": "Bajaj Auto", "symbol": "BAJAJ-AUTO.NS"},
        {"name": "Hero MotoCorp", "symbol": "HEROMOTOCO.NS"},
    ],
    "it": [
        {"name": "TCS", "symbol": "TCS.NS"},
        {"name": "Infosys", "symbol": "INFY.NS"},
        {"name": "Wipro", "symbol": "WIPRO.NS"},
        {"name": "HCL Tech", "symbol": "HCLTECH.NS"},
        {"name": "Tech Mahindra", "symbol": "TECHM.NS"},
    ],
    "hero": [
        {"name": "Hero MotoCorp", "symbol": "HEROMOTOCO.NS"},
    ],
    "hcl": [
        {"name": "HCL Technologies", "symbol": "HCLTECH.NS"},
    ],
    "cipla": [
        {"name": "Cipla", "symbol": "CIPLA.NS"},
    ],
    "dr reddy": [
        {"name": "Dr Reddy's", "symbol": "DRREDDY.NS"},
    ],
    "coal": [
        {"name": "Coal India", "symbol": "COALINDIA.NS"},
    ],
    "hindalco": [
        {"name": "Hindalco Industries", "symbol": "HINDALCO.NS"},
    ],
    "jio": [
        {"name": "Reliance Industries (Jio parent)", "symbol": "RELIANCE.NS"},
    ],
    "vedanta": [
        {"name": "Vedanta Limited", "symbol": "VEDL.NS"},
    ],
    "zomato": [
        {"name": "Zomato", "symbol": "ZOMATO.NS"},
    ],
    "paytm": [
        {"name": "Paytm (One97)", "symbol": "PAYTM.NS"},
    ],
}


def search_stocks(query):
    """Search for stocks by company name — NLP style fuzzy match."""
    query = query.lower().strip()
    if not query:
        return []

    results = []
    seen = set()
    for keyword, stocks in STOCK_DB.items():
        if query in keyword or keyword in query:
            for s in stocks:
                if s["symbol"] not in seen:
                    results.append(s)
                    seen.add(s["symbol"])

    # Also try to match inside stock names
    for keyword, stocks in STOCK_DB.items():
        for s in stocks:
            if query in s["name"].lower() and s["symbol"] not in seen:
                results.append(s)
                seen.add(s["symbol"])

    return results


def stock_decision(symbol, period="3mo"):
    """Fetch real stock data and return analysis with recommendation."""
    try:
        valid_periods = ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"]
        if period not in valid_periods:
            period = "3mo"

        tk = yf.Ticker(symbol)
        hist = tk.history(period=period)

        if hist.empty:
            return {"error": f"No data found for {symbol}"}

        last = round(hist["Close"].iloc[-1], 2)
        prev = round(hist["Close"].iloc[-2], 2)
        change_pct = round(((last - prev) / prev) * 100, 2)

        sma_20 = round(hist["Close"].rolling(20).mean().iloc[-1], 2)
        sma_50 = round(hist["Close"].rolling(50).mean().iloc[-1], 2)
        vol = int(hist["Volume"].iloc[-1])

        # High / Low
        high_52w = round(hist["Close"].max(), 2)
        low_52w = round(hist["Close"].min(), 2)

        if last > sma_20 > sma_50:
            signal = "BUY"
        elif last < sma_20 < sma_50:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "symbol": symbol,
            "last_price": last,
            "change_pct": change_pct,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "volume": vol,
            "high_3m": high_52w,
            "low_3m": low_52w,
            "signal": signal,
            "history": hist,  # pass full history for charts
        }
    except Exception as e:
        return {"error": str(e)}