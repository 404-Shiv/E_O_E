import requests, pandas as pd
from datetime import datetime


def mutual_fund_decision():
    """Fetch live mutual fund NAV from AMFI India with all scheme data."""
    try:
        text = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=20).text
        rows = text.split("\n")

        data = []
        current_category = ""
        for r in rows:
            c = r.split(";")
            if len(c) == 1 and r.strip():
                current_category = r.strip()
            elif len(c) > 5:
                try:
                    nav = float(c[4])
                    data.append({
                        "code": c[0].strip(),
                        "scheme": c[3].strip(),
                        "nav": nav,
                        "date": c[5].strip(),
                        "category": current_category,
                    })
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def search_mutual_funds(query, df):
    """Search mutual funds by name."""
    query = query.lower().strip()
    if not query:
        return df.head(50)
    mask = df["scheme"].str.lower().str.contains(query, na=False)
    return df[mask].head(50)


def get_mf_nav_history(scheme_code):
    """Fetch NAV history for a scheme from AMFI."""
    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        resp = requests.get(url, timeout=15).json()
        nav_data = resp.get("data", [])

        if not nav_data:
            return None, {}

        records = []
        for d in nav_data:  # all history
            try:
                records.append({
                    "date": datetime.strptime(d["date"], "%d-%m-%Y"),
                    "nav": float(d["nav"]),
                })
            except (ValueError, KeyError):
                continue

        if not records:
            return None, {}

        df_hist = pd.DataFrame(records).sort_values("date")

        # Calculate performance metrics
        latest_nav = df_hist["nav"].iloc[-1]
        metrics = {}

        # 1-month return
        if len(df_hist) >= 22:
            nav_1m = df_hist["nav"].iloc[-22]
            metrics["return_1m"] = round(((latest_nav - nav_1m) / nav_1m) * 100, 2)

        # 3-month return
        if len(df_hist) >= 66:
            nav_3m = df_hist["nav"].iloc[-66]
            metrics["return_3m"] = round(((latest_nav - nav_3m) / nav_3m) * 100, 2)

        # 6-month return
        if len(df_hist) >= 132:
            nav_6m = df_hist["nav"].iloc[-132]
            metrics["return_6m"] = round(((latest_nav - nav_6m) / nav_6m) * 100, 2)

        # 1-year return (XIRR approximation using simple return)
        if len(df_hist) >= 250:
            nav_1y = df_hist["nav"].iloc[-250]
            metrics["return_1y"] = round(((latest_nav - nav_1y) / nav_1y) * 100, 2)
        elif len(df_hist) >= 2:
            first_nav = df_hist["nav"].iloc[0]
            days = (df_hist["date"].iloc[-1] - df_hist["date"].iloc[0]).days
            if days > 0 and first_nav > 0:
                total_return = (latest_nav - first_nav) / first_nav
                annualized = ((1 + total_return) ** (365 / days) - 1) * 100
                metrics["return_1y"] = round(annualized, 2)

        # Volatility (std dev of daily returns)
        daily_returns = df_hist["nav"].pct_change().dropna()
        metrics["volatility"] = round(daily_returns.std() * (252 ** 0.5) * 100, 2)

        # Max drawdown
        peak = df_hist["nav"].expanding().max()
        drawdown = ((df_hist["nav"] - peak) / peak) * 100
        metrics["max_drawdown"] = round(drawdown.min(), 2)

        # Performance verdict
        r1y = metrics.get("return_1y", metrics.get("return_3m", 0))
        vol = metrics.get("volatility", 50)
        if r1y > 15 and vol < 20:
            metrics["verdict"] = "★ OUTPERFORMER"
        elif r1y > 8:
            metrics["verdict"] = "● GOOD PERFORMER"
        elif r1y > 0:
            metrics["verdict"] = "◆ NEUTRAL"
        elif r1y > -5:
            metrics["verdict"] = "▲ UNDERPERFORMER"
        else:
            metrics["verdict"] = "▼ POOR PERFORMER"

        return df_hist, metrics
    except Exception as e:
        return None, {"error": str(e)}