import requests
import pandas as pd

def mutual_fund_decision():
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    rows = requests.get(url).text.split("\n")

    data = []
    for r in rows:
        c = r.split(";")
        if len(c) > 5:
            data.append({
                "scheme": c[3],
                "nav": c[4],
                "date": c[5]
            })

    return pd.DataFrame(data).head()
