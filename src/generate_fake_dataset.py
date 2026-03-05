import random, pandas as pd

verbs = ["guarantees", "delivers", "promises", "ensures", "generates", "creates", "provides"]
objects = ["massive profits", "instant wealth", "quick returns", "huge income", "unlimited cash",
           "10x returns", "guaranteed gains", "risk-free money", "secret profits"]
phrases = ["overnight", "with zero risk", "banks hide this", "100% guaranteed",
           "never fails", "government cover-up", "this one trick", "before it's banned",
           "while you sleep", "no investment needed", "using hidden loophole"]

rows = []
for _ in range(1500):
    rows.append({
        "text": f"This method {random.choice(verbs)} {random.choice(objects)} {random.choice(phrases)}",
        "label": 0,
    })

pd.DataFrame(rows).to_csv("data/raw/Fake.csv", index=False)
print("Fake dataset created:", len(rows), "rows")