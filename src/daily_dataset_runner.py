import time
from realtime_multi_news_fetch import collect

while True:
    collect()
    time.sleep(86400)  # daily