import { useState, useEffect } from 'react'
import axios from 'axios'

const FALLBACK_DATA = [
    { name: 'S&P 500', value: '...', change: '...', up: true },
    { name: 'NIFTY 50', value: '...', change: '...', up: true },
    { name: 'SENSEX', value: '...', change: '...', up: true },
    { name: 'DOW JONES', value: '...', change: '...', up: true },
    { name: 'NIKKEI', value: '...', change: '...', up: true },
    { name: 'BANK NIFTY', value: '...', change: '...', up: true },
    { name: 'NASDAQ', value: '...', change: '...', up: true },
]

export default function Ticker() {
    const [tickerData, setTickerData] = useState(FALLBACK_DATA)

    useEffect(() => {
        const fetchTicker = async () => {
            try {
                const { data } = await axios.get('/api/market/ticker')
                if (data && data.length > 0) {
                    setTickerData(data)
                }
            } catch (err) { /* ignore */ }
        }

        // Fetch immediately, then every 15 seconds
        fetchTicker()
        const interval = setInterval(fetchTicker, 15000)
        return () => clearInterval(interval)
    }, [])

    const items = [...tickerData, ...tickerData]

    return (
        <div className="ticker">
            <div className="ticker-track">
                {items.map((item, i) => (
                    <span key={i}>
                        <span className="ticker-item">
                            <span className="ticker-name">{item.name}</span>
                            {item.value}
                            <span className={item.up ? 'ticker-up' : 'ticker-down'}>
                                {item.up ? '▲' : '▼'} {item.change}
                            </span>
                        </span>
                        <span className="ticker-sep">│</span>
                    </span>
                ))}
            </div>
        </div>
    )
}
