import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'
import {
    ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
    CartesianGrid, Cell, ReferenceLine
} from 'recharts'

const pageMotion = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
}

const cardPop = {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1, transition: { duration: 0.3, ease: 'easeOut' } },
}

const TIME_RANGES = [
    { label: '1M', value: '1mo' },
    { label: '3M', value: '3mo' },
    { label: '6M', value: '6mo' },
    { label: '1Y', value: '1y' },
    { label: '3Y', value: '3y' },
    { label: '5Y', value: '5y' },
    { label: 'All', value: 'max' },
]

/* ── Custom Candlestick Shape ── */
const CandlestickBar = (props) => {
    const { x, y, width, height, payload } = props
    if (!payload) return null
    const { open, close, high, low } = payload
    const isUp = close >= open
    const color = isUp ? 'var(--green)' : 'var(--red)'
    const barWidth = Math.max(width * 0.6, 3)
    const barX = x + (width - barWidth) / 2
    const wickX = x + width / 2

    // Calculate pixel positions using Y axis scale
    const yScale = props.yScale || props.background?.yScale
    // We use the y and height from recharts and map OHLC manually
    const yDomain = props.yAxisDomain
    if (!yDomain) return null

    const [yMin, yMax] = yDomain
    const plotHeight = props.plotHeight
    const plotY = props.plotY || 0

    const toY = (val) => plotY + plotHeight - ((val - yMin) / (yMax - yMin)) * plotHeight

    const bodyTop = toY(Math.max(open, close))
    const bodyBottom = toY(Math.min(open, close))
    const bodyHeight = Math.max(bodyBottom - bodyTop, 1)
    const wickTop = toY(high)
    const wickBottom = toY(low)

    return (
        <g>
            {/* Wick */}
            <line x1={wickX} y1={wickTop} x2={wickX} y2={wickBottom}
                stroke={color} strokeWidth={1.2} />
            {/* Body */}
            <rect x={barX} y={bodyTop} width={barWidth} height={bodyHeight}
                fill={isUp ? color : color} stroke={color} strokeWidth={0.5}
                rx={1} />
        </g>
    )
}

/* ── Custom Tooltip ── */
const ChartTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null
    const d = payload[0]?.payload
    if (!d) return null
    const isUp = d.close >= d.open
    return (
        <div className="chart-tooltip">
            <div className="chart-tooltip-date">{label}</div>
            <div className="chart-tooltip-row">
                <span>Open</span><span>₹{d.open?.toLocaleString()}</span>
            </div>
            <div className="chart-tooltip-row">
                <span>High</span><span style={{ color: 'var(--green)' }}>₹{d.high?.toLocaleString()}</span>
            </div>
            <div className="chart-tooltip-row">
                <span>Low</span><span style={{ color: 'var(--red)' }}>₹{d.low?.toLocaleString()}</span>
            </div>
            <div className="chart-tooltip-row">
                <span>Close</span><span style={{ color: isUp ? 'var(--green)' : 'var(--red)' }}>₹{d.close?.toLocaleString()}</span>
            </div>
            <div className="chart-tooltip-row">
                <span>Volume</span><span>{d.volume?.toLocaleString()}</span>
            </div>
        </div>
    )
}

export default function StockAnalysis() {
    const [query, setQuery] = useState('')
    const [searchResults, setSearchResults] = useState([])
    const [selectedSymbol, setSelectedSymbol] = useState('')
    const [directSymbol, setDirectSymbol] = useState('')
    const [stockData, setStockData] = useState(null)
    const [companyNews, setCompanyNews] = useState([])
    const [loading, setLoading] = useState(false)
    const [period, setPeriod] = useState('3mo')
    const [chartType, setChartType] = useState('candle') // 'candle' or 'line'
    const [currentSymbol, setCurrentSymbol] = useState('')
    const [livePrice, setLivePrice] = useState(null)
    const [priceFlash, setPriceFlash] = useState('')

    useEffect(() => {
        if (!currentSymbol || loading) return
        const interval = setInterval(async () => {
            try {
                const { data } = await axios.get(`/api/stocks/price/${encodeURIComponent(currentSymbol)}`)
                if (data.price) {
                    setLivePrice(prev => {
                        if (prev && data.price > prev) {
                            setPriceFlash('flash-green')
                            setTimeout(() => setPriceFlash(''), 1000)
                        } else if (prev && data.price < prev) {
                            setPriceFlash('flash-red')
                            setTimeout(() => setPriceFlash(''), 1000)
                        }
                        return data.price
                    })
                }
            } catch (e) { /* ignore */ }
        }, 5000)
        return () => clearInterval(interval)
    }, [currentSymbol, loading])

    const handleSearch = async (q) => {
        setQuery(q)
        if (q.trim().length < 2) { setSearchResults([]); return }
        try {
            const { data } = await axios.get(`/api/stocks/search?q=${encodeURIComponent(q)}`)
            setSearchResults(data)
            if (data.length > 0) setSelectedSymbol(data[0].symbol)
        } catch { setSearchResults([]) }
    }

    const analyzeStock = async (symbol, p = period) => {
        if (!symbol) return
        setLoading(true)
        setStockData(null)
        setCompanyNews([])
        setCurrentSymbol(symbol)
        try {
            const { data } = await axios.get(`/api/stocks/analyze/${encodeURIComponent(symbol)}?period=${p}`)
            setStockData(data)
            setLivePrice(data.last_price || null)
            // Fetch company news
            try {
                const newsRes = await axios.get(`/api/stocks/news/${encodeURIComponent(symbol)}`)
                setCompanyNews(newsRes.data || [])
            } catch { /* ignore */ }
        } catch (err) {
            setStockData({ error: err.response?.data?.error || 'Failed to fetch data' })
        }
        setLoading(false)
    }

    const handlePeriodChange = (newPeriod) => {
        setPeriod(newPeriod)
        if (currentSymbol) {
            analyzeStock(currentSymbol, newPeriod)
        }
    }

    const sigBadge = (signal) => {
        const map = { BUY: 'badge-buy', SELL: 'badge-sell', HOLD: 'badge-hold' }
        const emoji = { BUY: '▲', SELL: '▼', HOLD: '◆' }
        return (
            <span className={`badge ${map[signal] || ''}`}>
                {emoji[signal] || ''} {signal}
            </span>
        )
    }

    /* Compute Y domain for candlestick */
    const getYDomain = (history) => {
        if (!history || !history.length) return [0, 100]
        let min = Infinity, max = -Infinity
        history.forEach(d => {
            if (d.low < min) min = d.low
            if (d.high > max) max = d.high
        })
        const pad = (max - min) * 0.05
        return [Math.floor(min - pad), Math.ceil(max + pad)]
    }

    /* Determine tick format based on data length */
    const getTickFormatter = (history) => {
        if (!history) return d => d
        const len = history.length
        if (len <= 60) return d => d.slice(5) // MM-DD
        if (len <= 365) return d => d.slice(0, 7) // YYYY-MM
        return d => d.slice(0, 4) // YYYY
    }

    /* Downsample data for very long ranges */
    const getChartData = (history) => {
        if (!history) return []
        if (history.length <= 250) return history
        // Sample every Nth item
        const step = Math.ceil(history.length / 250)
        return history.filter((_, i) => i % step === 0 || i === history.length - 1)
    }

    /* Render Candlestick Chart */
    const renderCandlestickChart = (history) => {
        const data = getChartData(history)
        const yDomain = getYDomain(data)
        const tickFmt = getTickFormatter(data)

        return (
            <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" opacity={0.3} />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        tickFormatter={tickFmt}
                        interval={Math.max(Math.floor(data.length / 8), 0)}
                    />
                    <YAxis
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        domain={yDomain}
                        tickFormatter={v => `₹${v.toLocaleString()}`}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    {/* Invisible bar for hover detection */}
                    <Bar dataKey="high" fill="transparent" barSize={12}
                        shape={(props) => {
                            const chartEl = props.background
                            const plotHeight = chartEl?.height || 300
                            const plotY = chartEl?.y || 0
                            return (
                                <CandlestickBar
                                    {...props}
                                    yAxisDomain={yDomain}
                                    plotHeight={plotHeight}
                                    plotY={plotY}
                                />
                            )
                        }}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        )
    }

    /* Render Line Chart */
    const renderLineChart = (history) => {
        const data = getChartData(history)
        const tickFmt = getTickFormatter(data)

        return (
            <ResponsiveContainer width="100%" height={380}>
                <ComposedChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                    <defs>
                        <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.02} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" opacity={0.3} />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        tickFormatter={tickFmt}
                        interval={Math.max(Math.floor(data.length / 8), 0)}
                    />
                    <YAxis
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        domain={['auto', 'auto']}
                        tickFormatter={v => `₹${v.toLocaleString()}`}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey="close" stroke="var(--accent)" strokeWidth={2}
                        dot={false} activeDot={{ r: 4, fill: 'var(--accent)' }} />
                </ComposedChart>
            </ResponsiveContainer>
        )
    }

    const periodLabel = TIME_RANGES.find(t => t.value === period)?.label || '3M'

    // Compute live daily change (compare live price to last available previous close)
    let liveChange = null
    let liveChangePct = null
    let isPositive = true
    if (stockData && stockData.history && stockData.history.length >= 2) {
        // Current display price
        const currentP = livePrice || stockData.last_price
        // Previous day's close
        const prevClose = stockData.history[stockData.history.length - 2].close
        if (prevClose > 0) {
            liveChange = currentP - prevClose
            liveChangePct = (liveChange / prevClose) * 100
            isPositive = liveChange >= 0
        }
    } else if (stockData && stockData.change_pct !== undefined) {
        // Fallback to static backend data if history is too short
        liveChangePct = stockData.change_pct
        // Rough estimate of change amount if we only have pct and current price
        const currentP = livePrice || stockData.last_price
        liveChange = currentP - (currentP / (1 + (liveChangePct / 100)))
        isPositive = liveChangePct >= 0
    }

    return (
        <motion.div {...pageMotion}>
            <h1 className="section-title">Stock Analysis</h1>
            <p className="section-desc">Search any company to view real-time price, SMA signals, and recommendation</p>
            <div className="divider" />

            {/* Search */}
            <div className="input-group">
                <label>Search company or sector</label>
                <input
                    className="input"
                    placeholder="e.g. tata, reliance, bank, pharma, auto, hdfc..."
                    value={query}
                    onChange={e => handleSearch(e.target.value)}
                />
            </div>

            {searchResults.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                    <label style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', fontWeight: 600, marginBottom: 6, display: 'block' }}>
                        {searchResults.length} matching stocks:
                    </label>
                    <select
                        className="input"
                        value={selectedSymbol}
                        onChange={e => setSelectedSymbol(e.target.value)}
                    >
                        {searchResults.map(s => (
                            <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>
                        ))}
                    </select>
                    <button
                        className="btn btn-primary btn-full"
                        style={{ marginTop: 12 }}
                        onClick={() => analyzeStock(selectedSymbol)}
                        disabled={loading}
                    >
                        {loading ? <><span className="spinner" /> Analyzing...</> : 'Analyze Selected Stock'}
                    </button>
                </div>
            )}

            <div className="divider" />
            <div style={{ marginBottom: 8, fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
                Or enter a symbol directly:
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
                <input
                    className="input"
                    placeholder="e.g. INFY.NS"
                    value={directSymbol}
                    onChange={e => setDirectSymbol(e.target.value)}
                    style={{ flex: 1 }}
                />
                <button
                    className="btn btn-primary"
                    onClick={() => analyzeStock(directSymbol.trim())}
                    disabled={loading}
                >
                    Analyze
                </button>
            </div>

            {/* Results */}
            {loading && (
                <div className="loading-overlay">
                    <span className="spinner" /> Fetching stock data...
                </div>
            )}

            {stockData && !stockData.error && (
                <motion.div initial="initial" animate="animate" style={{ marginTop: 24 }}>

                    {/* Real-time Hero Banner */}
                    <motion.div variants={cardPop} className={`hero-container ${priceFlash}`}>
                        <div className="hero-title-group">
                            <div className="hero-logo-placeholder">
                                {stockData.symbol.charAt(0)}
                            </div>
                            <div>
                                <div className="hero-title">{stockData.symbol.replace('.NS', '')}</div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                                    <span className="hero-symbol">{stockData.symbol}</span>
                                    {sigBadge(stockData.signal)}
                                </div>
                            </div>
                        </div>

                        <div className="hero-price-group">
                            <span className="hero-price">
                                {livePrice ? livePrice.toLocaleString() : stockData.last_price.toLocaleString()}
                            </span>
                            <span className="hero-currency">INR</span>

                            {liveChange !== null && (
                                <span className={`hero-change ${isPositive ? 'positive' : 'negative'}`}>
                                    {isPositive ? '+' : ''}{liveChange.toFixed(2)}
                                    ({isPositive ? '+' : ''}{liveChangePct.toFixed(2)}%)
                                    {isPositive ? ' ▲' : ' ▼'}
                                </span>
                            )}
                        </div>
                        <div className="hero-meta">
                            Real-time live updates {livePrice ? 'Active' : 'Fetching'} • GMT+5:30
                        </div>
                    </motion.div>

                    <div className="metrics-grid">
                        {/* Last Price and Change% moved to Hero Banner */}
                        <motion.div variants={cardPop} className="metric-card">
                            <div className="metric-label">SMA 20</div>
                            <div className="metric-value">₹{stockData.sma_20}</div>
                        </motion.div>
                        <motion.div variants={cardPop} className="metric-card">
                            <div className="metric-label">SMA 50</div>
                            <div className="metric-value">₹{stockData.sma_50}</div>
                        </motion.div>
                        <motion.div variants={cardPop} className="metric-card">
                            <div className="metric-label">{periodLabel} High</div>
                            <div className="metric-value">₹{stockData.high_3m}</div>
                        </motion.div>
                        <motion.div variants={cardPop} className="metric-card">
                            <div className="metric-label">{periodLabel} Low</div>
                            <div className="metric-value">₹{stockData.low_3m}</div>
                        </motion.div>
                        <motion.div variants={cardPop} className="metric-card">
                            <div className="metric-label">Volume</div>
                            <div className="metric-value">{stockData.volume?.toLocaleString()}</div>
                        </motion.div>
                    </div>

                    {/* Price Chart */}
                    {stockData.history && stockData.history.length > 0 && (
                        <motion.div variants={cardPop} className="chart-container" style={{ marginTop: 20 }}>
                            {/* Chart Controls Bar */}
                            <div className="chart-controls">
                                <div className="chart-controls-left">
                                    <div className="chart-type-toggle">
                                        <button
                                            className={`chart-type-btn ${chartType === 'candle' ? 'active' : ''}`}
                                            onClick={() => setChartType('candle')}
                                        >
                                            Candle
                                        </button>
                                        <button
                                            className={`chart-type-btn ${chartType === 'line' ? 'active' : ''}`}
                                            onClick={() => setChartType('line')}
                                        >
                                            Line
                                        </button>
                                    </div>
                                </div>
                                <div className="time-range-pills">
                                    {TIME_RANGES.map(t => (
                                        <button
                                            key={t.value}
                                            className={`time-pill ${period === t.value ? 'active' : ''}`}
                                            onClick={() => handlePeriodChange(t.value)}
                                            disabled={loading}
                                        >
                                            {t.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Chart */}
                            <div style={{ marginTop: 12 }}>
                                {chartType === 'candle'
                                    ? renderCandlestickChart(stockData.history)
                                    : renderLineChart(stockData.history)
                                }
                            </div>
                        </motion.div>
                    )}

                    {/* Company News */}
                    {companyNews.length > 0 && (
                        <div style={{ marginTop: 24 }}>
                            <div className="divider" />
                            <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Related News</h3>
                            {companyNews.slice(0, 5).map((article, i) => (
                                <motion.div
                                    key={i}
                                    variants={cardPop}
                                    initial="initial"
                                    animate="animate"
                                    className="company-news-item"
                                >
                                    <a href={article.url} target="_blank" rel="noopener noreferrer">{article.title}</a>
                                    <div className="src">⊙ {article.source}</div>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </motion.div>
            )}

            {stockData?.error && (
                <div className="alert alert-error" style={{ marginTop: 16 }}>{stockData.error}</div>
            )}
        </motion.div>
    )
}
