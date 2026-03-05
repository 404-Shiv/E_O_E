import { useState, useMemo, useEffect } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

const pageMotion = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
}

const cardPop = {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1, transition: { duration: 0.3 } },
}

const TIME_RANGES = [
    { label: '1M', days: 30, value: '1m' },
    { label: '3M', days: 90, value: '3m' },
    { label: '6M', days: 180, value: '6m' },
    { label: '1Y', days: 365, value: '1y' },
    { label: '3Y', days: 1095, value: '3y' },
    { label: '5Y', days: 1825, value: '5y' },
    { label: 'All', days: 99999, value: 'all' },
]

export default function MutualFunds() {
    const [loaded, setLoaded] = useState(false)
    const [loadCount, setLoadCount] = useState(0)
    const [funds, setFunds] = useState([])
    const [query, setQuery] = useState('')
    const [selectedCode, setSelectedCode] = useState('')
    const [perfData, setPerfData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [analyzing, setAnalyzing] = useState(false)
    const [period, setPeriod] = useState('1y')
    const [showChart, setShowChart] = useState(false)
    const [liveNav, setLiveNav] = useState(null)
    const [navFlash, setNavFlash] = useState('')

    useEffect(() => {
        if (!perfData || !selectedCode || analyzing) return
        const interval = setInterval(async () => {
            try {
                const { data } = await axios.get(`/api/mf/price/${encodeURIComponent(selectedCode)}`)
                if (data.nav) {
                    setLiveNav(prev => {
                        if (prev && data.nav > prev) {
                            setNavFlash('flash-green')
                            setTimeout(() => setNavFlash(''), 1000)
                        } else if (prev && data.nav < prev) {
                            setNavFlash('flash-red')
                            setTimeout(() => setNavFlash(''), 1000)
                        }
                        return data.nav
                    })
                }
            } catch (e) { /* ignore */ }
        }, 15000)
        return () => clearInterval(interval)
    }, [perfData, selectedCode, analyzing])

    const loadFunds = async () => {
        setLoading(true)
        try {
            const { data } = await axios.post('/api/mf/load')
            setLoadCount(data.count)
            setLoaded(true)
            await searchFunds('')
        } catch (err) {
            setLoaded(false)
        }
        setLoading(false)
    }

    const searchFunds = async (q) => {
        setQuery(q)
        try {
            const { data } = await axios.get(`/api/mf/search?q=${encodeURIComponent(q)}`)
            setFunds(data)
            if (data.length > 0) setSelectedCode(data[0].code)
        } catch { setFunds([]) }
    }

    const analyzeFund = async () => {
        if (!selectedCode) return
        setAnalyzing(true)
        setPerfData(null)
        setShowChart(false)
        try {
            const { data } = await axios.get(`/api/mf/analyze/${encodeURIComponent(selectedCode)}`)
            setPerfData(data)
            if (data.history && data.history.length > 0) {
                setLiveNav(data.history[data.history.length - 1].nav)
            }
        } catch (err) {
            setPerfData({ metrics: { error: err.response?.data?.error || 'Failed' } })
        }
        setAnalyzing(false)
    }

    const verdictBadge = (verdict) => {
        const classMap = {
            '★ OUTPERFORMER': 'perf-outperformer',
            '● GOOD PERFORMER': 'perf-good',
            '◆ NEUTRAL': 'perf-neutral',
            '▲ UNDERPERFORMER': 'perf-under',
            '▼ POOR PERFORMER': 'perf-poor',
        }
        const colorMap = {
            '★ OUTPERFORMER': 'var(--green)',
            '● GOOD PERFORMER': '#34d399',
            '◆ NEUTRAL': 'var(--yellow)',
            '▲ UNDERPERFORMER': '#f97316',
            '▼ POOR PERFORMER': 'var(--red)',
        }
        return (
            <span style={{
                display: 'inline-block',
                padding: '8px 20px',
                borderRadius: 8,
                fontWeight: 800,
                fontSize: '1.1rem',
                color: colorMap[verdict] || 'var(--yellow)',
                background: 'var(--bg-input)',
                border: `1px solid ${colorMap[verdict] || 'var(--yellow)'}`,
            }}>
                {verdict}
            </span>
        )
    }

    const chartData = useMemo(() => {
        if (!perfData?.history) return []
        const days = TIME_RANGES.find(t => t.value === period)?.days || 365
        const cutoff = new Date()
        cutoff.setDate(cutoff.getDate() - days)
        return perfData.history.filter(d => new Date(d.date) >= cutoff)
    }, [perfData, period])

    // Get current NAV safely
    const currentNav = liveNav || (perfData?.history?.length > 0
        ? perfData.history[perfData.history.length - 1].nav
        : null)

    const yDomain = useMemo(() => {
        if (!chartData.length) return ['auto', 'auto']
        let min = Infinity, max = -Infinity
        chartData.forEach(d => {
            if (d.nav < min) min = d.nav
            if (d.nav > max) max = d.nav
        })
        const pad = (max - min) * 0.05
        return [Math.max(0, min - pad), max + pad]
    }, [chartData])

    return (
        <motion.div {...pageMotion}>
            <h1 className="section-title">Mutual Funds</h1>
            <p className="section-desc">Search mutual funds, view live NAV, and check performance ratings</p>
            <div className="divider" />

            {!loaded ? (
                <div style={{ textAlign: 'center', padding: 40 }}>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
                        Click below to fetch mutual fund schemes from AMFI India
                    </p>
                    <button className="btn btn-primary" onClick={loadFunds} disabled={loading}>
                        {loading ? <><span className="spinner" /> Loading...</> : 'Load Live Fund Data'}
                    </button>
                </div>
            ) : (
                <>
                    <div className="alert alert-success">Loaded {loadCount.toLocaleString()} mutual fund schemes</div>

                    <div className="input-group" style={{ marginTop: 16 }}>
                        <label>Search fund by name</label>
                        <input
                            className="input"
                            placeholder="e.g. HDFC, SBI, Axis, Nippon, large cap, flexi..."
                            value={query}
                            onChange={e => searchFunds(e.target.value)}
                        />
                    </div>

                    {funds.length > 0 && (
                        <>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>
                                Showing {funds.length} funds:
                            </div>
                            <div style={{ maxHeight: 350, overflow: 'auto', borderRadius: 'var(--radius-md)', marginBottom: 16 }}>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>Scheme Name</th>
                                            <th>NAV</th>
                                            <th>Date</th>
                                            <th>Category</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {funds.map((f, i) => (
                                            <tr key={i} style={{ cursor: 'pointer' }} onClick={() => setSelectedCode(f.code)}>
                                                <td style={{ fontWeight: selectedCode === f.code ? 700 : 400, color: selectedCode === f.code ? 'var(--accent)' : 'var(--text-primary)' }}>
                                                    {f.scheme}
                                                </td>
                                                <td>₹{f.nav}</td>
                                                <td>{f.date}</td>
                                                <td style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{f.category}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            <div className="divider" />
                            <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Fund Performance Analysis</h3>
                            <div style={{ display: 'flex', gap: 12, alignItems: 'end', flexWrap: 'wrap' }}>
                                <div className="input-group" style={{ flex: 1, margin: 0 }}>
                                    <label>Select a fund</label>
                                    <select className="input" value={selectedCode} onChange={e => setSelectedCode(e.target.value)}>
                                        {funds.slice(0, 20).map(f => (
                                            <option key={f.code} value={f.code}>{f.scheme} (Code: {f.code})</option>
                                        ))}
                                    </select>
                                </div>
                                <button className="btn btn-primary" onClick={analyzeFund} disabled={analyzing}>
                                    {analyzing ? <><span className="spinner" /> Analyzing...</> : 'Analyze Performance'}
                                </button>
                            </div>
                        </>
                    )}

                    {funds.length === 0 && query && (
                        <div className="alert alert-info">No funds found matching your search.</div>
                    )}

                    {/* Performance Results */}
                    {perfData && !perfData.metrics?.error && (
                        <motion.div initial="initial" animate="animate" style={{ marginTop: 24 }}>
                            {/* Summary Cards */}
                            <div className="grid-2" style={{ marginBottom: 20 }}>
                                <motion.div variants={cardPop} className={`card ${navFlash}`} style={{ textAlign: 'center' }}>
                                    <div className="card-header">Current NAV</div>
                                    <div className="card-value">₹{currentNav || '—'}</div>
                                    <div className="card-sub">Live updated end-of-day NAV</div>
                                </motion.div>
                                <motion.div variants={cardPop} className="card" style={{ textAlign: 'center' }}>
                                    <div className="card-header">Performance Rating</div>
                                    <div style={{ marginTop: 12 }}>{verdictBadge(perfData.metrics.verdict)}</div>
                                </motion.div>
                            </div>

                            <div className="metrics-grid">
                                {perfData.metrics.return_1m !== undefined && (
                                    <div className="metric-card hover-lift">
                                        <div className="metric-label">1M Return</div>
                                        <div className="metric-value">{perfData.metrics.return_1m}%</div>
                                        <div className={`metric-delta ${perfData.metrics.return_1m >= 0 ? 'positive' : 'negative'}`}>
                                            {perfData.metrics.return_1m >= 0 ? '▲' : '▼'} {perfData.metrics.return_1m}%
                                        </div>
                                    </div>
                                )}
                                {perfData.metrics.return_3m !== undefined && (
                                    <div className="metric-card hover-lift">
                                        <div className="metric-label">3M Return</div>
                                        <div className="metric-value">{perfData.metrics.return_3m}%</div>
                                        <div className={`metric-delta ${perfData.metrics.return_3m >= 0 ? 'positive' : 'negative'}`}>
                                            {perfData.metrics.return_3m >= 0 ? '▲' : '▼'} {perfData.metrics.return_3m}%
                                        </div>
                                    </div>
                                )}
                                {perfData.metrics.return_6m !== undefined && (
                                    <div className="metric-card hover-lift">
                                        <div className="metric-label">6M Return</div>
                                        <div className="metric-value">{perfData.metrics.return_6m}%</div>
                                        <div className={`metric-delta ${perfData.metrics.return_6m >= 0 ? 'positive' : 'negative'}`}>
                                            {perfData.metrics.return_6m >= 0 ? '▲' : '▼'} {perfData.metrics.return_6m}%
                                        </div>
                                    </div>
                                )}
                                {perfData.metrics.return_1y !== undefined && (
                                    <div className="metric-card hover-lift">
                                        <div className="metric-label">1Y Return (XIRR)</div>
                                        <div className="metric-value">{perfData.metrics.return_1y}%</div>
                                        <div className={`metric-delta ${perfData.metrics.return_1y >= 0 ? 'positive' : 'negative'}`}>
                                            {perfData.metrics.return_1y >= 0 ? '▲' : '▼'} {perfData.metrics.return_1y}%
                                        </div>
                                    </div>
                                )}
                                {perfData.metrics.volatility !== undefined && (
                                    <div className="metric-card hover-lift">
                                        <div className="metric-label">Volatility (Annual)</div>
                                        <div className="metric-value">{perfData.metrics.volatility}%</div>
                                    </div>
                                )}
                                {perfData.metrics.max_drawdown !== undefined && (
                                    <div className="metric-card hover-lift">
                                        <div className="metric-label">Max Drawdown</div>
                                        <div className="metric-value">{perfData.metrics.max_drawdown}%</div>
                                    </div>
                                )}
                            </div>

                            <div style={{ marginTop: 24, textAlign: 'center' }}>
                                <button className="btn btn-outline" onClick={() => setShowChart(!showChart)}>
                                    {showChart ? 'Hide NAV Chart' : 'Show NAV Chart'}
                                </button>
                            </div>

                            {/* NAV History Chart */}
                            {showChart && chartData.length > 0 && (
                                <motion.div variants={cardPop} initial="initial" animate="animate" className="chart-container" style={{ marginTop: 16 }}>
                                    <div className="chart-controls" style={{ marginBottom: 12 }}>
                                        <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>NAV History</div>
                                        <div className="time-range-pills">
                                            {TIME_RANGES.map(t => (
                                                <button
                                                    key={t.value}
                                                    className={`time-pill ${period === t.value ? 'active' : ''}`}
                                                    onClick={() => setPeriod(t.value)}
                                                >
                                                    {t.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                    <ResponsiveContainer width="100%" height={280}>
                                        <LineChart data={chartData}>
                                            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" opacity={0.3} />
                                            <XAxis
                                                dataKey="date"
                                                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                                tickFormatter={d => {
                                                    if (period === 'all' || period === '5y' || period === '3y') return d.slice(0, 4) // YYYY
                                                    if (period === '1m') return d.slice(5) // MM-DD
                                                    return d.slice(0, 7) // YYYY-MM
                                                }}
                                                minTickGap={30}
                                            />
                                            <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} domain={yDomain} tickFormatter={v => `₹${v.toFixed(1)}`} />
                                            <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }} labelStyle={{ color: 'var(--text-secondary)' }} />
                                            <Line type="monotone" dataKey="nav" stroke="var(--accent)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </motion.div>
                            )}
                        </motion.div>
                    )}

                    {perfData?.metrics?.error && (
                        <div className="alert alert-error" style={{ marginTop: 16 }}>{perfData.metrics.error}</div>
                    )}
                </>
            )}
        </motion.div>
    )
}
