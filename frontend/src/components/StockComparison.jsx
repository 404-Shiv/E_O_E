import { useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

const pageMotion = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
}

const cardPop = {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1, transition: { duration: 0.3 } },
}

export default function StockComparison() {
    const [queryA, setQueryA] = useState('')
    const [queryB, setQueryB] = useState('')
    const [resultsA, setResultsA] = useState([])
    const [resultsB, setResultsB] = useState([])
    const [symA, setSymA] = useState('')
    const [symB, setSymB] = useState('')
    const [dataA, setDataA] = useState(null)
    const [dataB, setDataB] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const searchStock = async (q, setter, symSetter) => {
        if (q.trim().length < 2) { setter([]); return }
        try {
            const { data } = await axios.get(`/api/stocks/search?q=${encodeURIComponent(q)}`)
            setter(data)
            if (data.length > 0) symSetter(data[0].symbol)
        } catch { setter([]) }
    }

    const handleCompare = async () => {
        if (!symA || !symB) { setError('Enter both stock symbols'); return }
        setError('')
        setLoading(true)
        setDataA(null)
        setDataB(null)
        try {
            const [resA, resB] = await Promise.all([
                axios.get(`/api/stocks/analyze/${encodeURIComponent(symA)}`),
                axios.get(`/api/stocks/analyze/${encodeURIComponent(symB)}`),
            ])
            if (resA.data.error) { setError(`Stock A: ${resA.data.error}`); setLoading(false); return }
            if (resB.data.error) { setError(`Stock B: ${resB.data.error}`); setLoading(false); return }
            setDataA(resA.data)
            setDataB(resB.data)
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to fetch data')
        }
        setLoading(false)
    }

    const sigBadge = (signal) => {
        const cls = { BUY: 'badge-buy', SELL: 'badge-sell', HOLD: 'badge-hold' }
        const emoji = { BUY: '▲', SELL: '▼', HOLD: '◆' }
        return <span className={`badge ${cls[signal] || ''}`}>{emoji[signal]} {signal}</span>
    }

    const metrics = [
        { label: 'Last Price', key: 'last_price', prefix: '₹' },
        { label: 'Change %', key: 'change_pct', suffix: '%' },
        { label: 'SMA 20', key: 'sma_20', prefix: '₹' },
        { label: 'SMA 50', key: 'sma_50', prefix: '₹' },
        { label: '3M High', key: 'high_3m', prefix: '₹' },
        { label: '3M Low', key: 'low_3m', prefix: '₹' },
        { label: 'Volume', key: 'volume', format: v => v?.toLocaleString() },
    ]

    // Merge chart data
    let chartData = []
    if (dataA?.history && dataB?.history) {
        const mapA = Object.fromEntries(dataA.history.map(h => [h.date, h.close]))
        const mapB = Object.fromEntries(dataB.history.map(h => [h.date, h.close]))
        const allDates = [...new Set([...Object.keys(mapA), ...Object.keys(mapB)])].sort()
        chartData = allDates.map(d => ({ date: d, [dataA.symbol]: mapA[d], [dataB.symbol]: mapB[d] }))
    }

    return (
        <motion.div {...pageMotion}>
            <h1 className="section-title">Stock Comparison</h1>
            <p className="section-desc">Compare two stocks side by side with real-time metrics and charts</p>
            <div className="divider" />

            <div className="grid-2">
                {/* Stock A */}
                <div>
                    <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Stock A</h3>
                    <div className="input-group">
                        <label>Search company A</label>
                        <input className="input" placeholder="e.g. tata" value={queryA}
                            onChange={e => { setQueryA(e.target.value); searchStock(e.target.value, setResultsA, setSymA) }} />
                    </div>
                    {resultsA.length > 0 ? (
                        <select className="input" value={symA} onChange={e => setSymA(e.target.value)}>
                            {resultsA.map(s => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
                        </select>
                    ) : (
                        <div className="input-group">
                            <label>Symbol A</label>
                            <input className="input" placeholder="INFY.NS" value={symA} onChange={e => setSymA(e.target.value)} />
                        </div>
                    )}
                </div>

                {/* Stock B */}
                <div>
                    <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Stock B</h3>
                    <div className="input-group">
                        <label>Search company B</label>
                        <input className="input" placeholder="e.g. infosys" value={queryB}
                            onChange={e => { setQueryB(e.target.value); searchStock(e.target.value, setResultsB, setSymB) }} />
                    </div>
                    {resultsB.length > 0 ? (
                        <select className="input" value={symB} onChange={e => setSymB(e.target.value)}>
                            {resultsB.map(s => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
                        </select>
                    ) : (
                        <div className="input-group">
                            <label>Symbol B</label>
                            <input className="input" placeholder="TCS.NS" value={symB} onChange={e => setSymB(e.target.value)} />
                        </div>
                    )}
                </div>
            </div>

            <button className="btn btn-primary btn-full" style={{ marginTop: 16 }} onClick={handleCompare} disabled={loading}>
                {loading ? <><span className="spinner" /> Comparing...</> : 'Compare Stocks'}
            </button>

            {error && <div className="alert alert-error" style={{ marginTop: 12 }}>{error}</div>}

            {/* Results */}
            {dataA && dataB && (
                <motion.div initial="initial" animate="animate" style={{ marginTop: 24 }}>
                    {/* Signal Cards */}
                    <div className="grid-2" style={{ marginBottom: 20 }}>
                        <motion.div variants={cardPop} className="card" style={{ textAlign: 'center' }}>
                            <div className="card-header">Signal</div>
                            <div className="card-value">{dataA.symbol}</div>
                            <div style={{ marginTop: 8 }}>{sigBadge(dataA.signal)}</div>
                        </motion.div>
                        <motion.div variants={cardPop} className="card" style={{ textAlign: 'center' }}>
                            <div className="card-header">Signal</div>
                            <div className="card-value">{dataB.symbol}</div>
                            <div style={{ marginTop: 8 }}>{sigBadge(dataB.signal)}</div>
                        </motion.div>
                    </div>

                    {/* Metrics Table */}
                    <div className="divider" />
                    <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Side-by-Side Metrics</h3>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>{dataA.symbol}</th>
                                <th>{dataB.symbol}</th>
                            </tr>
                        </thead>
                        <tbody>
                            {metrics.map(m => {
                                const va = dataA[m.key]
                                const vb = dataB[m.key]
                                const fmtA = m.format ? m.format(va) : `${m.prefix || ''}${va}${m.suffix || ''}`
                                const fmtB = m.format ? m.format(vb) : `${m.prefix || ''}${vb}${m.suffix || ''}`
                                return (
                                    <tr key={m.key}>
                                        <td style={{ fontWeight: 600 }}>{m.label}</td>
                                        <td>{fmtA}</td>
                                        <td>{fmtB}</td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>

                    {/* Chart Overlay */}
                    {chartData.length > 0 && (
                        <motion.div variants={cardPop} className="chart-container" style={{ marginTop: 20 }}>
                            <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>Price Chart Comparison</div>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={chartData}>
                                    <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                                    <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickFormatter={d => d.slice(5)} />
                                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} domain={['auto', 'auto']} />
                                    <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }} />
                                    <Legend />
                                    <Line type="monotone" dataKey={dataA.symbol} stroke="#3b82f6" strokeWidth={2} dot={false} />
                                    <Line type="monotone" dataKey={dataB.symbol} stroke="#f59e0b" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </motion.div>
                    )}
                </motion.div>
            )}
        </motion.div>
    )
}
