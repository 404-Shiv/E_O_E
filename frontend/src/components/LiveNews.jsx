import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const pageMotion = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
}

const stagger = {
    animate: { transition: { staggerChildren: 0.04 } },
}

const fadeUp = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

export default function LiveNews() {
    const [articles, setArticles] = useState([])
    const [stats, setStats] = useState({ total: 0, true_count: 0, fake_count: 0 })
    const [filter, setFilter] = useState('All')
    const [sort, setSort] = useState('Latest')
    const [loading, setLoading] = useState(true)
    const [refreshing, setRefreshing] = useState(false)

    const fetchNews = async () => {
        setLoading(true)
        try {
            const { data } = await axios.get('/api/news')
            setArticles(data.articles || [])
            setStats({ total: data.total || 0, true_count: data.true_count || 0, fake_count: data.fake_count || 0 })
        } catch {
            setArticles([])
        }
        setLoading(false)
    }

    useEffect(() => { fetchNews() }, [])

    const handleRefresh = async () => {
        setRefreshing(true)
        try {
            await axios.post('/api/news/refresh')
            await fetchNews()
        } catch { /* ignore */ }
        setRefreshing(false)
    }

    let filtered = [...articles]
    if (filter === 'TRUE Only') filtered = filtered.filter(a => a.verdict === 'TRUE')
    if (filter === 'FAKE Only') filtered = filtered.filter(a => a.verdict === 'FAKE')
    if (sort === 'Highest Score') filtered.sort((a, b) => b.truth_score - a.truth_score)
    if (sort === 'Lowest Score') filtered.sort((a, b) => a.truth_score - b.truth_score)

    return (
        <motion.div {...pageMotion}>
            <h1 className="section-title">Live News Feed</h1>
            <p className="section-desc">Real-time financial news classified by our AI model</p>
            <div className="divider" />

            {loading ? (
                <div className="loading-overlay">
                    <span className="spinner" />
                    Loading news...
                </div>
            ) : (
                <>
                    <div className="grid-3" style={{ marginBottom: 16 }}>
                        <div className="metric-card">
                            <div className="metric-label">Total Articles</div>
                            <div className="metric-value">{stats.total}</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">True News</div>
                            <div className="metric-value" style={{ color: 'var(--green)' }}>{stats.true_count}</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">Fake News</div>
                            <div className="metric-value" style={{ color: 'var(--red)' }}>{stats.fake_count}</div>
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap', alignItems: 'end' }}>
                        <div className="input-group" style={{ margin: 0, flex: '1 1 180px' }}>
                            <label>Filter</label>
                            <select className="input" value={filter} onChange={e => setFilter(e.target.value)}>
                                <option>All</option>
                                <option>TRUE Only</option>
                                <option>FAKE Only</option>
                            </select>
                        </div>
                        <div className="input-group" style={{ margin: 0, flex: '1 1 180px' }}>
                            <label>Sort by</label>
                            <select className="input" value={sort} onChange={e => setSort(e.target.value)}>
                                <option>Latest</option>
                                <option>Highest Score</option>
                                <option>Lowest Score</option>
                            </select>
                        </div>
                        <button className="btn btn-outline" onClick={handleRefresh} disabled={refreshing}>
                            {refreshing ? <><span className="spinner" /> Refreshing...</> : '↻ Refresh Feed'}
                        </button>
                    </div>

                    <div className="divider" />

                    {filtered.length === 0 ? (
                        <div className="alert alert-info">No news articles found. Click Refresh to fetch news.</div>
                    ) : (
                        <motion.div variants={stagger} initial="initial" animate="animate">
                            {filtered.map((article, i) => (
                                <motion.div
                                    key={i}
                                    variants={fadeUp}
                                    className={`news-item ${article.verdict === 'TRUE' ? 'true' : 'fake'}`}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span className={`badge ${article.verdict === 'TRUE' ? 'badge-true' : 'badge-fake'}`}>
                                            {article.verdict === 'TRUE' ? '●' : '✕'} {article.verdict} — {article.truth_score}%
                                        </span>
                                        {article.url && article.url.startsWith('http') && (
                                            <a
                                                href={article.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="read-article-link"
                                                onClick={e => e.stopPropagation()}
                                            >
                                                ⟶ Read article ↗
                                            </a>
                                        )}
                                    </div>
                                    {article.url && article.url.startsWith('http') ? (
                                        <a
                                            href={article.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="news-title-link"
                                        >
                                            {article.text}
                                        </a>
                                    ) : (
                                        <div className="news-title">{article.text}</div>
                                    )}
                                    <div className="news-meta">
                                        <span>⊙ {article.source}</span>
                                        <span>◷ {article.date}</span>
                                    </div>
                                </motion.div>
                            ))}
                        </motion.div>
                    )}
                </>
            )}
        </motion.div>
    )
}
