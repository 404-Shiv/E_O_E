import { useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

const pageMotion = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.2 } },
}

const cardPop = {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1, transition: { duration: 0.4, ease: [0.34, 1.56, 0.64, 1] } },
}

export default function NewsVerifier() {
    const [text, setText] = useState('')
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)

    const handleVerify = async () => {
        if (!text.trim()) return
        setLoading(true)
        setResult(null)
        try {
            const { data } = await axios.post('/api/verify', { text })
            setResult(data)
        } catch (err) {
            setResult({ error: err.response?.data?.error || 'Verification failed' })
        }
        setLoading(false)
    }

    return (
        <motion.div {...pageMotion}>
            <h1 className="section-title">News Verifier</h1>
            <p className="section-desc">Paste any financial news headline to check if it's genuine or fake</p>
            <div className="divider" />

            <div className="input-group">
                <label>Enter a news headline or article text</label>
                <textarea
                    className="input"
                    rows={5}
                    placeholder="e.g.  RBI keeps repo rate unchanged at 6.5% amid inflation concerns..."
                    value={text}
                    onChange={e => setText(e.target.value)}
                />
            </div>

            <button
                className="btn btn-primary"
                onClick={handleVerify}
                disabled={loading || !text.trim()}
                style={{ marginBottom: 8 }}
            >
                {loading ? <><span className="spinner" /> Verifying...</> : 'Verify'}
            </button>

            {result && !result.error && (
                <motion.div initial="initial" animate="animate" style={{ marginTop: 24 }}>
                    <div className="grid-2">
                        {/* Verdict Card */}
                        <motion.div variants={cardPop} className="card" style={{ textAlign: 'center' }}>
                            {result.verdict === 'TRUE' ? (
                                <>
                                    <span className="badge badge-true" style={{ fontSize: '0.9rem', padding: '6px 20px' }}>● LIKELY TRUE</span>
                                    <div className="card-value" style={{ color: 'var(--green)', marginTop: 16 }}>{result.truth_pct}%</div>
                                    <div className="card-sub">Truth Confidence</div>
                                </>
                            ) : (
                                <>
                                    <span className="badge badge-fake" style={{ fontSize: '0.9rem', padding: '6px 20px' }}>✕ LIKELY FAKE</span>
                                    <div className="card-value" style={{ color: 'var(--red)', marginTop: 16 }}>{result.fake_pct}%</div>
                                    <div className="card-sub">Fake Confidence</div>
                                </>
                            )}
                        </motion.div>

                        {/* Score Bars */}
                        <motion.div variants={cardPop} className="card">
                            <div className="card-header">Truth Score Distribution</div>
                            <div style={{ marginTop: 12 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                                    <span>True</span>
                                    <span>{result.truth_pct}%</span>
                                </div>
                                <div className="score-bar-bg">
                                    <motion.div
                                        className="score-bar-fill"
                                        style={{ background: 'linear-gradient(90deg, #059669, #34d399)' }}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${result.truth_pct}%` }}
                                        transition={{ duration: 0.8, ease: [0.34, 1.56, 0.64, 1] }}
                                    >
                                        {result.truth_pct > 15 ? `${result.truth_pct}%` : ''}
                                    </motion.div>
                                </div>
                            </div>
                            <div style={{ marginTop: 12 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                                    <span>Fake</span>
                                    <span>{result.fake_pct}%</span>
                                </div>
                                <div className="score-bar-bg">
                                    <motion.div
                                        className="score-bar-fill"
                                        style={{ background: 'linear-gradient(90deg, #dc2626, #f87171)' }}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${result.fake_pct}%` }}
                                        transition={{ duration: 0.8, ease: [0.34, 1.56, 0.64, 1], delay: 0.1 }}
                                    >
                                        {result.fake_pct > 15 ? `${result.fake_pct}%` : ''}
                                    </motion.div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </motion.div>
            )}

            {result?.error && (
                <div className="alert alert-error" style={{ marginTop: 16 }}>{result.error}</div>
            )}
        </motion.div>
    )
}
