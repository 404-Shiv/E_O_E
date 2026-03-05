import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import Header from './components/Header'
import Ticker from './components/Ticker'
import LiveNews from './components/LiveNews'
import StockAnalysis from './components/StockAnalysis'
import StockComparison from './components/StockComparison'
import MutualFunds from './components/MutualFunds'
import NewsVerifier from './components/NewsVerifier'

const TABS = [
  { id: 'news', label: 'Live News', icon: '◉' },
  { id: 'stocks', label: 'Stocks', icon: '△' },
  { id: 'compare', label: 'Compare', icon: '⇆' },
  { id: 'mf', label: 'Mutual Funds', icon: '◈' },
  { id: 'verifier', label: 'News Verifier', icon: '⊘' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('news')
  const [theme, setTheme] = useState(() => {
    // Default to dark theme
    document.documentElement.setAttribute('data-theme', 'dark')
    return 'dark'
  })

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light'
    setTheme(next)
    document.documentElement.setAttribute('data-theme', next)
  }

  const renderTab = () => {
    switch (activeTab) {
      case 'news': return <LiveNews key="news" />
      case 'stocks': return <StockAnalysis key="stocks" />
      case 'compare': return <StockComparison key="compare" />
      case 'mf': return <MutualFunds key="mf" />
      case 'verifier': return <NewsVerifier key="verifier" />
      default: return <LiveNews key="news" />
    }
  }

  return (
    <>
      <Header theme={theme} toggleTheme={toggleTheme} />
      <Ticker />
      <div className="app-container">
        <nav className="tab-nav">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <AnimatePresence mode="wait">
          {renderTab()}
        </AnimatePresence>
      </div>
    </>
  )
}
