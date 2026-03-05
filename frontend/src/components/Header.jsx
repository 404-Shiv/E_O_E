export default function Header({ theme, toggleTheme }) {
    return (
        <header className="header">
            <div className="header-left">
                <div className="logo">
                    <span className="logo-icon">◆</span>
                    <span>EOE<span className="logo-accent"> Intelligence</span></span>
                </div>
                <span className="header-subtitle">Financial Analysis Platform</span>
            </div>
            <button className="theme-toggle" onClick={toggleTheme}>
                {theme === 'light' ? '🌙' : '☀️'}
                <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>
                    {theme === 'light' ? 'Dark' : 'Light'}
                </span>
            </button>
        </header>
    )
}
