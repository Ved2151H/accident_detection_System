import React, { useState, useEffect } from 'react';
import { Shield, Eye, Settings, AlertTriangle, List, CheckSquare, ChevronLeft, ChevronRight } from 'lucide-react';
import CollisionDetection from './pages/CollisionDetection/CollisionDetection';
import HelmetDetection from './pages/HelmetDetection/HelmetDetection';

function App() {
  const [theme, setTheme] = useState('cyberpunk');
  const [currentPage, setCurrentPage] = useState('collision');
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <div className={`app-container ${collapsed ? 'sidebar-collapsed' : ''}`}>
      {/* Sidebar Navigation */}
      <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header" style={{ display: 'flex', justifyContent: collapsed ? 'center' : 'space-between', alignItems: 'center', width: '100%' }}>
          {!collapsed ? (
            <div>
              <div className="sidebar-title">⚡ SMART TRAFFIC</div>
              <div className="sidebar-subtitle">SAFETY CONTROL PANEL</div>
            </div>
          ) : (
            <div className="sidebar-title-collapsed">⚡</div>
          )}
          <button className="sidebar-toggle-btn" onClick={() => setCollapsed(!collapsed)} title={collapsed ? "Expand Panel" : "Collapse Panel"}>
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        <div className="sidebar-divider" />

        {!collapsed && <div className="sidebar-heading">Navigation Menu</div>}
        <nav className="nav-menu">
          <button
            className={`nav-item ${currentPage === 'collision' ? 'active' : ''}`}
            onClick={() => setCurrentPage('collision')}
            title="Collision Detection"
          >
            <AlertTriangle size={18} style={{ flexShrink: 0 }} />
            {!collapsed && <span>Collision Detection</span>}
          </button>
          <button
            className={`nav-item ${currentPage === 'helmet' ? 'active' : ''}`}
            onClick={() => setCurrentPage('helmet')}
            title="Helmet Detection"
          >
            <Shield size={18} style={{ flexShrink: 0 }} />
            {!collapsed && <span>Helmet Detection</span>}
          </button>
        </nav>

        <div className="sidebar-divider" />

        {!collapsed && <div className="sidebar-heading">Select System Theme</div>}
        <div className="theme-selector">
          {collapsed ? (
            <button
              className="theme-cycle-btn"
              onClick={() => {
                const themes = ['cyberpunk', 'light', 'slate'];
                const nextIndex = (themes.indexOf(theme) + 1) % themes.length;
                setTheme(themes[nextIndex]);
              }}
              title={`Active Theme: ${theme.toUpperCase()} (Click to toggle)`}
            >
              <Settings size={18} className="theme-spin-icon" />
            </button>
          ) : (
            <>
              <button
                className={`theme-btn ${theme === 'cyberpunk' ? 'active' : ''}`}
                onClick={() => setTheme('cyberpunk')}
              >
                Dark Mode
              </button>
              <button
                className={`theme-btn ${theme === 'light' ? 'active' : ''}`}
                onClick={() => setTheme('light')}
              >
                Light Mode
              </button>
              <button
                className={`theme-btn ${theme === 'slate' ? 'active' : ''}`}
                onClick={() => setTheme('slate')}
              >
                Slate
              </button>
            </>
          )}
        </div>
      </aside>

      {/* Main Panel Content */}
      <main className="main-content">
        <header>
          <h2 className="header-title">
            {currentPage === 'collision' ? '🛡️ Accident Detection & Geolocalization' : '🏍️ Helmet Compliance Monitor'}
          </h2>
        </header>

        {currentPage === 'collision' ? (
          <CollisionDetection theme={theme} />
        ) : (
          <HelmetDetection theme={theme} />
        )}
      </main>
    </div>
  );
}

export default App;
