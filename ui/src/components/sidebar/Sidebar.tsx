import React, { useRef, useState } from 'react';
import { css } from '@emotion/react';
import { Radio, Play, FileJson, Bug } from 'lucide-react';
import { useTraceContext } from '../../context/TraceContext';
import type { ViewType } from '../../lib/types';
import { BugReportModal } from '../bug-report/BugReportModal';
import { loadBugReport } from '../../api/client';

type SidebarSection = 'streaming' | 'offline' | 'builder';

function getActiveSection(currentView: ViewType): SidebarSection | null {
  switch (currentView) {
    case 'streaming': return 'streaming';
    case 'upload':
    case 'dashboard':
    case 'inspector':
    case 'comparison':
      return 'offline';
    case 'builder': return 'builder';
    default: return null;
  }
}

export const Sidebar: React.FC = () => {
  const { state, actions } = useTraceContext();
  const activeSection = getActiveSection(state.currentView);
  const [showBugReport, setShowBugReport] = useState(false);
  const [loadStatus, setLoadStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const clickTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleLoadBugReport = async (file: File) => {
    setLoadStatus('Loading...');
    try {
      const result = await loadBugReport(file);
      setLoadStatus(`Loaded ${result.count} session${result.count !== 1 ? 's' : ''}`);
      actions.setCurrentView('streaming');
      setTimeout(() => setLoadStatus(null), 3000);
    } catch (err) {
      setLoadStatus('Load failed');
      console.error('Bug report load failed:', err);
      setTimeout(() => setLoadStatus(null), 3000);
    }
  };

  return (
    <>
      <nav css={sidebarStyle}>
        <div css={brandStyle} onClick={() => actions.setCurrentView('welcome')}>
          agentevals
        </div>

        <div css={navListStyle}>
          <button
            css={[navItemStyle, activeSection === 'streaming' && activeItemStyle]}
            onClick={() => actions.setCurrentView('streaming')}
          >
            <Radio size={18} />
            Local Development
          </button>

          <button
            css={[navItemStyle, activeSection === 'offline' && activeItemStyle]}
            onClick={() => actions.setCurrentView('upload')}
          >
            <Play size={18} />
            Evaluations
          </button>

          <button
            css={[navItemStyle, activeSection === 'builder' && activeItemStyle]}
            onClick={() => actions.setCurrentView('builder')}
          >
            <FileJson size={18} />
            EvalSet Builder
          </button>
        </div>

        <div css={footerStyle}>
          <button
            onClick={() => {
              if (clickTimer.current) return;
              clickTimer.current = setTimeout(() => {
                clickTimer.current = null;
                setShowBugReport(true);
              }, 250);
            }}
            onDoubleClick={() => {
              if (clickTimer.current) {
                clearTimeout(clickTimer.current);
                clickTimer.current = null;
              }
              fileInputRef.current?.click();
            }}
            css={bugReportButtonStyle}
            title="Generate bug report"
          >
            <Bug size={14} />
            {loadStatus ?? 'Bug Report'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleLoadBugReport(file);
              e.target.value = '';
            }}
          />
          {state.version && (
            <span>v{state.version}</span>
          )}
        </div>
      </nav>

      {showBugReport && (
        <BugReportModal onClose={() => setShowBugReport(false)} />
      )}
    </>
  );
};

const sidebarStyle = css`
  width: 220px;
  flex-shrink: 0;
  height: 100vh;
  background: var(--bg-surface);
  border-right: 1px solid var(--border-default);
  display: flex;
  flex-direction: column;
  padding: 20px 0 0;
  overflow: hidden;
`;

const brandStyle = css`
  padding: 0 20px 20px;
  border-bottom: 1px solid var(--border-default);
  margin-bottom: 12px;
  font-size: 1.125rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  cursor: pointer;
  letter-spacing: -0.01em;
`;

const navListStyle = css`
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 0 8px;
`;

const navItemStyle = css`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 6px;
  border: none;
  border-left: 3px solid transparent;
  background: transparent;
  color: var(--text-secondary);
  font-size: 0.875rem;
  font-weight: 500;
  font-family: var(--font-display);
  cursor: pointer;
  width: 100%;
  text-align: left;
  transition: all 0.15s ease;

  &:hover {
    color: var(--text-primary);
    background: var(--bg-elevated);
  }
`;

const footerStyle = css`
  margin-top: auto;
  padding: 12px 16px;
  border-top: 1px solid var(--border-default);
  font-size: 0.75rem;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const bugReportButtonStyle = css`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid var(--border-default);
  background: transparent;
  color: var(--text-tertiary);
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  font-family: var(--font-display);
  width: 100%;

  &:hover {
    color: var(--text-secondary);
    background: var(--bg-elevated);
    border-color: var(--accent-cyan);
  }
`;

const activeItemStyle = css`
  color: var(--accent-cyan);
  background: rgba(0, 217, 255, 0.06);
  border-left-color: var(--accent-cyan);
  font-weight: 600;

  &:hover {
    color: var(--accent-cyan);
    background: rgba(0, 217, 255, 0.1);
  }
`;
