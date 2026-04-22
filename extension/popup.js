// ╔══════════════════════════════════════════════════════════╗
// ║  PhishGuard Extension — popup.js                        ║
// ║  Live analysis + history tab with premium UI            ║
// ╚══════════════════════════════════════════════════════════╝

'use strict';

// ── CONSTANTS ──────────────────────────────────────────────
const STORAGE_HISTORY_KEY = 'phishguard_history';
const MAX_HISTORY         = 50;

const TIER = {
  safe:   { color: '#10B981', glow: 'rgba(16,185,129,0.25)', icon: '🟢', label: 'SAFE',     action: 'This website appears safe. No significant threats detected.' },
  low:    { color: '#06B6D4', glow: 'rgba(6,182,212,0.25)',  icon: '🔵', label: 'LOW RISK',  action: 'Minimal risk detected. Remain cautious and verify if unsure.' },
  medium: { color: '#F59E0B', glow: 'rgba(245,158,11,0.25)', icon: '🟡', label: 'MEDIUM',    action: 'Potential phishing indicators found. Verify this site carefully.' },
  high:   { color: '#F97316', glow: 'rgba(249,115,22,0.25)', icon: '🟠', label: 'HIGH RISK', action: 'Strong phishing signals detected. Do NOT enter credentials.' },
  crit:   { color: '#EF4444', glow: 'rgba(239,68,68,0.30)',  icon: '🔴', label: 'CRITICAL',  action: 'Coordinated phishing attack confirmed. DO NOT INTERACT.' },
};

function getTier(score) {
  if (score >= 0.80) return 'crit';
  if (score >= 0.60) return 'high';
  if (score >= 0.40) return 'medium';
  if (score >= 0.20) return 'low';
  return 'safe';
}

// ── TAB SWITCHING ─────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab${capitalize(name)}`).classList.add('active');
  document.getElementById(`panel${capitalize(name)}`).classList.add('active');
  if (name === 'history') renderHistory();
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ── MAIN INIT ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Bind UI Events
  document.getElementById('tabCurrent').addEventListener('click', () => switchTab('current'));
  document.getElementById('tabHistory').addEventListener('click', () => switchTab('history'));
  const btnClear = document.getElementById('clearBtn');
  if (btnClear) btnClear.addEventListener('click', clearHistory);

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0]) return;
    const tab = tabs[0];

    // Display URL
    const urlEl = document.getElementById('targetUrl');
    if (urlEl) urlEl.textContent = tab.url || '—';

    // Look up stored analysis for this tab
    chrome.storage.local.get([tab.id.toString()], (result) => {
      const data = result[tab.id.toString()];

      if (data && !data.error && data.final_risk_score !== undefined) {
        setStatus('online', 'Protected');
        renderAnalysis(data);
      } else if (data && data.error) {
        setStatus('offline', 'Offline');
        renderOffline();
      } else {
        setStatus('busy', 'Analyzing');
        renderWaiting();
      }
    });
  });
});

// ── STATUS INDICATOR ─────────────────────────────────────
function setStatus(state, label) {
  const dot = document.getElementById('statusDot');
  const lbl = document.getElementById('statusLabel');
  if (dot) { dot.className = 'status-dot ' + state; }
  if (lbl) { lbl.textContent = label; }
}

// ── OFFLINE STATE ─────────────────────────────────────────
function renderOffline() {
  setGaugeScore(null, null);
  setText('riskLabel',  'OFFLINE');
  setText('riskAction', 'Start server.py then refresh the page.');
  setText('geminiText', 'PhishGuard server is not running. Open a terminal and run: python server.py');
}

// ── WAITING STATE ─────────────────────────────────────────
function renderWaiting() {
  setText('riskLabel',  'ANALYZING');
  setText('riskAction', 'Analysis in progress... If it persists, reload the page.');
  setText('geminiText', 'Waiting for PhishGuard fusion engine...');
}

// ── MAIN RENDER ───────────────────────────────────────────
function renderAnalysis(data) {
  const score = data.final_risk_score || 0;
  const tier  = getTier(score);
  const t     = TIER[tier];

  // 1. Gauge
  setGaugeScore(score, t.color);

  // 2. Risk badge
  const badge = document.getElementById('riskBadge');
  if (badge) {
    badge.style.background   = `${t.color}18`;
    badge.style.borderColor  = `${t.color}45`;
  }
  setText('riskIcon',   t.icon);
  setStyledText('riskLabel', t.label, t.color);
  setText('riskAction', data.action_required || t.action);

  // 3. Modality bars
  const risks = data.individual_risks || {};
  renderBar('Url',   risks.url   ?? 0);
  renderBar('Text',  risks.text  ?? 0);
  renderBar('Image', risks.image ?? 0);
  renderBar('Audio', risks.audio ?? 0);
  renderBar('Video', risks.video ?? 0);

  // 4. Model info badges
  const infoRow = document.getElementById('modelInfoRow');
  if (infoRow) {
    infoRow.style.display = 'flex';
    const uMethod = data.url_method  === 'ml' ? '✅ ML' : 'Rule';
    const tMethod = data.text_method === 'ml' ? '✅ ML' : 'Rule';
    setText('urlMethodBadge',  `URL: ${uMethod}`);
    setText('textMethodBadge', `Text: ${tMethod}`);
  }

  // 5. Fusion agreement
  const fusionRow = document.getElementById('fusionRow');
  const fusionVal = document.getElementById('fusionValue');
  if (data.agreement_multiplier !== undefined && fusionRow && fusionVal) {
    fusionRow.style.display = 'flex';
    const mult = data.agreement_multiplier;
    fusionVal.textContent = mult.toFixed(3) + (mult >= 1 ? '  ↑ Agreement' : '  ↓ Conflict');
  }

  // 6. Red flags
  const allFlags = [
    ...(data.red_flags       || []),
    ...(data.url_indicators  || []).slice(0, 3),
    ...(data.text_indicators || []).slice(0, 2),
    ...(data.image_indicators|| []).slice(0, 2),
  ].filter(Boolean).slice(0, 7);

  const flagsCard = document.getElementById('flagsCard');
  const flagsList = document.getElementById('flagsList');
  if (allFlags.length > 0 && score >= 0.35 && flagsCard && flagsList) {
    flagsCard.style.display = 'block';
    flagsList.innerHTML = allFlags
      .map(f => `<li>${escapeHTML(f)}</li>`)
      .join('');
  }

  // 6b. SHAP Analysis
  const shapCard = document.getElementById('shapCard');
  const shapCont = document.getElementById('shapContainer');
  if (data.shap_values && data.shap_values.length > 0 && shapCard && shapCont) {
    shapCard.style.display = 'block';
    shapCont.innerHTML = data.shap_values.map(s => {
      const isPos = s.direction === '+';
      const w = Math.min(Math.abs(s.impact) * 200, 48); // scale for UI width (max 48%)
      return `
        <div class="shap-row" style="margin-top:10px;">
          <div class="shap-label ${isPos ? 'left' : 'right'}">${escapeHTML(s.feature)}</div>
          <div class="shap-bar ${isPos ? 'positive' : 'negative'}" style="width: ${w}%"></div>
        </div>
      `;
    }).join('');
  }

  // 7. Gemini explanation
  setText('geminiText', data.ai_explanation || 'No explanation available.');
}

// ── GAUGE ─────────────────────────────────────────────────
function setGaugeScore(score, color) {
  const fill  = document.getElementById('gaugeFill');
  const label = document.getElementById('gaugeScore');
  const ARC   = 251.3;

  if (score === null || score === undefined) {
    if (fill)  fill.style.strokeDashoffset = ARC;
    if (label) { label.textContent = '--'; label.style.fill = 'rgba(255,255,255,0.3)'; }
    return;
  }

  const offset = ARC - (ARC * score);
  if (fill) {
    fill.style.strokeDashoffset = offset;
    // Update gradient direction based on score
    // (uses the CSS gradient already defined via url(#gaugeGrad))
  }
  if (label) {
    label.textContent  = Math.round(score * 100) + '%';
    label.style.fill   = color || 'white';
    label.style.filter = `drop-shadow(0 0 6px ${color || 'white'}66)`;
  }
}

// ── MODALITY BAR ──────────────────────────────────────────
function renderBar(name, score) {
  const bar   = document.getElementById(`bar${name}`);
  const label = document.getElementById(`pct${name}`);
  if (!bar || !label) return;

  const tier  = getTier(score);
  const color = TIER[tier].color;
  const pct   = Math.round(score * 100);

  setTimeout(() => {
    bar.style.width      = pct + '%';
    bar.style.background = color;
    bar.style.boxShadow  = `0 0 6px ${color}80`;
  }, 280);

  label.textContent = pct + '%';
  label.style.color = color;
}

// ── HISTORY ───────────────────────────────────────────────
function renderHistory() {
  chrome.storage.local.get([STORAGE_HISTORY_KEY], (result) => {
    const items = result[STORAGE_HISTORY_KEY] || [];
    const list  = document.getElementById('historyList');
    if (!list) return;

    if (items.length === 0) {
      list.innerHTML = '<div class="history-empty">No analyses yet. Browse some websites!</div>';
      return;
    }

    list.innerHTML = items.slice(0, MAX_HISTORY).map(item => {
      const tier  = getTier(item.score || 0);
      const t     = TIER[tier];
      const pct   = Math.round((item.score || 0) * 100);
      const ts    = item.timestamp
        ? new Date(item.timestamp).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
        : '';

      return `
        <div class="history-item">
          <div class="hist-dot" style="background:${t.color}; box-shadow:0 0 5px ${t.color}80;"></div>
          <div class="hist-body">
            <div class="hist-url">${escapeHTML((item.url||'').substring(0,55))}</div>
            <div class="hist-time">${ts} · ${t.label}</div>
          </div>
          <div class="hist-score" style="color:${t.color}">${pct}%</div>
        </div>
      `;
    }).join('');
  });
}

function clearHistory() {
  chrome.storage.local.remove(STORAGE_HISTORY_KEY, () => {
    renderHistory();
  });
}

// ── HELPERS ───────────────────────────────────────────────
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setStyledText(id, value, color) {
  const el = document.getElementById(id);
  if (el) { el.textContent = value; el.style.color = color; }
}

function escapeHTML(str) {
  return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
