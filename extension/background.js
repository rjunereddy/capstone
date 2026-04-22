// PhishGuard Extension — background.js (Service Worker)
// Monitors tabs, sends to analysis server, updates badge + history

'use strict';

const SERVER_URL       = 'https://phishguard-ai-r5wa.onrender.com/analyze';
const HEALTH_URL       = 'https://phishguard-ai-r5wa.onrender.com/health';
const HISTORY_KEY      = 'phishguard_history';
const MAX_HISTORY      = 50;

// ── STARTUP ───────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  console.log('[PhishGuard] Extension installed.');
  checkServer();
});

chrome.runtime.onStartup.addListener(checkServer);

function checkServer() {
  fetch(HEALTH_URL, { signal: AbortSignal.timeout(60000) })
    .then(r => r.json())
    .then(d => console.log(`[PhishGuard] Server online — ${d.system} v${d.version}`))
    .catch(() => console.warn('[PhishGuard] Server offline — start server.py'));
}

// ── TAB ANALYSIS ─────────────────────────────────────────
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete') return;
  if (!tab.url) return;
  if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://') && !tab.url.startsWith('file://')) return;

  // Give DOM time to render
  await sleep(900);

  // Extract visible page text via scripting API
  let pageText = '';
  let domFeatures = {};
  try {
    const res = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        try {
          let t = document.body.innerText || document.body.textContent || '';
          let textContent = t.replace(/\s+/g, ' ').trim().substring(0, 3500);
          
          let features = {
            has_video: document.querySelectorAll('video').length > 0,
            has_audio: document.querySelectorAll('audio').length > 0,
            num_images: document.querySelectorAll('img').length,
            has_password: document.querySelectorAll('input[type="password"]').length > 0
          };
          return { text: textContent, features: features };
        } catch { return { text: '', features: {} }; }
      },
    });
    if (res?.[0]?.result) {
      pageText = res[0].result.text;
      domFeatures = res[0].result.features;
    }
  } catch (e) {
    console.warn(`[PhishGuard] Script injection failed for tab ${tabId}:`, e.message);
  }

  // Send to PhishGuard server
  try {
    const res = await fetch(SERVER_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url: tab.url, text: pageText, dom_features: domFeatures }),
      signal:  AbortSignal.timeout(60000),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === 'ignored') return;

    // Save per-tab result for popup
    await chrome.storage.local.set({ [tabId.toString()]: data });

    // Append to global history
    await appendHistory({
      url:       (tab.url || '').substring(0, 100),
      score:     data.final_risk_score,
      level:     data.risk_level,
      timestamp: data.timestamp || new Date().toISOString(),
    });

    // Update badge
    const score = data.final_risk_score || 0;
    if      (score >= 0.80) setBadge(tabId, '!!', '#EF4444');
    else if (score >= 0.60) setBadge(tabId, '!',  '#F97316');
    else if (score >= 0.40) setBadge(tabId, '~',  '#F59E0B');
    else                    setBadge(tabId, '✓',  '#10B981');

    console.log(`[PhishGuard] ${tab.url.substring(0,60)} → ${(score*100).toFixed(0)}% ${data.risk_level}`);

    // INJECT WARNING OVERLAY FOR MALICIOUS PAGES
    if (score >= 0.60) {
      chrome.scripting.executeScript({
        target: { tabId },
        func: (riskScore, explanation) => {
          if (document.getElementById('phishguard-warning-overlay')) return;
          const overlay = document.createElement('div');
          overlay.id = 'phishguard-warning-overlay';
          Object.assign(overlay.style, {
            position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
            backgroundColor: 'rgba(20, 0, 0, 0.90)', color: '#fff',
            zIndex: '2147483647', display: 'flex', flexDirection: 'column',
            justifyContent: 'center', alignItems: 'center', fontFamily: 'system-ui, sans-serif',
            backdropFilter: 'blur(10px)', textAlign: 'center', padding: '20px', boxSizing: 'border-box'
          });
          overlay.innerHTML = `
            <div style="background: #111; padding: 40px; border-radius: 16px; max-width: 600px; box-shadow: 0 0 50px rgba(239,68,68,0.3); border: 2px solid #ef4444;">
              <div style="font-size: 4rem; margin-bottom: 10px;">🛡️</div>
              <h1 style="font-size: 2rem; margin: 0 0 10px 0; color: #ef4444;">PhishGuard Alert</h1>
              <h2 style="font-size: 1.2rem; margin: 0 0 20px 0; color: #fca5a5;">High Risk Detected: ${(riskScore * 100).toFixed(0)}%</h2>
              <p style="font-size: 1rem; color: #ccc; margin-bottom: 30px; background: rgba(239,68,68,0.1); padding: 15px; border-radius: 8px; line-height: 1.5;">
                ${explanation || 'This page exhibits highly suspicious patterns and may be attempting to steal your information.'}
              </p>
              <button id="phishguard-dismiss-btn" style="background: #ef4444; color: white; border: none; padding: 12px 24px; font-size: 1rem; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; transition: background 0.2s;">
                I understand the risks, proceed anyway
              </button>
            </div>
          `;
          document.body.appendChild(overlay);
          document.getElementById('phishguard-dismiss-btn').addEventListener('click', () => {
            overlay.remove();
          });
        },
        args: [score, data.gemini_explanation]
      }).catch(err => console.warn('[PhishGuard] Failed to inject warning overlay:', err));
    }

  } catch (err) {
    console.warn('[PhishGuard] Analysis failed:', err.message);
    await chrome.storage.local.set({ [tabId.toString()]: { error: true, url: tab.url } });
    setBadge(tabId, '?', '#6B7280');
  }
});

// ── HISTORY HELPERS ───────────────────────────────────────
async function appendHistory(entry) {
  return new Promise(resolve => {
    chrome.storage.local.get([HISTORY_KEY], (result) => {
      const items = result[HISTORY_KEY] || [];
      // Avoid duplicate consecutive same URL
      if (items.length > 0 && items[0].url === entry.url) {
        items[0] = entry;   // update in place
      } else {
        items.unshift(entry);
      }
      const trimmed = items.slice(0, MAX_HISTORY);
      chrome.storage.local.set({ [HISTORY_KEY]: trimmed }, resolve);
    });
  });
}

// ── CLEANUP ───────────────────────────────────────────────
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.local.remove(tabId.toString());
});

// ── UTILS ─────────────────────────────────────────────────
function setBadge(tabId, text, color) {
  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
