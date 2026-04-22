// PhishGuard Extension — background.js (Service Worker)
// Monitors tabs, sends to analysis server, updates badge + history

'use strict';

const SERVER_URL       = 'http://127.0.0.1:5000/analyze';
const HEALTH_URL       = 'http://127.0.0.1:5000/health';
const HISTORY_KEY      = 'phishguard_history';
const MAX_HISTORY      = 50;

// ── STARTUP ───────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  console.log('[PhishGuard] Extension installed.');
  checkServer();
});

chrome.runtime.onStartup.addListener(checkServer);

function checkServer() {
  fetch(HEALTH_URL, { signal: AbortSignal.timeout(3000) })
    .then(r => r.json())
    .then(d => console.log(`[PhishGuard] Server online — ${d.system} v${d.version}`))
    .catch(() => console.warn('[PhishGuard] Server offline — start server.py'));
}

// ── TAB ANALYSIS ─────────────────────────────────────────
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete') return;
  if (!tab.url) return;
  if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://')) return;

  // Give DOM time to render
  await sleep(900);

  // Extract visible page text via scripting API
  let pageText = '';
  try {
    const res = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        try {
          let t = document.body.innerText || document.body.textContent || '';
          return t.replace(/\s+/g, ' ').trim().substring(0, 3500);
        } catch { return ''; }
      },
    });
    if (res?.[0]?.result) pageText = res[0].result;
  } catch (e) {
    console.warn(`[PhishGuard] Script injection failed for tab ${tabId}:`, e.message);
  }

  // Send to PhishGuard server
  try {
    const res = await fetch(SERVER_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url: tab.url, text: pageText }),
      signal:  AbortSignal.timeout(12000),
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
