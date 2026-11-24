// PhishGuard Extension — background.js (Service Worker)
// Monitors tabs, detects Single-Page App navigations, extracts rich metadata,
// queries local/Render server with fallback, updates badges, and serves popup requests.

'use strict';

const RENDER_URL = 'https://capstone-3b05.onrender.com';
const LOCAL_URL = 'http://127.0.0.1:5000';
const HISTORY_KEY = 'phishguard_history';
const MAX_HISTORY = 50;

// Track pending analyses to prevent duplicate rapid requests on SPAs
const pendingTabs = new Map();

// ── STARTUP ───────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  console.log('[PhishGuard] Extension installed.');
  checkServer();
});

chrome.runtime.onStartup.addListener(checkServer);

function checkServer() {
  fetch(`${LOCAL_URL}/health`, { signal: AbortSignal.timeout(1500) })
    .then(r => r.json())
    .then(d => console.log(`[PhishGuard] Local Server online — ${d.service || d.system} v${d.version}`))
    .catch(() => {
      fetch(`${RENDER_URL}/health`, { signal: AbortSignal.timeout(8000) })
        .then(r => r.json())
        .then(d => console.log(`[PhishGuard] Cloud Server online — ${d.service || d.system} v${d.version}`))
        .catch(() => console.warn('[PhishGuard] Cloud and Local servers offline'));
    });
}

// ── CORE TAB ANALYSIS ─────────────────────────────────────
async function analyzeTab(tabId, tab) {
  if (!tab || !tab.url) return { error: true, message: 'No URL provided' };
  if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://') && !tab.url.startsWith('file://')) {
    return { status: 'ignored' };
  }

  // Prevent duplicate concurrent requests for the exact same URL within 1.5 seconds
  const existing = pendingTabs.get(tabId);
  if (existing && existing.url === tab.url && (Date.now() - existing.time) < 1500) {
    const cached = await chrome.storage.local.get([tabId.toString()]);
    if (cached[tabId.toString()]) return cached[tabId.toString()];
  }
  pendingTabs.set(tabId, { url: tab.url, time: Date.now() });

  // Brief pause for client-side frameworks to render title/DOM
  await sleep(650);

  // Extract visible page text, titles, headings, and DOM features
  let pageTitle = tab.title || '';
  let pageText = '';
  let domFeatures = {};

  try {
    const tabCheck = await chrome.tabs.get(tabId).catch(() => null);
    if (!tabCheck) return { error: true, message: 'Tab closed' };

    const res = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        try {
          const docTitle = document.title || '';
          const metaTitle = document.querySelector('meta[name="title"]')?.content ||
                            document.querySelector('meta[property="og:title"]')?.content || '';
          const metaDesc = document.querySelector('meta[name="description"]')?.content ||
                           document.querySelector('meta[property="og:description"]')?.content || '';
          
          // YouTube & media specific selectors
          const ytTitle = document.querySelector('h1.ytd-watch-metadata yt-formatted-string')?.textContent ||
                          document.querySelector('#title h1')?.textContent ||
                          document.querySelector('h1.title')?.textContent || '';

          // Key page headings
          const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
            .map(h => h.textContent.trim())
            .filter(Boolean)
            .slice(0, 10)
            .join(' | ');

          // Body text sample
          const bodyText = (document.body.innerText || document.body.textContent || '')
            .replace(/\s+/g, ' ')
            .trim()
            .substring(0, 3000);

          const fullText = [docTitle, metaTitle, ytTitle, metaDesc, headings, bodyText]
            .filter(Boolean)
            .join(' ')
            .replace(/\s+/g, ' ')
            .trim()
            .substring(0, 4500);

          const features = {
            has_video: document.querySelectorAll('video').length > 0,
            has_audio: document.querySelectorAll('audio').length > 0,
            num_images: document.querySelectorAll('img').length,
            has_password: document.querySelectorAll('input[type="password"]').length > 0
          };

          return { title: ytTitle || docTitle || metaTitle, text: fullText, features: features };
        } catch {
          return { title: '', text: '', features: {} };
        }
      },
    });

    if (res?.[0]?.result) {
      if (res[0].result.title) pageTitle = res[0].result.title;
      pageText = res[0].result.text || '';
      domFeatures = res[0].result.features || {};
    }
  } catch (e) {
    console.warn(`[PhishGuard] Script injection notice for tab ${tabId}:`, e.message);
  }

  // Prepend tab.title if not already present
  if (tab.title && !pageText.toLowerCase().includes(tab.title.toLowerCase().substring(0, 20))) {
    pageText = `${tab.title}. ${pageText}`.trim();
  }

  // Send to PhishGuard server (Local Server Primary with Cloud Fallback)
  try {
    let res;
    const payload = JSON.stringify({
      url: tab.url,
      title: pageTitle,
      text: pageText,
      dom_features: domFeatures
    });

    try {
      res = await fetch(`${LOCAL_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok) throw new Error(`Local HTTP ${res.status}`);
    } catch (localErr) {
      console.log(`[PhishGuard] Local server unavailable, routing to Render cloud...`);
      res = await fetch(`${RENDER_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        signal: AbortSignal.timeout(35000), // Render free tier wake-up tolerance
      });
      if (!res.ok) throw new Error(`Cloud HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.status === 'ignored') return data;

    // Verify tab is still open
    const currentTab = await chrome.tabs.get(tabId).catch(() => null);
    if (!currentTab) return data;

    // Save per-tab result for popup (including active URL to verify sync)
    data.url = tab.url;
    await chrome.storage.local.set({ [tabId.toString()]: data });

    // Append to global history
    await appendHistory({
      url: (tab.url || '').substring(0, 100),
      score: data.final_risk_score,
      level: data.risk_level,
      timestamp: data.timestamp || new Date().toISOString(),
    });

    // Update badge
    const score = data.final_risk_score || 0;
    if (score >= 0.80) setBadge(tabId, '!!', '#EF4444');
    else if (score >= 0.60) setBadge(tabId, '!', '#F97316');
    else if (score >= 0.40) setBadge(tabId, '~', '#F59E0B');
    else setBadge(tabId, '✓', '#10B981');

    console.log(`[PhishGuard] ${tab.url.substring(0, 60)} → ${(score * 100).toFixed(0)}% ${data.risk_level}`);

    // INJECT WARNING OVERLAY ONLY FOR CONFIRMED DANGEROUS ATTACKS (Never on verified domains)
    const isSafeDomain = ['google.com', 'accounts.google.com', 'youtube.com', 'microsoft.com', 'apple.com', 'github.com'].some(d => (tab.url || '').toLowerCase().includes(d));
    const isSafeVerdict = data.risk_level === 'SAFE' || data.risk_level === 'LOW RISK' || score < 0.40;
    
    if (score >= 0.70 && !isSafeVerdict && !isSafeDomain) {
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
        args: [score, data.ai_explanation || '']
      }).catch(err => console.warn('[PhishGuard] Failed to inject warning overlay:', err));
    }

    return data;

  } catch (err) {
    console.warn('[PhishGuard] Analysis failed:', err.message);
    const errData = { error: true, url: tab.url, message: err.message };
    await chrome.storage.local.set({ [tabId.toString()]: errData });
    setBadge(tabId, '?', '#6B7280');
    return errData;
  }
}

// ── EVENT LISTENERS ───────────────────────────────────────

// 1. Tab update listener (fires on full reload AND client-side SPA URL changes)
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== 'complete' && !changeInfo.url) return;
  if (!tab || !tab.url) return;
  analyzeTab(tabId, tab);
});

// 2. Tab switched / activated (ensures popup is accurate when user switches tabs)
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    if (!tab || !tab.url) return;
    if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://') && !tab.url.startsWith('file://')) return;
    const stored = await chrome.storage.local.get([activeInfo.tabId.toString()]);
    const tabData = stored[activeInfo.tabId.toString()];
    if (!tabData || tabData.url !== tab.url) {
      analyzeTab(activeInfo.tabId, tab);
    }
  } catch {}
});

// 3. Runtime messaging (allows popup.js to request instant on-demand analysis)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeNow' && request.tabId) {
    chrome.tabs.get(request.tabId).then(tab => {
      if (tab) {
        analyzeTab(request.tabId, tab).then(res => sendResponse(res));
      } else {
        sendResponse({ error: true, message: 'Tab not accessible' });
      }
    }).catch(err => sendResponse({ error: true, message: err.message }));
    return true; // Keep message channel open for async response
  }
});

// ── HISTORY HELPERS ───────────────────────────────────────
async function appendHistory(entry) {
  return new Promise(resolve => {
    chrome.storage.local.get([HISTORY_KEY], (result) => {
      const items = result[HISTORY_KEY] || [];
      // Avoid duplicate consecutive same URL
      if (items.length > 0 && items[0].url === entry.url) {
        items[0] = entry; // update in place
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
  pendingTabs.delete(tabId);
});

// ── UTILS ─────────────────────────────────────────────────
function setBadge(tabId, text, color) {
  try {
    chrome.action.setBadgeText({ text, tabId }).catch(() => {});
    chrome.action.setBadgeBackgroundColor({ color, tabId }).catch(() => {});
  } catch {}
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
