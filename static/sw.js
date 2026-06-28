// ── UNIT Service Worker ──────────────────────────────────────────────────────
// Cache strategy:
//   HTML pages  → network-first (4s timeout) → offline fallback
//   Static JS/CSS/img → cache-first with versioned key
//   API calls   → network-only (never cache mutations)
// Cache is VERSIONED — change CACHE_VER on every deploy to bust stale HTML.
// ─────────────────────────────────────────────────────────────────────────────

const CACHE_VER  = 'unit-v8';           // ← bump this on every deploy
const STATIC_VER = 'unit-static-v8';
const OFFLINE_PAGE = '/offline';

// Pages to pre-cache for offline use
const SHELL_URLS = [
  '/driver/login',
  '/offline',
  '/static/manifest.json',
  '/static/icon-192.png',
];

// Routes that should NEVER be cached (mutations, auth, live data)
const NO_CACHE_RE = /\/(api|driver\/scan\/(add|remove|build|clear|process|import|optimize|infer|validate|quick-nav|screenshots|item|live-sort|add-photo)|driver\/stop\/\d+\/(deliver|send-message|unit|pin|phone)|driver\/route\/\d+\/(optimize|reset|add-stop)|driver\/live|manager|admin)/i;

// Static assets — cache aggressively by version
const STATIC_RE = /\.(js|css|png|jpg|jpeg|webp|svg|ico|woff2?|ttf)(\?|$)/i;

// ── Install: pre-cache shell ──────────────────────────────────────────────────
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_VER)
      .then(c => c.addAll(SHELL_URLS).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

// ── Activate: delete ALL old caches ──────────────────────────────────────────
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys
          .filter(k => k !== CACHE_VER && k !== STATIC_VER)
          .map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

// ── Fetch ─────────────────────────────────────────────────────────────────────
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // 1) External CDN (Bootstrap, Mapbox, Leaflet, etc.) — pass through
  if (url.origin !== self.location.origin) {
    e.respondWith(fetch(req).catch(() => new Response('', { status: 503 })));
    return;
  }

  // 2) Mutations / live API — network only, no cache
  if (NO_CACHE_RE.test(url.pathname)) {
    e.respondWith(fetch(req));
    return;
  }

  // 3) Static assets — cache-first (versioned key busts on deploy)
  if (STATIC_RE.test(url.pathname)) {
    e.respondWith(
      caches.open(STATIC_VER).then(c =>
        c.match(req).then(cached => {
          if (cached) return cached;
          return fetch(req).then(resp => {
            if (resp.ok) c.put(req, resp.clone());
            return resp;
          });
        })
      )
    );
    return;
  }

  // 4) HTML pages — network-first with 4s timeout, then offline fallback
  e.respondWith(networkFirstHtml(req));
});

async function networkFirstHtml(req) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);

  try {
    const resp = await fetch(req, {
      signal: controller.signal,
      credentials: 'same-origin',
      // Force server — bypass HTTP cache, get fresh HTML every time
      cache: 'no-store',
    });
    clearTimeout(timer);

    if (resp.ok) {
      // Store fresh copy in cache for offline fallback
      const cache = await caches.open(CACHE_VER);
      cache.put(req, resp.clone());
    }
    return resp;
  } catch (err) {
    clearTimeout(timer);
    // Network failed or timed out — try cache
    const cached = await caches.match(req);
    if (cached) {
      // Inject offline banner into cached HTML
      const html = await cached.text();
      const patched = html.replace(
        '</body>',
        `<div id="unit-offline-bar" style="
          position:fixed;top:0;left:0;right:0;z-index:999999;
          background:#ff8c00;color:#000;font-weight:700;
          text-align:center;padding:8px 16px;font-size:0.82rem;
          display:flex;align-items:center;justify-content:center;gap:8px;">
          ⚡ Offline — showing saved data. Changes will sync when signal returns.
          <button onclick="location.reload()" style="background:#000;color:#ff8c00;border:none;border-radius:6px;padding:4px 10px;font-size:0.78rem;font-weight:700;cursor:pointer;">Retry</button>
        </div></body>`
      );
      return new Response(patched, {
        headers: { 'Content-Type': 'text/html; charset=utf-8' }
      });
    }
    // Nothing cached — serve offline page
    return caches.match(OFFLINE_PAGE) || new Response(offlineFallback(), {
      headers: { 'Content-Type': 'text/html; charset=utf-8' }
    });
  }
}

function offlineFallback() {
  return `<!DOCTYPE html>
<html lang="en" style="background:#0a0a0a;color:#fff;">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>UNIT — Offline</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { background: #0a0a0a; color: #fff; font-family: -apple-system, sans-serif;
         min-height: 100vh; display: flex; flex-direction: column;
         align-items: center; justify-content: center; padding: 32px 24px; }
  .logo { font-size: 2rem; font-weight: 900; letter-spacing: 0.3em; color: #00ff88; margin-bottom: 8px; }
  .logo span { color: #fff; }
  .icon { font-size: 3rem; margin: 24px 0 16px; }
  h1 { font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }
  p { font-size: 0.9rem; color: #aaa; text-align: center; max-width: 280px; line-height: 1.5; margin-bottom: 24px; }
  .retry-btn { background: #00ff88; color: #000; border: none; border-radius: 14px;
               padding: 14px 32px; font-size: 1rem; font-weight: 700; cursor: pointer; }
  .tip { font-size: 0.78rem; color: #555; margin-top: 24px; text-align: center; }
</style>
</head>
<body>
  <div class="logo">UNIT<span>.</span></div>
  <div class="icon">📡</div>
  <h1>No Signal</h1>
  <p>You're offline. Open the app when you have signal and your route will load automatically.</p>
  <button class="retry-btn" onclick="location.reload()">Try Again</button>
  <p class="tip">Tip: Load your route before heading out — it saves for offline use.</p>
</body>
</html>`;
}

// ── Message: force refresh (called from app after deploy detected) ─────────────
self.addEventListener('message', e => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
  if (e.data && e.data.type === 'CLEAR_CACHE') {
    caches.keys().then(keys => Promise.all(keys.map(k => caches.delete(k))));
  }
});
