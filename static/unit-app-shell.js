/**
 * UNIT app shell — dark transitions, no white flash between pages.
 * Uses in-app fetch navigation (Turbo-style) so the WebView never unloads.
 */
(function () {
  'use strict';

  var isNative = !!(window.UNITNative && window.UNITNative.isNative);
  var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  var isStandalone = window.matchMedia && window.matchMedia('(display-mode: standalone)').matches;
  if (!isNative && !isMobile && !isStandalone) {
    document.documentElement.classList.add('unit-page-ready');
    return;
  }

  document.documentElement.classList.add('unit-app-feel');
  if (isNative || isStandalone) document.documentElement.classList.add('unit-native-feel');

  var overlay = document.getElementById('unit-nav-overlay');
  var overlayTimer = null;
  var navBusy = false;

  var GLOBAL_SCRIPT_RE = /unit-native|unit-app-shell|bootstrap|unit-autocomplete/i;
  var NO_TURBO_RE = /\/driver\/scan(\?|$)/;

  function showNavOverlay() {
    if (!overlay) return;
    clearTimeout(overlayTimer);
    overlay.classList.add('active');
    overlayTimer = setTimeout(function () {
      overlay.classList.add('slow');
    }, 350);
  }

  function hideNavOverlay() {
    if (!overlay) return;
    clearTimeout(overlayTimer);
    overlay.classList.remove('active', 'slow');
  }

  function markPageReady() {
    requestAnimationFrame(function () {
      document.documentElement.classList.add('unit-page-ready');
      hideNavOverlay();
    });
  }

  function hapticLight() {
    if (window.UNITNative && UNITNative.hapticSuccess) {
      UNITNative.hapticSuccess();
      return;
    }
    if (typeof navigator.vibrate === 'function') navigator.vibrate(8);
  }

  function isInternalNavLink(a) {
    if (!a || a.target === '_blank' || a.hasAttribute('download')) return null;
    if (a.dataset.noShell === 'true' || a.dataset.noTurbo === 'true') return null;
    var href = a.getAttribute('href');
    if (!href || href.charAt(0) === '#') return null;
    try {
      var url = new URL(a.href, location.href);
      if (url.origin !== location.origin) return null;
      if (/^\/static\//.test(url.pathname)) return null;
      return url;
    } catch (e) {
      return null;
    }
  }

  function shouldTurbo(url) {
    if (NO_TURBO_RE.test(url.pathname)) return false;
    return true;
  }

  function syncHeadStyles(doc) {
    doc.querySelectorAll('head link[rel="stylesheet"], head style').forEach(function (node) {
      if (node.tagName === 'LINK') {
        var href = node.getAttribute('href');
        if (!href) return;
        if (document.querySelector('link[rel="stylesheet"][href="' + href + '"]')) return;
        document.head.appendChild(node.cloneNode(true));
        return;
      }
      if (node.id === 'unit-page-style') return;
      var s = document.createElement('style');
      s.id = 'unit-page-style';
      s.textContent = node.textContent;
      var old = document.getElementById('unit-page-style');
      if (old) old.remove();
      document.head.appendChild(s);
    });
  }

  function runPageScripts(doc) {
    document.querySelectorAll('script[data-page-script]').forEach(function (s) {
      s.remove();
    });

    doc.querySelectorAll('body script').forEach(function (oldScript) {
      var src = oldScript.getAttribute('src') || '';
      if (src && GLOBAL_SCRIPT_RE.test(src)) return;
      var text = oldScript.textContent || '';
      if (text.indexOf('serviceWorker') !== -1) return;

      var s = document.createElement('script');
      s.setAttribute('data-page-script', '1');
      if (src) {
        s.src = src;
        s.async = false;
      } else {
        s.textContent = text;
      }
      document.body.appendChild(s);
    });
  }

  function swapPage(html, url, push) {
    var doc = new DOMParser().parseFromString(html, 'text/html');
    var newRoot = doc.getElementById('unit-app-root');
    if (!newRoot) throw new Error('no root');

    var title = doc.querySelector('title');
    if (title) document.title = title.textContent;

    syncHeadStyles(doc);
    document.getElementById('unit-app-root').innerHTML = newRoot.innerHTML;
    runPageScripts(doc);

    if (push !== false) {
      history.pushState({ unitTurbo: true }, '', url.pathname + url.search + url.hash);
    }

    window.scrollTo(0, 0);
    markPageReady();
    document.dispatchEvent(new CustomEvent('unit:page-load'));
    if (window.UNITNative && UNITNative.syncNativeTabs) UNITNative.syncNativeTabs();
  }

  function turboNavigate(url, push) {
    if (navBusy) return Promise.resolve(false);
    navBusy = true;
    showNavOverlay();

    return fetch(url.href, {
      credentials: 'same-origin',
      headers: { Accept: 'text/html' },
      cache: 'no-cache'
    }).then(function (resp) {
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.text();
    }).then(function (html) {
      swapPage(html, url, push);
      return true;
    }).catch(function () {
      hideNavOverlay();
      location.href = url.href;
      return false;
    }).finally(function () {
      navBusy = false;
    });
  }

  function hardNavigate(url) {
    try { sessionStorage.setItem('unit-nav', '1'); } catch (e) {}
    showNavOverlay();
    location.href = url.href;
  }

  // Page enter — boot cover hides once DOM is painted
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', markPageReady);
  } else {
    markPageReady();
  }

  // In-app navigation — fetch + swap (no full WebView reload = no white flash)
  document.addEventListener('click', function (e) {
    var a = e.target.closest('a[href]');
    var url = isInternalNavLink(a);
    if (!url) return;

    hapticLight();

    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

    if (shouldTurbo(url)) {
      e.preventDefault();
      turboNavigate(url, true);
      return;
    }

    e.preventDefault();
    hardNavigate(url);
  }, true);

  window.addEventListener('popstate', function () {
    var url = new URL(location.href);
    if (NO_TURBO_RE.test(url.pathname)) {
      location.reload();
      return;
    }
    turboNavigate(url, false);
  });

  // Smooth POST — overlay stays up until redirect page is swapped or loaded
  document.addEventListener('submit', function (e) {
    var form = e.target.closest('form');
    if (!form || form.method.toLowerCase() !== 'post') return;
    if (form.dataset.fullReload === 'true' || form.dataset.noShell === 'true') return;
    if (form.querySelector('input[type="file"]')) return;
    var action;
    try { action = new URL(form.action || location.href, location.href); } catch (err) { return; }
    if (action.origin !== location.origin) return;

    e.preventDefault();
    showNavOverlay();
    hapticLight();

    fetch(form.action || location.href, {
      method: 'POST',
      body: new FormData(form),
      credentials: 'same-origin',
      redirect: 'follow'
    }).then(function (resp) {
      var next = new URL(resp.url, location.href);
      if (shouldTurbo(next) && resp.ok) {
        return resp.text().then(function (html) {
          swapPage(html, next, true);
        });
      }
      location.href = resp.url;
    }).catch(function () {
      hideNavOverlay();
      form.submit();
    });
  }, true);

  // Press feedback
  document.addEventListener('touchstart', function (e) {
    var el = e.target.closest('a, button, .btn, .role-card, .nav-bottom a, .zone-chip, .list-tab, [data-press]');
    if (!el || el.disabled) return;
    el.classList.add('unit-pressed');
  }, { passive: true });

  document.addEventListener('touchend', function () {
    document.querySelectorAll('.unit-pressed').forEach(function (el) {
      el.classList.remove('unit-pressed');
    });
  }, { passive: true });

  document.addEventListener('touchcancel', function () {
    document.querySelectorAll('.unit-pressed').forEach(function (el) {
      el.classList.remove('unit-pressed');
    });
  }, { passive: true });

  // Block pull-to-refresh at top
  var touchStartY = 0;
  document.addEventListener('touchstart', function (e) {
    touchStartY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener('touchmove', function (e) {
    if (window.scrollY > 0) return;
    if (e.touches[0].clientY - touchStartY > 12) e.preventDefault();
  }, { passive: false });

  window.addEventListener('pageshow', function (e) {
    if (e.persisted) markPageReady();
    try { sessionStorage.removeItem('unit-nav'); } catch (err) {}
  });

  window.UNITAppShell = {
    showNavOverlay: showNavOverlay,
    hideNavOverlay: hideNavOverlay,
    turboNavigate: turboNavigate
  };

  if (!window.showToast) {
    var toastEl = null;
    var toastTimer = null;
    window.showToast = function (msg, isErr) {
      if (!toastEl) {
        toastEl = document.createElement('div');
        toastEl.id = 'unit-toast';
        document.body.appendChild(toastEl);
      }
      clearTimeout(toastTimer);
      toastEl.textContent = msg;
      toastEl.className = 'unit-toast show' + (isErr ? ' err' : '');
      toastTimer = setTimeout(function () { toastEl.classList.remove('show'); }, 2400);
    };
  }
})();
