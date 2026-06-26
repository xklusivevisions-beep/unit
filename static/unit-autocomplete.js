/**
 * UNIT address autocomplete — attach to any input with class "js-addr-auto".
 * Suggestions come from /api/address-suggest (past deliveries + Mapbox rooftop).
 * Optional data attributes:
 *   data-lat-field="#id"  data-lng-field="#id"  → filled with chosen coords
 *   data-unit-field="#id" data-name-field="#id" → filled from past-delivery hits
 */
(function () {
  'use strict';

  var DROP_ID = 'unit-ac-dropdown';
  var activeInput = null;

  function dropdown() {
    var el = document.getElementById(DROP_ID);
    if (!el) {
      el = document.createElement('div');
      el.id = DROP_ID;
      el.style.cssText =
        'position:fixed;z-index:99999;background:#0d0d0d;border:1px solid #333;' +
        'border-radius:12px;overflow:hidden;display:none;box-shadow:0 10px 30px rgba(0,0,0,0.6);' +
        'max-height:60vh;overflow-y:auto;';
      document.body.appendChild(el);
    }
    return el;
  }

  function hide() {
    var el = document.getElementById(DROP_ID);
    if (el) el.style.display = 'none';
    activeInput = null;
  }

  function position(input) {
    var el = dropdown();
    var r = input.getBoundingClientRect();
    el.style.left = r.left + 'px';
    el.style.top = (r.bottom + 4) + 'px';
    el.style.width = r.width + 'px';
  }

  function fill(input, item) {
    input.value = item.address || '';
    var setField = function (attr, val) {
      var sel = input.getAttribute(attr);
      if (sel && val != null && val !== '') {
        var f = document.querySelector(sel);
        if (f && !f.value) f.value = val;
      }
    };
    setField('data-lat-field', item.lat);
    setField('data-lng-field', item.lng);
    setField('data-unit-field', item.unit);
    setField('data-name-field', item.name);
    input.dispatchEvent(new Event('change', { bubbles: true }));
    input.dispatchEvent(new Event('input', { bubbles: true }));
    hide();
  }

  function render(input, items) {
    var el = dropdown();
    el.innerHTML = '';
    if (!items.length) { hide(); return; }
    activeInput = input;
    items.forEach(function (item) {
      var row = document.createElement('div');
      row.style.cssText = 'padding:12px 14px;cursor:pointer;border-bottom:1px solid #1e1e1e;min-height:46px;';
      var tag = item.source === 'history'
        ? '<span style="color:#10b981;font-size:0.68rem;font-weight:700;">★ Past delivery</span>'
        : '<span style="color:#60a5fa;font-size:0.68rem;font-weight:700;">📍 Address</span>';
      row.innerHTML =
        '<div style="color:#fff;font-size:0.9rem;font-weight:600;">' +
        (item.address || '').replace(/</g, '&lt;') + '</div>' +
        '<div style="margin-top:2px;">' + tag +
        (item.unit ? '<span style="color:#fdba74;font-size:0.68rem;"> · Unit ' + item.unit + '</span>' : '') +
        '</div>';
      row.addEventListener('mousedown', function (e) { e.preventDefault(); fill(input, item); });
      el.appendChild(row);
    });
    position(input);
    el.style.display = 'block';
  }

  function attach(input) {
    if (input.dataset.unitAc) return;
    input.dataset.unitAc = '1';
    input.setAttribute('autocomplete', 'off');
    var timer = null;
    input.addEventListener('input', function () {
      var q = input.value.trim();
      clearTimeout(timer);
      if (q.length < 3) { hide(); return; }
      timer = setTimeout(function () {
        fetch('/api/address-suggest?q=' + encodeURIComponent(q))
          .then(function (r) { return r.json(); })
          .then(function (items) { if (input.value.trim().length >= 3) render(input, items || []); })
          .catch(function () { hide(); });
      }, 220);
    });
    input.addEventListener('blur', function () { setTimeout(hide, 150); });
  }

  function scan(root) {
    (root || document).querySelectorAll('input.js-addr-auto').forEach(attach);
  }

  document.addEventListener('DOMContentLoaded', function () { scan(document); });
  document.addEventListener('click', function (e) {
    var el = document.getElementById(DROP_ID);
    if (el && activeInput && e.target !== activeInput && !el.contains(e.target)) hide();
  });
  window.addEventListener('scroll', function () { if (activeInput) position(activeInput); }, true);
  window.addEventListener('resize', function () { if (activeInput) position(activeInput); });

  // Expose for dynamically added inputs (e.g. route_manual "add stop")
  window.UNITAddrAuto = { scan: scan, attach: attach };
})();
