/**
 * Native background location for drivers — posts to /api/location even when app is backgrounded.
 */
(function () {
  'use strict';

  if (!window.UNITNative || !window.UNITNative.isNative || !window.Capacitor) return;

  var Cap = window.Capacitor;
  function plugin(name) {
    if (Cap.Plugins && Cap.Plugins[name]) return Cap.Plugins[name];
    if (typeof Cap.registerPlugin === 'function') return Cap.registerPlugin(name);
    return null;
  }

  var UnitLocation = plugin('UnitLocation');
  var Geo = plugin('Geolocation');
  if (!UnitLocation && !Geo) return;

  var active = false;
  var watchId = null;
  var lastPost = 0;
  var stopId = null;
  var listenerHandles = [];

  function isDriverArea() {
    var p = location.pathname;
    return p.indexOf('/driver') === 0 && p.indexOf('/driver/login') !== 0;
  }

  function postLocation(lat, lng, accuracy) {
    var now = Date.now();
    if (now - lastPost < 6000) return;
    lastPost = now;
    var body = { lat: lat, lng: lng };
    if (stopId) body.stop_id = stopId;
    fetch('/api/location', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).catch(function () {});

    document.dispatchEvent(new CustomEvent('unit:location', {
      detail: { lat: lat, lng: lng, accuracy: accuracy || null }
    }));
  }

  function onCoords(lat, lng, accuracy) {
    if (!isFinite(lat) || !isFinite(lng)) return;
    postLocation(lat, lng, accuracy);
  }

  function attachNativeListeners() {
    if (!UnitLocation || !UnitLocation.addListener) return;
    listenerHandles.push(
      UnitLocation.addListener('locationUpdate', function (pos) {
        onCoords(pos.latitude, pos.longitude, pos.accuracy);
      })
    );
    listenerHandles.push(
      UnitLocation.addListener('authorizationChange', function (info) {
        if (info.status === 'always' && !active) start();
      })
    );
  }

  async function requestAlways() {
    if (UnitLocation && UnitLocation.requestAlways) {
      try { await UnitLocation.requestAlways(); } catch (e) {}
    }
    if (Geo && Geo.requestPermissions) {
      try { await Geo.requestPermissions({ permissions: ['location'] }); } catch (e) {}
    }
  }

  async function start() {
    if (active || !isDriverArea()) return;
    active = true;
    await requestAlways();

    if (UnitLocation && UnitLocation.startBackground) {
      try {
        var res = await UnitLocation.startBackground();
        if (res && res.ok) return;
      } catch (e) { /* fall through to Geolocation watch */ }
    }

    if (Geo && Geo.watchPosition && !watchId) {
      try {
        watchId = await Geo.watchPosition(
          { enableHighAccuracy: true, timeout: 20000, maximumAge: 3000 },
          function (pos, err) {
            if (err || !pos) return;
            onCoords(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy);
          }
        );
      } catch (e) { active = false; }
    }
  }

  function stop() {
    active = false;
    if (UnitLocation && UnitLocation.stopBackground) {
      UnitLocation.stopBackground().catch(function () {});
    }
    if (Geo && Geo.clearWatch && watchId) {
      Geo.clearWatch({ id: watchId }).catch(function () {});
      watchId = null;
    }
  }

  function setStop(id) {
    stopId = id || null;
  }

  function sync() {
    if (isDriverArea()) start();
    else stop();
  }

  attachNativeListeners();
  document.addEventListener('DOMContentLoaded', sync);
  document.addEventListener('unit:page-load', sync);

  window.UNITLocation = { start: start, stop: stop, setStop: setStop, requestAlways: requestAlways };
})();
