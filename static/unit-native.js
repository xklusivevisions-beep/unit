/**
 * UNIT native shell bridge (Capacitor iOS).
 * Loaded on Render — activates when running inside the TestFlight app.
 */
(function () {
  'use strict';

  var Cap = window.Capacitor;
  if (!Cap || typeof Cap.isNativePlatform !== 'function' || !Cap.isNativePlatform()) {
    window.UNITNative = { isNative: false };
    return;
  }

  function plugin(name) {
    if (Cap.Plugins && Cap.Plugins[name]) return Cap.Plugins[name];
    if (typeof Cap.registerPlugin === 'function') return Cap.registerPlugin(name);
    return null;
  }

  var Haptics = plugin('Haptics');
  var UnitNav = plugin('UnitNav');

  function playBeep() {
    try {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      var o = ctx.createOscillator();
      var g = ctx.createGain();
      o.connect(g);
      g.connect(ctx.destination);
      o.frequency.value = 920;
      g.gain.setValueAtTime(0.18, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.14);
      o.start(ctx.currentTime);
      o.stop(ctx.currentTime + 0.14);
    } catch (e) { /* silent */ }
  }

  window.UNITNative = {
    isNative: true,

    hapticSuccess: function () {
      if (!Haptics) return Promise.resolve();
      return Haptics.impact({ style: 'MEDIUM' }).catch(function () {});
    },

    hapticError: function () {
      if (!Haptics) return Promise.resolve();
      return Haptics.notification({ type: 'ERROR' }).catch(function () {});
    },

    playBeep: function () {
      playBeep();
      return Promise.resolve();
    },

    startCarPlayNavigation: function (lat, lng, address) {
      if (!UnitNav) return Promise.resolve({ ok: false, reason: 'no_plugin' });
      return UnitNav.startNavigation({
        lat: lat,
        lng: lng,
        address: address || ''
      }).catch(function () { return { ok: false }; });
    },

    stopCarPlayNavigation: function () {
      if (!UnitNav) return Promise.resolve();
      return UnitNav.stopNavigation().catch(function () {});
    }
  };

  document.documentElement.classList.add('unit-native-app');
})();
