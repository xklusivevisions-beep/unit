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

  var BarcodeScanner = plugin('BarcodeScanner');
  var Haptics = plugin('Haptics');
  var UnitNav = plugin('UnitNav');

  var _liveListener = null;
  var _liveCallback = null;

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

    /** Push turn-by-turn destination to native CarPlay (Mapbox nav layer). */
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
    },

    /** Spoke-style continuous barcode scan (native ML Kit camera overlay). */
    startLiveScan: function (onBarcode) {
      if (!BarcodeScanner || typeof onBarcode !== 'function') {
        return Promise.resolve({ ok: false, reason: 'no_scanner' });
      }
      _liveCallback = onBarcode;
      return BarcodeScanner.isSupported()
        .then(function (r) {
          if (!r || !r.supported) return { ok: false, reason: 'unsupported' };
          return BarcodeScanner.requestPermissions();
        })
        .then(function (perm) {
          if (perm && perm.camera === 'denied') return { ok: false, reason: 'denied' };
          return BarcodeScanner.removeAllListeners().catch(function () {});
        })
        .then(function () {
          return BarcodeScanner.addListener('barcodesScanned', function (ev) {
            var codes = (ev && ev.barcodes) || [];
            for (var i = 0; i < codes.length; i++) {
              var raw = codes[i].rawValue || codes[i].displayValue || '';
              if (raw && _liveCallback) _liveCallback(raw);
            }
          });
        })
        .then(function (handle) {
          _liveListener = handle;
          return BarcodeScanner.startScan({
            formats: ['CODE_128', 'CODE_39', 'ITF', 'QR_CODE', 'DATA_MATRIX', 'PDF_417'],
            lensFacing: 'BACK'
          });
        })
        .then(function () { return { ok: true }; })
        .catch(function (e) {
          return { ok: false, reason: String(e && e.message || e) };
        });
    },

    stopLiveScan: function () {
      _liveCallback = null;
      var p = BarcodeScanner
        ? BarcodeScanner.stopScan().catch(function () {})
        : Promise.resolve();
      return p.then(function () {
        if (_liveListener && _liveListener.remove) return _liveListener.remove();
        if (BarcodeScanner) return BarcodeScanner.removeAllListeners().catch(function () {});
      }).catch(function () {});
    }
  };

  document.documentElement.classList.add('unit-native-app');
})();
