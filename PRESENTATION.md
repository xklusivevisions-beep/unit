# UNIT — Tomorrow Presentation Checklist

**Live URL:** https://unit-6gxn.onrender.com  
**Manager PIN:** `5555` (set on Render as `MANAGER_PIN`)  
**Native app:** TestFlight build 1.0 (1) — or run from Xcode on device

---

## Demo flow (15 min)

### 1. Manager portal (Rolling Logistics)
1. Open site → **Manager** → PIN **`5555`**
2. **Team** — check-in dots, route assignment, finish-by-9 ETA, rescue
3. **Roster** — add driver, set $/stop, SMS PIN
4. **Payroll** — manual entry (Sat→Fri week), export CSV
5. **Messages** — broadcast SMS to team

### 2. Driver flow
1. **Driver** login with driver PIN (from roster)
2. Check in **I'm In** on dashboard
3. **Scan Packages** — import route screenshots first, then live barcode scan
4. Start route → **NAV** → Mapbox turn-by-turn + geofence POD

### 3. Native app (TestFlight / Xcode)
- Same login flows inside the app
- Owner Admin hidden in app (web only)
- Live scan uses phone camera + BarcodeDetector
- CarPlay shows stop when driver taps NAV (full turn-by-turn needs Apple CarPlay entitlement)

---

## What's working (verified)

| Item | Status |
|------|--------|
| Render health | ✅ DB, Mapbox, Gemini vision |
| Manager login PIN 5555 | ✅ |
| All portals load | ✅ |
| unit-native.js on CDN | ✅ |
| Manager self-heal login | ✅ |
| Payroll (manager-scoped) | ✅ |
| GPS-stamped POD | ✅ |
| Finish-by-9 ETA | ✅ |

---

## Known limits (be honest if asked)

| Topic | Reality |
|-------|---------|
| TestFlight emails | Often don't arrive — use Internal Testing or Xcode Run |
| CarPlay full nav | Needs Apple entitlement + Mapbox Navigation SDK |
| Live scan in browser | BarcodeDetector (Safari/iOS app); fallback = Capture button |
| App Store listing | 1–3 days review; TestFlight works first |
| Drivers not on UNIT yet | Manager uses manual payroll entry this week |

---

## If something breaks during demo

| Problem | Fix |
|---------|-----|
| Wrong manager PIN | Render env `MANAGER_PIN=5555`, redeploy |
| Empty manager team | Roster → add drivers under Rolling Logistics |
| Scan says "import route first" | Upload Speed X screenshots before barcode scan |
| DB timeout | Check Render Postgres `unit-db` is running |
| Native app old UI | Pull to refresh — app loads live Render URL |

---

## Xcode rebuild (if needed tonight)

```bash
cd "/Users/directorxklusive/Desktop/UNIT App/native"
npm install
npx cap sync ios
open ios/App/unit.xcodeproj
```

**Important:** Open `unit.xcodeproj` NOT `App.xcodeproj`

Archive → increment build → Upload to TestFlight.

---

## PINs to have ready

- **Manager:** 5555
- **Admin:** your Render `ADMIN_PIN` (don't demo unless needed)
- **Driver PINs:** set in Manager → Roster → each driver
