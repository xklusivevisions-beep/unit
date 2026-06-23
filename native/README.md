# UNIT Native (iOS)

Capacitor 8 shell that wraps the live UNIT web app at **https://unit-6gxn.onrender.com**.

One app, all roles: **Driver**, **Manager**, **Resident**, and **Owner Admin** — same as the website landing page.

## What this is

- Not a rewrite of Flask — a native WebView container with real app icon, splash, camera, and GPS permissions
- Backend stays on Render; change `MANAGER_PIN`, env vars, etc. on Render and the app picks them up immediately
- Submit to **TestFlight** tonight; App Store review is usually 1–3 days after that

## Prerequisites

1. **Xcode** (full app, not just Command Line Tools)
   - Mac App Store → search **Xcode** → Install (~7 GB)
   - After install, run once and accept the license
   - Point tools at Xcode:
     ```bash
     sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
     ```
2. **Apple Developer Program** ($99/year) — you already have this
3. **Node.js** — already installed on this Mac

Verify:

```bash
xcodebuild -version
node -v
npm -v
```

## Quick start (after Xcode installs)

```bash
cd "/Users/directorxklusive/Desktop/UNIT App/native"
npm install
npm run sync          # regen icons + cap sync ios
npm run open:ios      # opens Xcode
```

Or one command:

```bash
npm run build:ios
```

## Xcode signing (first time)

1. Xcode opens **App** target → **Signing & Capabilities**
2. Check **Automatically manage signing**
3. **Team** → select your Apple Developer account
4. **Bundle Identifier** must be: `com.unitlogistics.unit`
5. Connect your iPhone (optional) or choose a simulator to smoke-test

## Run on your iPhone (immediate, no TestFlight)

1. Plug in iPhone → trust computer
2. Xcode top bar → select your device
3. Press **Run** (▶)
4. iPhone: **Settings → General → VPN & Device Management** → trust your developer cert

## TestFlight tonight

1. In Xcode menu: **Product → Archive**
2. When Organizer opens: **Distribute App**
3. **App Store Connect** → **Upload**
4. Go to [App Store Connect](https://appstoreconnect.apple.com)
5. **My Apps** → **+** → New App
   - Name: **UNIT**
   - Bundle ID: `com.unitlogistics.unit`
   - SKU: `unit-logistics-001` (any unique string)
6. **TestFlight** tab → add internal testers (your Apple ID email)
7. Build processes in ~5–30 minutes, then install **TestFlight** app on phones

### App Store Connect checklist

- **Privacy Policy URL** — required even for TestFlight external testers; use your site or a simple GitHub page
- **Export compliance** — typically “No” for HTTPS-only apps
- **Screenshots** — not required for internal TestFlight; required for public App Store listing

## Regenerate icon / splash

Dark background, white **UNIT** with blue dot:

```bash
npm run assets
```

Outputs:

- `ios/App/App/Assets.xcassets/AppIcon.appiconset/AppIcon-512@2x.png` (1024×1024)
- `ios/App/App/Assets.xcassets/Splash.imageset/splash-*.png`

## Project layout

```
native/
├── capacitor.config.json   # server.url → Render
├── www/index.html          # offline fallback while loading
├── scripts/generate_assets.py
├── package.json
└── ios/                    # Xcode project (commit this)
```

## Config notes

| Setting | Value |
|--------|--------|
| App ID | `com.unitlogistics.unit` |
| Display name | UNIT |
| Server URL | `https://unit-6gxn.onrender.com` |
| iOS permissions | Camera, Photos, Location (Info.plist) |

To point at a different server (staging), edit `capacitor.config.json` → `server.url`, then `npm run sync`.

## Troubleshooting

| Problem | Fix |
|--------|-----|
| `xcodebuild requires Xcode` | Install Xcode from App Store, run `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` |
| Signing failed | Xcode → Settings → Accounts → add Apple ID → pick Team on target |
| White screen in app | Check Render is up: `/health` → `"status":"ok"` |
| Camera/GPS denied | iPhone Settings → UNIT → allow Camera & Location |
| Manager PIN wrong | Set `MANAGER_PIN` on Render web service; redeploy |

## npm scripts

| Script | Action |
|--------|--------|
| `npm run assets` | Generate icon + splash PNGs |
| `npm run sync` | assets + `cap sync ios` |
| `npm run open:ios` | Open Xcode |
| `npm run build:ios` | sync + open Xcode |
| `npm run doctor` | Capacitor health check |
