#!/usr/bin/env bash
# Quick production smoke test — no auth required for most checks.
set -euo pipefail

BASE="${UNIT_API_BASE:-https://unit-6gxn.onrender.com}"
PASS=0
FAIL=0

check() {
  local name="$1"
  local expect="$2"
  shift 2
  if "$@"; then
    echo "✅ $name"
    PASS=$((PASS + 1))
  else
    echo "❌ $name"
    FAIL=$((FAIL + 1))
  fi
}

http_code() {
  curl -sS -o /dev/null -w "%{http_code}" "$1"
}

body_contains() {
  local url="$1"
  local needle="$2"
  curl -sS "$url" | grep -q "$needle"
}

echo "=== UNIT production smoke test ==="
echo "Base: $BASE"
echo

check "GET /health → 200" "200" test "$(http_code "$BASE/health")" = "200"

check "GET /api/address-suggest returns mapbox lat" "lat" \
  body_contains "$BASE/api/address-suggest?q=554+Holbrook+Detroit" '"lat"'

check "GET /api/geocode-verify exists (not 404)" "not404" \
  test "$(http_code "$BASE/api/geocode-verify?q=554+Holbrook+Street+Detroit+MI+48202")" != "404"

check "GET /api/mobile/v1/geocode-verify exists (401 or 200)" "auth" \
  sh -c 'c=$(curl -sS -o /dev/null -w "%{http_code}" "'"$BASE"'/api/mobile/v1/geocode-verify?q=test"); test "$c" = "401" -o "$c" = "200"'

check "POST /api/mobile/v1/routes/manual exists (401 not 404)" "manual" \
  sh -c 'c=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "'"$BASE"'/api/mobile/v1/routes/manual" -H "Content-Type: application/json" -d "{}"); test "$c" != "404"'

check "POST /api/mobile/v1/scan/build-route exists (401 not 404)" "build" \
  sh -c 'c=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "'"$BASE"'/api/mobile/v1/scan/build-route" -H "Content-Type: application/json" -d "{}"); test "$c" != "404"'

check "POST /api/mobile/v1/login registered (not 404)" "mobile" \
  sh -c 'c=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "'"$BASE"'/api/mobile/v1/login" -H "Content-Type: application/json" -d "{}"); test "$c" != "404"'

echo
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
