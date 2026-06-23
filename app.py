from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, get_flashed_messages
from datetime import datetime, timedelta
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import sqlite3, os, json, requests, logging, traceback, csv, io, secrets, re, base64
import pdfplumber, math
from PIL import Image
import anthropic

ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
# Free vision: get a key at https://aistudio.google.com/apikey
GEMINI_API_KEY  = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY', '')
# gemini-1.5-* and gemini-2.0-* shut down June 2026 — 2.5 only
GEMINI_MODELS   = [
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash',
]
_GEMINI_MODELS_CACHE = None


def _resolve_gemini_models():
    """Discover vision-capable Gemini models; fall back to GEMINI_MODELS."""
    global _GEMINI_MODELS_CACHE
    if _GEMINI_MODELS_CACHE is not None:
        return _GEMINI_MODELS_CACHE
    preferred = list(GEMINI_MODELS)
    if not GEMINI_API_KEY:
        _GEMINI_MODELS_CACHE = preferred
        return preferred
    try:
        r = requests.get(
            'https://generativelanguage.googleapis.com/v1beta/models',
            headers={'x-goog-api-key': GEMINI_API_KEY},
            params={'pageSize': 100},
            timeout=12,
        )
        if r.ok:
            live = []
            for m in r.json().get('models') or []:
                name = (m.get('name') or '').replace('models/', '')
                methods = m.get('supportedGenerationMethods') or []
                if 'generateContent' in methods and 'flash' in name.lower():
                    live.append(name)
            ordered = [m for m in preferred if m in live]
            ordered += [m for m in live if m not in ordered and '2.5' in m]
            if ordered:
                log.info(f'[gemini] using models: {ordered[:4]}')
                _GEMINI_MODELS_CACHE = ordered[:4]
                return _GEMINI_MODELS_CACHE
    except Exception as e:
        log.warning(f'[gemini] model list failed: {e}')
    _GEMINI_MODELS_CACHE = preferred
    return preferred


def _vision_available():
    return bool(GEMINI_API_KEY or ANTHROPIC_KEY)


def _vision_provider_label():
    if GEMINI_API_KEY:
        return GEMINI_MODELS[0]
    if ANTHROPIC_KEY:
        return 'claude-haiku-4-5'
    return 'none'


def _parse_json_response(text, expect='object'):
    """Parse JSON object or array from a vision model response."""
    text = (text or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    if expect == 'array':
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError('Could not parse label data from the photo')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError('Could not parse label data from the photo')


def _gemini_vision(prompt, img_bytes, max_tokens=512):
    """Google Gemini vision (free tier) — REST, no extra package."""
    if not GEMINI_API_KEY:
        raise ValueError('GEMINI_API_KEY not set')
    img_bytes = compress_for_api(img_bytes)
    b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
    payload = {
        'contents': [{'parts': [
            {'inline_data': {'mime_type': 'image/jpeg', 'data': b64}},
            {'text': prompt},
        ]}],
        'generationConfig': {'maxOutputTokens': max_tokens, 'temperature': 0.1},
    }
    last_err = None
    for model in _resolve_gemini_models():
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    headers={'x-goog-api-key': GEMINI_API_KEY, 'Content-Type': 'application/json'},
                    json=payload,
                    timeout=90,
                )
                if r.status_code in (429, 500, 503) and attempt < 2:
                    import time as _t; _t.sleep(0.6 * (attempt + 1))
                    continue
                if not r.ok:
                    try:
                        err_msg = r.json().get('error', {}).get('message', r.text[:200])
                    except Exception:
                        err_msg = r.text[:200]
                    last_err = f'{model}: {err_msg}'
                    log.warning(f'[gemini_vision] {last_err}')
                    break
                data = r.json()
                cands = data.get('candidates') or []
                if not cands:
                    feedback = data.get('promptFeedback') or {}
                    block = feedback.get('blockReason')
                    last_err = f'{model}: blocked ({block})' if block else f'{model}: empty response'
                    log.warning(f'[gemini_vision] {last_err}')
                    break
                parts = (cands[0].get('content') or {}).get('parts') or []
                text = ''.join(p.get('text', '') for p in parts).strip()
                if text:
                    log.info(f'[gemini_vision] ok model={model}')
                    return text
                last_err = f'{model}: no text returned'
                break
            except requests.RequestException as e:
                last_err = str(e)
                if attempt < 2:
                    import time as _t; _t.sleep(0.6 * (attempt + 1))
                    continue
    raise ValueError(f'Gemini vision failed: {last_err or "unknown error"}')


def _anthropic_vision(prompt, img_bytes, max_tokens=512):
    """Anthropic Claude vision — paid fallback when Gemini unavailable."""
    if not ANTHROPIC_KEY:
        raise ValueError('ANTHROPIC_API_KEY not set')
    img_bytes = compress_for_api(img_bytes)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
    last_err = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model='claude-haiku-4-5',
                max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': [
                    {'type': 'image',
                     'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}},
                    {'type': 'text', 'text': prompt},
                ]}],
            )
            for block in (resp.content or []):
                if getattr(block, 'text', None):
                    t = block.text.strip()
                    if t:
                        return t
            raise ValueError('Vision model returned an empty response')
        except Exception as e:
            last_err = e
            transient = any(t in str(e).lower() for t in ('overloaded', '529', 'rate', 'timeout', '500', '503'))
            if transient and attempt < 2:
                import time as _t; _t.sleep(0.6 * (attempt + 1))
                continue
            log.error(f'anthropic_vision error: {e}')
            raise ValueError(f'Vision API error: {e}')
    raise ValueError(f'Vision API error: {last_err}')


def _vision_extract_text(prompt, img_bytes, max_tokens=512):
    """Use Gemini when configured. Never silently fall back to Anthropic."""
    if GEMINI_API_KEY:
        return _gemini_vision(prompt, img_bytes, max_tokens)
    if ANTHROPIC_KEY:
        return _anthropic_vision(prompt, img_bytes, max_tokens)
    raise ValueError(
        'Vision AI not configured — add GEMINI_API_KEY (free at aistudio.google.com) in Render'
    )


# ── Label text parsing (mirrors phone OCR — works offline, no AI) ──
_US_STATES_RE = (
    'MI|OH|IL|IN|CA|NY|TX|FL|PA|GA|NC|VA|WA|AZ|CO|MN|WI|MO|TN|MD|MA|NJ|SC|AL|LA|KY|OR|OK|'
    'CT|UT|IA|NV|AR|MS|KS|NM|NE|WV|ID|HI|NH|ME|MT|RI|DE|SD|ND|AK|VT|WY|DC'
)
_TRACKING_RES = [
    re.compile(r'\b(SPXDTW\d{12,})\b', re.I),
    re.compile(r'\b(SPX[A-Z0-9]{14,})\b', re.I),
    re.compile(r'\b(1Z[A-Z0-9]{16,})\b', re.I),
    re.compile(r'\b(TBA\d{10,})\b', re.I),
    re.compile(r'\b(YWORD\d{10,})\b', re.I),
    re.compile(r'\b(VEHO[A-Z0-9]{8,})\b', re.I),
    re.compile(r'\b(LP\d{10,})\b', re.I),
    re.compile(r'\b(JJD\d{10,})\b', re.I),
]
_SHIP_TO_MARKERS = re.compile(
    r'ship\s*to|deliver\s*to|recipient|consignee|^to\s*:$', re.I
)


def _extract_tracking_from_text(raw):
    if not raw:
        return ''
    compact = re.sub(r'[\s\-]+', '', raw)
    m = re.search(r'SPXDTW\d{12,}', compact, re.I)
    if m:
        return m.group(0).upper()
    for pat in _TRACKING_RES:
        m = pat.search(raw)
        if m:
            return m.group(1).upper().replace(' ', '')
    m = re.search(r'\b(\d{18,})\b', raw)
    return m.group(1) if m else ''


def _is_junk_label_line(line):
    if not line or len(line) < 2:
        return True
    u = line.upper().strip()
    if re.match(r'^SHIP\s*TO$', line, re.I):
        return True
    if u in ('ORD', 'IGD', 'SDX', 'MAY', 'SHEIN', 'SPEEDX', 'FULFILLMENT'):
        return True
    if re.match(r'^(DTW|ORD|IGD|SDX)-[\dA-Z]+$', line, re.I):
        return True
    if re.match(r'^SPX[A-Z0-9]{8,}$', re.sub(r'\s', '', line), re.I):
        return True
    if re.match(r'^C\d{10,}$', re.sub(r'\s', '', line)):
        return True
    if re.match(r'^\d+\.\d+\s*(lb|kg)?$', line, re.I):
        return True
    if re.match(r'^\d{1,2}$', line):
        return True
    return False


def _looks_like_city_state_zip(line):
    if not line:
        return False
    t = line.strip()
    if re.search(r'\b\d{5}(?:-\d{4})?\b', t) and re.search(rf'\b({_US_STATES_RE})\b', t, re.I):
        return True
    return bool(re.match(rf'^[,.\s]*[A-Za-z.\s]+\s+({_US_STATES_RE})\s+\d{{5}}', t, re.I))


def parse_label_text(text):
    """Parse shipping label OCR text into tracking, name, address, zip."""
    if not text:
        return {'tracking': '', 'name': '', 'address': '', 'zip': ''}
    raw = text.replace('\r', '\n')
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    tracking = _extract_tracking_from_text(raw)
    name, address, zip_code = '', '', ''

    ship_idx = next((i for i, l in enumerate(lines) if _SHIP_TO_MARKERS.search(l)), -1)
    block = []
    if ship_idx >= 0:
        for i in range(ship_idx + 1, min(ship_idx + 10, len(lines))):
            line = lines[i]
            if not line or _is_junk_label_line(line):
                continue
            if re.match(r'^[A-Z]{2,4}-[\dA-Z]', line, re.I):
                break
            if re.match(r'^SPX', re.sub(r'\s', '', line), re.I) and len(re.sub(r'\s', '', line)) > 12:
                break
            block.append(line)
            if re.search(rf'\b({_US_STATES_RE})\b\s*\d{{5}}', line, re.I):
                break

    state_zip_re = re.compile(
        rf'^(.+?),\s*({_US_STATES_RE})\s*(\d{{5}})(?:-\d{{4}})?\s*$', re.I
    )
    state_zip_alt = re.compile(rf'^(.+?)\s+({_US_STATES_RE})\s+(\d{{5}})', re.I)

    if block:
        city_line = None
        city, state = '', ''
        for i, line in enumerate(block):
            m = state_zip_re.match(line) or state_zip_alt.match(line)
            if m:
                city_line = i
                city = re.sub(r'[,.\s]+$', '', m.group(1)).strip()
                state = m.group(2).upper()
                zip_code = m.group(3)
                break
        if city_line is not None:
            before = block[:city_line]
            streets = []
            for bi, l in enumerate(before):
                if re.match(r'^\d{2,5}$', l) and streets:
                    streets[-1] = streets[-1] + f' Apt {l}'
                elif re.match(r'^\d', l) or re.search(r'\b(apt|unit|suite|ste|#)\b', l, re.I):
                    streets.append(_normalize_informal_street(l))
                elif not name and re.search(r'[A-Za-z]{2,}', l) and not _looks_like_city_state_zip(l):
                    if not re.search(r'\b(apt|unit|suite)\b', l, re.I):
                        name = l
            street = ', '.join(streets)
            street = re.sub(r',\s*apt\s*(\d+)', r' Apt \1', street, flags=re.I)
            street = re.sub(r'\b(St\.?|Street|Ave\.?|Rd\.?|Dr\.?)\s+(\d{3,4})\b(?!\d)', r'\1 Apt \2', street, flags=re.I)
            city = re.sub(r'^[,.\s]+', '', city)
            if street:
                address = f'{street}, {city}, {state} {zip_code}'.strip()
            else:
                address = f'{city}, {state} {zip_code}'.strip()
            for j, l in enumerate(block):
                if j < city_line and not name and not re.match(r'^\d', l) and not _looks_like_city_state_zip(l):
                    if re.search(r'[A-Za-z]{2,}', l) and not re.search(r'\b(apt|unit|suite)\b', l, re.I):
                        name = l
                        break

    if not address:
        inline = re.search(
            rf'\b([A-Za-z][A-Za-z\s.\'-]{{1,35}}),\s*({_US_STATES_RE})\s*(\d{{5}})', raw, re.I
        )
        if inline:
            zip_code = inline.group(3)
            for line in lines:
                if re.match(r'^\d+\s+[A-Za-z]', line) and len(line) > 8 and not _is_junk_label_line(line):
                    address = f'{line}, {inline.group(1).strip()}, {inline.group(2).upper()} {zip_code}'
                    break

    if _looks_like_city_state_zip(name):
        name = ''
    name = re.sub(r'[^\w\s.\'-]', ' ', name or '').strip()
    name = re.sub(r'\s+', ' ', name)
    if not zip_code and address:
        zm = re.search(r'\b(\d{5})\b', address)
        if zm:
            zip_code = zm.group(1)
    if _is_sender_address(address):
        address, name = '', name
    return {'tracking': tracking, 'name': name, 'address': address, 'zip': zip_code}


def _normalize_tracking_key(tracking):
    return re.sub(r'\s+', '', (tracking or '').upper())


def lookup_label_memory(tracking):
    """Return learned label fields for a tracking number (cross-session)."""
    key = _normalize_tracking_key(tracking)
    if not key or len(key) < 8:
        return None
    try:
        db = get_db()
        row = db.execute(
            "SELECT customer_name, address, zip_code, read_count FROM label_memory WHERE tracking=?",
            (key,),
        ).fetchone()
        db.close()
        if row:
            return {
                'tracking': key,
                'name': row['customer_name'] or '',
                'address': row['address'] or '',
                'zip': row['zip_code'] or '',
                'read_count': row['read_count'] or 0,
            }
    except Exception as e:
        log.warning(f'[label_memory] lookup error: {e}')
    return None


def upsert_label_memory(tracking, name='', address='', zip_code='', source='confirm'):
    """Remember a confirmed label read so future scans are instant."""
    key = _normalize_tracking_key(tracking)
    if not key or len(key) < 8:
        return
    if not address and not name:
        return
    try:
        db = get_db()
        existing = db.execute("SELECT id FROM label_memory WHERE tracking=?", (key,)).fetchone()
        now = datetime.now().isoformat()
        if existing:
            db.execute(
                """UPDATE label_memory SET customer_name=?, address=?, zip_code=?,
                   read_count=read_count+1, last_source=?, updated_at=? WHERE tracking=?""",
                (name or '', address or '', zip_code or '', source, now, key),
            )
        else:
            db.execute(
                """INSERT INTO label_memory (tracking, customer_name, address, zip_code, read_count, last_source, created_at, updated_at)
                   VALUES (?,?,?,?,1,?,?,?)""",
                (key, name or '', address or '', zip_code or '', source, now, now),
            )
        db.commit()
        db.close()
    except Exception as e:
        log.warning(f'[label_memory] upsert error: {e}')


_ADDR_FRAG_SKIP = {
    'SHIP', 'TO', 'FROM', 'THE', 'AND', 'APT', 'UNIT', 'SUITE', 'STE', 'FLOOR', 'FL',
    'SPEEDX', 'SHEIN', 'FEDEX', 'AMAZON', 'DETROIT', 'MICHIGAN', 'ILLINOIS', 'FULFILLMENT',
    'NORTH', 'AURORA', 'OVERLAND', 'ORD', 'IGD', 'SDX', 'MAY', 'STREET', 'AVENUE', 'ROAD',
    'DRIVE', 'LANE', 'COURT', 'PLACE', 'BOULEVARD',
}


def _extract_address_fragments(*texts):
    """Pull partial address pieces from OCR / form fields for inference."""
    combined = ' '.join(t for t in texts if t).strip()
    fr = {
        'street_num': '', 'street_name': '', 'unit': '', 'city': '',
        'state': '', 'zip': '', 'street_tokens': [],
    }
    if not combined:
        return fr

    zm = re.search(r'\b(\d{5})(?:-\d{4})?\b', combined)
    if zm:
        fr['zip'] = zm.group(1)
    sm = re.search(rf'\b({_US_STATES_RE})\b', combined, re.I)
    if sm:
        fr['state'] = sm.group(1).upper()

    csz = re.search(
        rf'([A-Za-z][A-Za-z\s.\'-]{{2,28}}),?\s*({_US_STATES_RE})\s+(\d{{5}})',
        combined, re.I,
    )
    if csz:
        fr['city'] = csz.group(1).strip(' ,.')
        fr['state'] = csz.group(2).upper()
        fr['zip'] = csz.group(3)

    street = re.search(
        r'\b(\d{1,6})\s+([A-Za-z0-9][A-Za-z0-9\s.\'-]{2,40}?'
        r'(?:\s+(?:St\.?|Street|Ave\.?|Avenue|Rd\.?|Road|Dr\.?|Drive|Blvd\.?|Ln\.?|Lane|Ct\.?|Court|Pl\.?|Place|Way|Pkwy\.?))?',
        combined, re.I,
    )
    if street:
        fr['street_num'] = street.group(1)
        fr['street_name'] = re.sub(r'\s+', ' ', street.group(2)).strip(' ,.')

    unit = re.search(r'(?:apt|apartment|unit|suite|ste|#)\s*([A-Za-z0-9-]+)', combined, re.I)
    if unit:
        fr['unit'] = unit.group(1)
    elif fr['street_name']:
        tail = re.search(r'\b(?:St\.?|Street|Ave\.?|Rd\.?|Dr\.?)\s+(\d{3,4})\b', combined, re.I)
        if tail:
            fr['unit'] = tail.group(1)

    words = re.findall(r'[A-Za-z]{4,}', combined)
    fr['street_tokens'] = [
        w for w in words
        if w.upper() not in _ADDR_FRAG_SKIP and w.upper() != fr.get('city', '').upper()
    ][:10]
    return fr


def _score_inferred_address(addr, fr, name_hint=''):
    """Higher score = better match to visible label fragments."""
    if not addr:
        return 0
    u = addr.upper()
    score = 0
    reasons = []
    if fr.get('zip') and fr['zip'] in addr:
        score += 28
        reasons.append('zip')
    if fr.get('street_num') and re.search(r'\b' + re.escape(fr['street_num']) + r'\b', addr):
        score += 32
        reasons.append('street #')
    for tok in fr.get('street_tokens') or []:
        if len(tok) >= 4 and tok.upper() in u:
            score += 14
            reasons.append(tok.lower())
    if fr.get('city') and fr['city'].upper() in u:
        score += 18
        reasons.append('city')
    if fr.get('state') and re.search(rf'\b{re.escape(fr["state"])}\b', u):
        score += 8
        reasons.append('state')
    if fr.get('unit') and fr['unit'].upper() in u:
        score += 16
        reasons.append('unit')
    if name_hint and len(name_hint) > 2 and name_hint.upper() in u:
        score += 6
    return score, reasons


def infer_address_suggestions(tracking='', partial_address='', ocr_text='', name='', zip_code='', limit=5):
    """
    Combine partial label fragments with delivery history to suggest full addresses.
    Used when labels are warped, torn, or weathered.
    """
    parsed = parse_label_text(ocr_text or partial_address or '')
    fr = _extract_address_fragments(partial_address, ocr_text, parsed.get('address', ''))
    if zip_code:
        fr['zip'] = zip_code
    elif parsed.get('zip'):
        fr['zip'] = parsed['zip']
    if not fr.get('street_num') and parsed.get('address'):
        pfr = _extract_address_fragments(parsed['address'])
        for k in ('street_num', 'street_name', 'unit', 'city', 'state', 'zip'):
            if not fr.get(k) and pfr.get(k):
                fr[k] = pfr[k]

    candidates = {}

    def _add(addr, nm='', source='history', base=0):
        if not addr or len(addr) < 8:
            return
        key = _normalize_addr_key(addr)
        sc, reasons = _score_inferred_address(addr, fr, name)
        sc += base
        if sc < 12:
            return
        prev = candidates.get(key)
        if not prev or sc > prev['confidence']:
            label = source
            if source == 'label_memory':
                label = 'past scan'
            elif source == 'address_intel':
                label = 'delivered here before'
            elif source == 'scan_items':
                label = 'recent scan'
            elif source == 'stops':
                label = 'route history'
            elif source == 'geocoder':
                label = 'address lookup'
            reason = 'Matched ' + ', '.join(dict.fromkeys(reasons[:4])) if reasons else 'partial match'
            candidates[key] = {
                'address': addr,
                'name': nm or '',
                'zip': _extract_zip(addr) or fr.get('zip') or '',
                'confidence': min(99, sc),
                'source': label,
                'reason': reason,
            }

    trk = _normalize_tracking_key(tracking)
    if trk:
        mem = lookup_label_memory(trk)
        if mem and mem.get('address'):
            _add(mem['address'], mem.get('name', ''), 'label_memory', base=40)

    tokens = [t for t in fr.get('street_tokens') or [] if len(t) >= 4][:5]
    zip_c = fr.get('zip')
    if not tokens and not zip_c and not fr.get('street_num'):
        return sorted(candidates.values(), key=lambda x: -x['confidence'])[:limit]

    try:
        db = get_db()
        searches = []
        if zip_c and tokens:
            for tok in tokens[:3]:
                like = f'%{tok}%'
                searches.extend([
                    ("SELECT address, customer_name AS nm FROM label_memory WHERE zip_code=? AND UPPER(address) LIKE UPPER(?) LIMIT 8", (zip_c, like), 'label_memory'),
                    ("SELECT address, customer_name AS nm FROM scan_items WHERE zip_code=? AND UPPER(address) LIKE UPPER(?) ORDER BY id DESC LIMIT 8", (zip_c, like), 'scan_items'),
                    ("SELECT address, '' AS nm FROM address_intel WHERE zip_code=? AND UPPER(address) LIKE UPPER(?) ORDER BY delivery_count DESC LIMIT 8", (zip_c, like), 'address_intel'),
                    ("SELECT address, customer_name AS nm FROM stops WHERE UPPER(address) LIKE UPPER(?) AND address LIKE ? ORDER BY id DESC LIMIT 8", (like, f'%{zip_c}%'), 'stops'),
                ])
        if fr.get('street_num') and tokens:
            num = fr['street_num']
            for tok in tokens[:2]:
                like = f'%{num}%{tok}%'
                searches.append(
                    ("SELECT address, customer_name AS nm FROM scan_items WHERE UPPER(address) LIKE UPPER(?) ORDER BY id DESC LIMIT 6", (like,), 'scan_items')
                )
        if zip_c and fr.get('street_num'):
            like = f"{fr['street_num']}%"
            searches.append(
                ("SELECT address, customer_name AS nm FROM scan_items WHERE zip_code=? AND UPPER(address) LIKE UPPER(?) ORDER BY id DESC LIMIT 6", (zip_c, like), 'scan_items')
            )
        for sql, params, src in searches:
            try:
                for row in db.execute(sql, params).fetchall():
                    _add(row['address'], row['nm'] or '', src)
            except Exception:
                pass
        db.close()
    except Exception as e:
        log.warning(f'[infer_address] db search: {e}')

    # Try assembling a best-guess and validating via Census geocoder
    if fr.get('street_num') and (fr.get('street_name') or tokens) and fr.get('zip'):
        street_part = fr.get('street_name') or tokens[0]
        city = fr.get('city') or ''
        state = fr.get('state') or 'MI'
        unit = fr.get('unit')
        base_addr = f"{fr['street_num']} {street_part}"
        if unit:
            base_addr += f" Apt {unit}"
        if city:
            guess = f"{base_addr}, {city}, {state} {fr['zip']}"
        else:
            guess = f"{base_addr}, {state} {fr['zip']}"
        lat, lng = _census_geocode(guess)
        if lat and lng:
            _add(guess, name, 'geocoder', base=22)
        elif city:
            guess2 = f"{base_addr}, {state} {fr['zip']}"
            lat2, lng2 = _census_geocode(guess2)
            if lat2 and lng2:
                _add(guess, name, 'geocoder', base=18)

    out = sorted(candidates.values(), key=lambda x: -x['confidence'])
    seen = set()
    deduped = []
    for item in out:
        k = _normalize_addr_key(item['address'])
        if k in seen:
            continue
        seen.add(k)
        partial_key = _normalize_addr_key(partial_address or '')
        if partial_key and k == partial_key and item['confidence'] < 50:
            continue
        deduped.append(item)
    return deduped[:limit]


_RETURN_ADDRESS_MARKERS = (
    'SHEIN FULFILLMENT', 'NORTH AURORA', 'COMPTON, CA', 'ARTESIA BLVD', 'ARTESIA',
    'CITY OF INDUSTRY', 'COINER CT', 'WILMINGTON, MA', 'ONTARIO, CA', 'JURUPA ST',
    'POINT2POINT', 'RETURN:', 'RETURN ', 'MERCHANT', 'TEMU', 'YC - LOG', 'COMPTON',
    'OVERLAND DRIVE', 'FULFILLMENT',
)
_RETURN_ZIPS = {'60542', '90220', '91748', '91761', '01887', '60642'}


def _is_sender_address(addr):
    if not addr:
        return False
    u = addr.upper()
    if any(m in u for m in _RETURN_ADDRESS_MARKERS):
        return True
    zm = re.search(r'\b(\d{5})\b', u)
    return bool(zm and zm.group(1) in _RETURN_ZIPS)


def _normalize_informal_street(line):
    if not line:
        return line
    line = line.strip()
    if re.search(r'\b(st|street|ave|avenue|rd|road|dr|drive|blvd|way|ln|lane|ct|court|pl|place|pkwy|box)\b', line, re.I):
        return line
    m = re.match(r'^(\d+\s*[A-Za-z]?\s*\d*)\s+([A-Za-z][A-Za-z\s.\'-]+)$', line)
    if m:
        return f'{m.group(1).strip()} {m.group(2).strip().title()} St'
    return line


def _address_quality_issues(address, lat=None, lng=None, route_centroid=None, max_outlier_mi=35):
    issues = []
    if not (address or '').strip():
        return ['empty']
    if _is_sender_address(address):
        issues.append('sender_address')
    if not re.search(r'\b(st|street|ave|avenue|rd|road|dr|drive|blvd|way|ln|lane|ct|court|pl|place|pkwy|box)\b', address, re.I):
        if not re.match(r'^\d+\s+\d', address):
            issues.append('informal_street')
    if lat is None or lng is None:
        issues.append('not_geocoded')
    elif route_centroid:
        try:
            if geodesic((lat, lng), route_centroid).miles > max_outlier_mi:
                issues.append('far_from_route')
        except Exception:
            pass
    return issues


def _route_centroid_from_items(items):
    pts = [(r['dest_lat'], r['dest_lng']) for r in items if r.get('dest_lat') and r.get('dest_lng')]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def compress_for_api(img_bytes, max_bytes=4 * 1024 * 1024):
    """Resize + compress image to stay under Anthropic 5MB API limit."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if max(img.width, img.height) > 1568:
            img.thumbnail((1568, 1568), Image.LANCZOS)
        quality = 85
        while quality >= 40:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue()
            quality -= 10
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=60)
        return buf.getvalue()
    except Exception as e:
        log.error(f'compress_for_api error: {e}')
        return img_bytes


def extract_stops_from_image(img_bytes):
    """Universal carrier-agnostic stop extraction from any delivery app screenshot.
    Supports: Speed X, FedEx, Veho, GoFor, Amazon Flex, OnTrac, and similar apps.
    """
    if not _vision_available():
        raise ValueError('Vision AI not configured on server — add GEMINI_API_KEY (free)')
    prompt = """This is a delivery driver app screenshot. It may be from ANY carrier:
Speed X, FedEx, Veho, GoFor, Amazon Flex, OnTrac, DoorDash, Roadie, or similar.

Extract EVERY delivery stop visible on screen.
Return ONLY a raw JSON array - no markdown, no code blocks, no explanation.

=== HOW TO FIND STOPS ===
Look for repeating card/row patterns that contain an address. Each stop typically has:
  - A delivery address (street, city, state, zip)
  - A customer or recipient name
  - A tracking or barcode number (any format: SPXDTW..., FX..., 1Z..., VEHO..., etc.)
  - A stop number or sequence number
  - A parcel/package count

=== ADDRESS PARSING RULES ===
Addresses may appear in different formats depending on the carrier app:

Format A - Comma-separated (Speed X, some others):
  "287 Alfred St,Detroit,MI,48201-3122,USA"
  "124 Alfred St 206,DETROIT,MI,48201,USA"  -> unit is 206
  "66 Winder St Apt 338,Detroit,MI,48201,USA"  -> unit is 338
  RULE: If a number appears between street and city with no label -> that is the unit/apt
  RULE: "Apt", "Apartment", "Unit", "#", "Suite", "Ste" before a number = unit label, number = unit

Format B - Multi-line (FedEx, Veho, Amazon):
  Line 1: "320 Edmund Pl"
  Line 2: "Suite 210"  or  "Apt 4B"
  Line 3: "Detroit, MI 48201"

Format C - Single line:
  "87 East Canfield Street, Storefront, Detroit, MI 48201"

=== OUTPUT RULES ===
- address: Full clean address. Format: "{street}, {City}, {STATE} {5-digit-zip}"
  - Drop ",USA" or "United States" from output
  - Use Title Case for city (Detroit not DETROIT)
  - Use only 5-digit zip (drop "-3193" from "48201-3193")
  - Do NOT include apt/unit in the address field - put it in the "unit" field
- unit: Apartment, unit, suite, floor, or storefront identifier. Empty string if none.
- name: Recipient name. If truncated ("Jaleeza Anz...") include what is visible.
- tracking: Full tracking number - copy exactly, any format (SPXDTW..., YWORD..., 1Z..., etc.)
- stop_num: Stop or sequence number shown on card. Empty string if not visible.
- carrier: Detected carrier name if visible ("Speed X", "FedEx", "Veho", "GoFor", etc.). "Unknown" if not clear.

JSON array format:
[
  {
    "stop_num": "52",
    "address": "287 Alfred St, Detroit, MI 48201",
    "unit": "",
    "name": "Jaleeza Anz...",
    "tracking": "SPXDTW138600193720",
    "carrier": "Speed X"
  }
]

Include EVERY stop card visible. Do not skip any.
If a field is not visible, use empty string - never null.
Return ONLY the JSON array."""
    try:
        text = _vision_extract_text(prompt, img_bytes, max_tokens=4096)
        log.info(f'[extract_stops] vision raw (first 400): {text[:400]}')
        stops = _parse_json_response(text, expect='array')
        return [{
            'address':  s.get('address', '').strip(),
            'name':     s.get('name', '').strip(),
            'tracking': s.get('tracking', '').strip(),
            'stop_num': str(s.get('stop_num', '')).strip(),
            'unit':     s.get('unit', '').strip(),
            'phone':    re.sub(r'\D', '', s.get('phone', '')),
            'carrier':  s.get('carrier', '').strip(),
        } for s in stops if s.get('address')]
    except ValueError:
        log.warning('[extract_stops] could not parse JSON array from vision response')
        return []
    except Exception as e:
        log.error(f'Vision API error ({type(e).__name__}): {e}')
        raise
        raise


def extract_package_label(img_bytes):
    """Extract delivery info from a shipping label photo (Gemini free / Claude fallback)."""
    if not _vision_available():
        raise ValueError(
            'Vision AI not configured — add GEMINI_API_KEY (free at aistudio.google.com) in Render'
        )
    prompt = '''This is a shipping label photo. Carriers include SpeedX, SHEIN, FedEx, UPS, Amazon, Veho, OnTrac, etc.
Return ONLY a JSON object, no markdown, no code blocks.

IMPORTANT — ignore these (NOT the delivery address):
- Sort/hub codes like ORD, DTW-08B, IGD, SDX
- Sender/return address (warehouse, fulfillment center, North Aurora IL, etc.)
- Weight, dates, handwritten route numbers, internal codes

Extract ONLY from the recipient block (look for "SHIP TO", "DELIVER TO", "TO:", or recipient name above street):
- tracking: longest barcode number (SpeedX: SPXDTW... 20+ chars). Copy EXACTLY.
- name: recipient name on SHIP TO line (e.g. "EvaMarie Jordan") — NOT city/state/zip
- address: full SHIP TO delivery address as one string: street + apt/unit + city + state + zip
  Example: "888 Pallister St Apt 705, Detroit, MI 48202"
  If unit appears as trailing number after street (e.g. "1310 Pallister St 807"), format as "1310 Pallister St Apt 807, Detroit, MI 48202"
- zip: 5-digit zip from SHIP TO only

Example:
{"tracking": "SPXDTW013662606010010144", "name": "EvaMarie Jordan", "address": "888 Pallister St Apt 705, Detroit, MI 48202", "zip": "48202"}

If a field is not visible, use empty string.
Return ONLY the JSON object.'''
    try:
        text = _vision_extract_text(prompt, img_bytes, max_tokens=512)
    except Exception as e:
        log.error(f'extract_package_label API error: {e}')
        raise ValueError(str(e))
    if not text:
        raise ValueError('Vision model returned an empty response')
    try:
        return _parse_json_response(text, expect='object')
    except json.JSONDecodeError:
        log.error(f'extract_package_label parse error; raw: {text[:200]}')
        raise ValueError('Could not parse label data from the photo')


from twilio.rest import Client
import stripe

# ─── DATABASE ABSTRACTION (SQLite local / PostgreSQL on Render) ─
DATABASE_URL = os.environ.get('DATABASE_URL', '')
USE_PG = bool(DATABASE_URL)

if USE_PG:
    import pg8000.dbapi as pg8000

class DBWrapper:
    """Normalizes sqlite3 and psycopg2 so the rest of the app is unchanged."""
    def __init__(self, conn, pg=False):
        self._conn = conn
        self._pg   = pg
        self._cur  = conn.cursor()
        self._last = None

    def _fix(self, q):
        """Translate SQLite ? placeholders and functions to PostgreSQL."""
        if not self._pg: return q
        q = q.replace('?', '%s')
        q = q.replace('last_insert_rowid()', 'lastval()')
        q = q.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')
        q = q.replace('rowid', 'id')
        return q

    def execute(self, query, params=None):
        self._cur.execute(self._fix(query), params or ())
        self._last = self._cur
        return self

    def executescript(self, script):
        """Execute multiple statements — splits on ; for PostgreSQL."""
        if self._pg:
            for stmt in script.split(';'):
                stmt = stmt.strip()
                if not stmt: continue
                if stmt.upper().startswith('PRAGMA'): continue
                try:
                    self._cur.execute(self._fix(stmt))
                    self._conn.commit()
                except Exception:
                    try: self._conn.rollback()
                    except: pass
        else:
            self._conn.executescript(script)
        return self

    class _RowDict(dict):
        """Dict that also supports integer index access (row[0] == first value)."""
        def __getitem__(self, key):
            if isinstance(key, int):
                return list(self.values())[key]
            return super().__getitem__(key)
        def __contains__(self, key):
            if isinstance(key, int):
                return key < len(self)
            return super().__contains__(key)

    def _to_dict(self, row):
        """Convert pg8000 tuple row to RowDict supporting both col name and int index."""
        if row is None: return None
        cols = [d[0] for d in self._cur.description]
        return DBWrapper._RowDict(zip(cols, row))

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None: return None
        if self._pg: return self._to_dict(row)
        return row

    def fetchall(self):
        rows = self._cur.fetchall() or []
        if self._pg: return [self._to_dict(r) for r in rows]
        return rows

    def __getitem__(self, key):
        """Allow row['col'] on last fetchone result (compatibility)."""
        return self._cur.fetchone()[key]

    def commit(self):
        self._conn.commit()

    def close(self):
        try: self._conn.close()
        except: pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'unit-secret-2025')

@app.template_filter('name_short')
def name_short_filter(name):
    """Format name as 'First L.' — e.g. 'Ebony Helton' -> 'Ebony H.' """
    if not name:
        return ''
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    last_initial = parts[-1][0].upper() + '.'
    return f'{first} {last_initial}'

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'): return jsonify({'error': 'Not found'}), 404
    return render_template('error.html', code=404, msg='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    log.error(f'500: {traceback.format_exc()}')
    if request.path.startswith('/api/'): return jsonify({'error': 'Server error'}), 500
    return render_template('error.html', code=500, msg='Something went wrong'), 500

@app.errorhandler(Exception)
def unhandled(e):
    log.error(f'Unhandled: {traceback.format_exc()}')
    if request.path.startswith('/api/'): return jsonify({'error': str(e)}), 500
    return render_template('error.html', code=500, msg='Unexpected error — please try again'), 500

DB = 'data/unit.db'
MAPBOX_TOKEN    = os.environ.get('MAPBOX_TOKEN', '')
GOOGLE_MAPS_KEY = os.environ.get('GOOGLE_MAPS_KEY', '')

# ── VEHICLE ZONE CONFIGS ──────────────────────────────────────────
# Each vehicle type defines ordered zones: index 0 = load FIRST (deepest),
# last index = load LAST (closest to door). Delivery order maps inversely.
VEHICLE_ZONES = {
    'small_car': [
        {'id': 'trunk-back',   'label': 'Trunk — Back Wall',   'icon': '🔵', 'desc': 'Against back wall of trunk (load first)'},
        {'id': 'trunk-mid',    'label': 'Trunk — Middle',      'icon': '🔵', 'desc': 'Middle of trunk'},
        {'id': 'trunk-front',  'label': 'Trunk — Front',      'icon': '🟡', 'desc': 'Front of trunk near seat (load last)'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',  'icon': '🟠', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',   'icon': '🟠', 'desc': 'Left rear passenger seat'},
        {'id': 'backseat-mid', 'label': 'Back Seat — Middle', 'icon': '🟠', 'desc': 'Middle rear seat'},
    ],
    'sedan': [
        {'id': 'trunk-back',   'label': 'Trunk — Back Wall',   'icon': '🔵', 'desc': 'Against back wall of trunk (load first)'},
        {'id': 'trunk-mid',    'label': 'Trunk — Middle',      'icon': '🔵', 'desc': 'Middle of trunk'},
        {'id': 'trunk-front',  'label': 'Trunk — Front',      'icon': '🟡', 'desc': 'Front of trunk near seat'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',  'icon': '🟠', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',   'icon': '🟠', 'desc': 'Left rear passenger seat'},
        {'id': 'backseat-mid', 'label': 'Back Seat — Middle', 'icon': '🟠', 'desc': 'Middle rear seat'},
        {'id': 'front-pass',   'label': 'Front Passenger',     'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'suv_midsize': [
        # No 3rd row — Jeep Grand Cherokee, Toyota 4Runner, Ford Explorer, etc.
        {'id': 'cargo-back',   'label': 'Cargo — Back',        'icon': '🔵', 'desc': 'Against rear seats — load first'},
        {'id': 'cargo-mid',    'label': 'Cargo — Middle',     'icon': '🔵', 'desc': 'Center of cargo area'},
        {'id': 'cargo-lift',   'label': 'Cargo — Liftgate',   'icon': '🟡', 'desc': 'Near liftgate — grab first'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',  'icon': '🟠', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-mid', 'label': 'Back Seat — Middle', 'icon': '🟠', 'desc': 'Middle rear seat'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',   'icon': '🟠', 'desc': 'Left rear passenger seat'},
        {'id': 'front-pass',   'label': 'Front Passenger',     'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'suv_fullsize': [
        # With 3rd row — Chevy Tahoe, GMC Yukon, Ford Expedition, etc.
        {'id': 'cargo-back',   'label': 'Cargo — Back',        'icon': '🔵', 'desc': 'Behind 3rd row or folded flat — load first'},
        {'id': 'cargo-mid',    'label': 'Cargo — Middle',     'icon': '🔵', 'desc': 'Center of cargo area'},
        {'id': 'cargo-lift',   'label': 'Cargo — Liftgate',   'icon': '🟡', 'desc': 'Near liftgate — grab first'},
        {'id': 'row3-r',       'label': '3rd Row — Right',    'icon': '🟠', 'desc': '3rd row folded flat, right side'},
        {'id': 'row3-l',       'label': '3rd Row — Left',     'icon': '🟠', 'desc': '3rd row folded flat, left side'},
        {'id': 'backseat-r',   'label': '2nd Row — Right',    'icon': '⚪', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-l',   'label': '2nd Row — Left',     'icon': '⚪', 'desc': 'Left rear passenger seat'},
        {'id': 'front-pass',   'label': 'Front Passenger',     'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'minivan': [
        {'id': 'cargo-back',   'label': 'Cargo — Back',       'icon': '🔵', 'desc': 'Rear cargo behind seats (load first)'},
        {'id': 'row3-r',       'label': '3rd Row — Right',    'icon': '🔵', 'desc': 'Right side 3rd row (fold flat)'},
        {'id': 'row3-l',       'label': '3rd Row — Left',     'icon': '🔵', 'desc': 'Left side 3rd row (fold flat)'},
        {'id': 'row2-r',       'label': '2nd Row — Right',    'icon': '🟠', 'desc': "Right captain's chair area"},
        {'id': 'row2-l',       'label': '2nd Row — Left',     'icon': '🟠', 'desc': "Left captain's chair area"},
        {'id': 'row2-mid',     'label': '2nd Row — Middle',   'icon': '🟠', 'desc': 'Center aisle / middle row'},
        {'id': 'front-pass',   'label': 'Front Passenger',     'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'pickup': [
        {'id': 'bed-cab',      'label': 'Bed — Cab Wall',     'icon': '🔵', 'desc': 'Against cab wall — load first'},
        {'id': 'bed-mid',      'label': 'Bed — Middle',       'icon': '🔵', 'desc': 'Middle of truck bed'},
        {'id': 'bed-gate',     'label': 'Bed — Tailgate',     'icon': '🟡', 'desc': 'Near tailgate — grab first'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',  'icon': '🟠', 'desc': 'Crew cab right rear'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',   'icon': '🟠', 'desc': 'Crew cab left rear'},
        {'id': 'front-pass',   'label': 'Front Passenger',     'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'cargo_van': [
        {'id': 'A1', 'label': 'Front Left — Bulkhead',  'icon': '🔵', 'desc': 'Driver side bulkhead wall — load first'},
        {'id': 'A2', 'label': 'Front Right — Bulkhead', 'icon': '🔵', 'desc': 'Passenger side bulkhead wall'},
        {'id': 'B1', 'label': 'Mid Left',               'icon': '🔵', 'desc': 'Mid-van left side'},
        {'id': 'B2', 'label': 'Mid Right',              'icon': '🔵', 'desc': 'Mid-van right side'},
        {'id': 'C1', 'label': 'Rear Left — Doors',     'icon': '🟡', 'desc': 'Near rear doors, left side'},
        {'id': 'C2', 'label': 'Rear Right — Doors',    'icon': '🟡', 'desc': 'Near rear doors, right side'},
        {'id': 'C3', 'label': 'Door Stack',             'icon': '🟡', 'desc': 'Stacked right at door opening — grab first'},
    ],
    'box_truck': [
        {'id': 'A1', 'label': 'Row A — Left Front',   'icon': '🔵', 'desc': 'Front of box, driver side — load first'},
        {'id': 'A2', 'label': 'Row A — Right Front',  'icon': '🔵', 'desc': 'Front of box, passenger side'},
        {'id': 'B1', 'label': 'Row B — Left Mid',     'icon': '🔵', 'desc': 'Middle of box, left'},
        {'id': 'B2', 'label': 'Row B — Right Mid',    'icon': '🔵', 'desc': 'Middle of box, right'},
        {'id': 'C1', 'label': 'Row C — Left Rear',    'icon': '🟡', 'desc': 'Near door, left side'},
        {'id': 'C2', 'label': 'Row C — Right Rear',   'icon': '🟡', 'desc': 'Near door, right side'},
        {'id': 'D1', 'label': 'Door — Left',          'icon': '🟡', 'desc': 'Right at door — grab first'},
        {'id': 'D2', 'label': 'Door — Right',         'icon': '🟡', 'desc': 'Right at door — grab first'},
    ],
}

VEHICLE_LABELS = {
    'small_car':   '🚗 Small Car',
    'sedan':       '🚗 Sedan',
    'suv_midsize': '🚙 SUV — Midsize (No 3rd Row)',
    'suv_fullsize':'🚙 SUV — Full Size (3rd Row)',
    'minivan':     '🚐 Minivan',
    'pickup':      '🚚 Pickup Truck',
    'cargo_van':   '🚐 Cargo Van',
    'box_truck':   '🚚 Box Truck',
}

# ── DELIVERY ZONE COLORS (A–F) ───────────────────────────────────────
ZONE_COLORS = {
    'A': {'hex': '#ef4444', 'name': 'Red',    'emoji': '🔴'},
    'B': {'hex': '#3b82f6', 'name': 'Blue',   'emoji': '🔵'},
    'C': {'hex': '#10b981', 'name': 'Green',  'emoji': '🟢'},
    'D': {'hex': '#f59e0b', 'name': 'Yellow', 'emoji': '🟡'},
    'E': {'hex': '#8b5cf6', 'name': 'Purple', 'emoji': '🟣'},
    'F': {'hex': '#f97316', 'name': 'Orange', 'emoji': '🟠'},
    'G': {'hex': '#06b6d4', 'name': 'Cyan',   'emoji': '🩵'},
    'H': {'hex': '#ec4899', 'name': 'Pink',   'emoji': '🩷'},
}

def calc_num_zones_adaptive(geocoded):
    """
    Determine zone count from geographic spread + package count.
    Calibrated for Detroit ZIP code scale (~3-4 mile diameter per zip).
    Also scales up zone count for large routes so no single zone is overwhelming.
    """
    n = len(geocoded)
    if n < 2: return 1
    lats = [p['lat'] for p in geocoded]
    lngs = [p['lng'] for p in geocoded]
    span = geodesic((min(lats), min(lngs)), (max(lats), max(lngs))).miles
    # Base zone count from geographic spread
    if span < 0.5:   k = 1
    elif span < 1.2: k = 2
    elif span < 2.5: k = 3
    elif span < 4.0: k = 4
    elif span < 6.0: k = 5
    elif span < 9.0: k = 6
    elif span < 12.0:k = 7
    else:            k = 8
    # Scale up if route is large — no zone should exceed ~20 stops
    # (keeps each zone manageable and evenly distributed)
    min_by_count = math.ceil(n / 20)
    k = max(k, min_by_count)
    return min(k, n, 8)    # never more than 8 zones or more than packages


def _dsq(a, b):
    """Squared Euclidean distance on lat/lng (fast, good enough for city scale)."""
    return (a['lat'] - b['lat'])**2 + (a['lng'] - b['lng'])**2

def kmeans_geo(points, k, max_iter=40, seed_centroids=None):
    """
    K-means clustering on lat/lng dicts. Returns list of cluster indices.
    seed_centroids: optional list of {lat, lng} dicts from address_intel history.
    When provided and count >= k, uses historical cluster positions as starting
    points instead of random k-means++ init — produces more consistent zones.
    """
    n = len(points)
    if k > n:  return list(range(n))   # more clusters than points — 1 each
    if k <= 1: return [0] * n

    # ── Centroid initialization ──
    if seed_centroids and len(seed_centroids) >= k:
        # Derive k representative seeds from historical points using mini k-means++
        # Pick k spread-out points from the seed pool as starting centroids
        seeds    = list(seed_centroids)
        chosen   = [seeds[0]]
        for _ in range(k - 1):
            dists  = [min(_dsq(s, c) for c in chosen) for s in seeds]
            best_i = max(range(len(seeds)), key=lambda i: dists[i])
            chosen.append(seeds[best_i])
        centroids = [{'lat': c['lat'], 'lng': c['lng']} for c in chosen]
        log.info(f'[address_intel] k-means seeded from {len(seed_centroids)} historical points')
    else:
        # Standard k-means++ init from current route points
        centroids = [{'lat': points[0]['lat'], 'lng': points[0]['lng']}]
        for _ in range(k - 1):
            dists = [min(_dsq(p, c) for c in centroids) for p in points]
            best  = max(range(n), key=lambda i: dists[i])
            centroids.append({'lat': points[best]['lat'], 'lng': points[best]['lng']})

    assignments = [0] * n
    for _ in range(max_iter):
        new_asgn = [min(range(k), key=lambda j: _dsq(p, centroids[j])) for p in points]
        if new_asgn == assignments:
            break
        assignments = new_asgn
        for j in range(k):
            grp = [points[i] for i in range(n) if assignments[i] == j]
            if grp:
                centroids[j] = {
                    'lat': sum(p['lat'] for p in grp) / len(grp),
                    'lng': sum(p['lng'] for p in grp) / len(grp),
                }
    return assignments

BAG_SIZE = 8  # stops per bag within a zone

def _mark_unknown(pkgs):
    for p in pkgs:
        p.update({'zone_letter':'?','zone_num':0,'zone_label_full':'?',
                  'zone_color':'#6b7280','zone_emoji':'⚪','bag_num':0,'bag_label':'?'})

def cluster_packages_geo(geocoded):
    """
    Smart adaptive zone clustering.

    Day 1  — pure geographic k-means (best guess).
    Day 3+ — high-confidence addresses are PINNED to their historical zone;
             floating addresses cluster around the pinned anchors.
             A balance pass then evens out zone sizes so no zone is overloaded.

    Gets smarter with every delivery. By day 4-5 of the same route,
    zone boundaries are essentially locked in.

    Returns dict: {cluster_index: [pkg, ...]} and cluster_letter map.
    """
    n = len(geocoded)
    if n == 0:
        return {}, {}
    k = calc_num_zones_adaptive(geocoded)
    if k <= 1 or n < 2:
        return {0: geocoded}, {0: 'A'}

    # ── Step 1: Get address-level confidence ───────────────────────────
    addresses  = [p.get('address', '') for p in geocoded]
    confidence = get_address_zone_confidence(addresses)

    pinned_pkgs   = []   # high-confidence: zone known
    floating_pkgs = []   # new/inconsistent: needs clustering
    for p in geocoded:
        key = _normalize_addr_key(p.get('address', ''))
        info = confidence.get(key)
        if info and info['pinned']:
            p['_pinned_zone']      = info['zone']
            p['_zone_confidence']  = info['confidence']
            p['_zone_count']       = info['count']
            pinned_pkgs.append(p)
        else:
            p['_pinned_zone']      = None
            p['_zone_confidence']  = info['confidence'] if info else 0.0
            p['_zone_count']       = info['count'] if info else 0
            floating_pkgs.append(p)

    pinned_ratio = len(pinned_pkgs) / n
    log.info(f'[zone_learn] {len(pinned_pkgs)}/{n} pinned ({pinned_ratio:.0%}), '
             f'{len(floating_pkgs)} floating, k={k}')

    # ── Step 2: Build initial groups ───────────────────────────────
    # Build zone-letter → cluster-index map from pinned packages
    zone_to_cluster = {}   # historical zone letter → cluster int
    cluster_to_pkgs = {}   # cluster int → [pkg, ...]
    next_cluster    = [0]

    def _get_or_create_cluster(zone_letter):
        if zone_letter not in zone_to_cluster:
            zone_to_cluster[zone_letter] = next_cluster[0]
            cluster_to_pkgs[next_cluster[0]] = []
            next_cluster[0] += 1
        return zone_to_cluster[zone_letter]

    for p in pinned_pkgs:
        c = _get_or_create_cluster(p['_pinned_zone'])
        cluster_to_pkgs[c].append(p)

    # ── Step 3: Cluster floating packages ───────────────────────────
    if floating_pkgs:
        if pinned_pkgs and pinned_ratio >= 0.4:
            # Enough anchors: cluster floaters around pinned centroids
            pinned_centroids = [
                {'lat': sum(p['lat'] for p in pkgs) / len(pkgs),
                 'lng': sum(p['lng'] for p in pkgs) / len(pkgs)}
                for pkgs in cluster_to_pkgs.values() if pkgs
            ]
            # Create extra clusters if needed to reach target k
            extra_k = max(0, k - len(cluster_to_pkgs))
            if extra_k > 0 and len(floating_pkgs) >= extra_k:
                # Add new clusters for geographic areas not yet covered by pinned anchors
                extra_assignments = kmeans_geo(floating_pkgs, extra_k)
                extra_groups = {}
                for i, p in enumerate(floating_pkgs):
                    extra_groups.setdefault(extra_assignments[i], []).append(p)
                for eg in extra_groups.values():
                    if eg:
                        c = next_cluster[0]
                        cluster_to_pkgs[c] = eg
                        next_cluster[0] += 1
            else:
                # Assign each floater to nearest pinned centroid
                existing_clusters = [cid for cid, pkgs in cluster_to_pkgs.items() if pkgs]
                existing_centroids = [
                    {'cluster': cid,
                     'lat': sum(p['lat'] for p in cluster_to_pkgs[cid]) / len(cluster_to_pkgs[cid]),
                     'lng': sum(p['lng'] for p in cluster_to_pkgs[cid]) / len(cluster_to_pkgs[cid])}
                    for cid in existing_clusters
                ]
                for p in floating_pkgs:
                    nearest_c = min(existing_centroids, key=lambda c: _dsq(p, c))
                    cluster_to_pkgs[nearest_c['cluster']].append(p)
        else:
            # Not enough pinned history — pure k-means on all packages
            zips       = set(filter(None, (_extract_zip(p.get('address', '')) for p in geocoded)))
            hist_seeds = get_historical_centroids_for_zips(zips) if zips else []
            assignments = kmeans_geo(
                geocoded, k,
                seed_centroids=hist_seeds if len(hist_seeds) >= k else None
            )
            cluster_to_pkgs = {}
            for i, p in enumerate(geocoded):
                cluster_to_pkgs.setdefault(assignments[i], []).append(p)

    # Remove empty clusters
    groups = {k: v for k, v in cluster_to_pkgs.items() if v}

    # ── Step 4: Balance pass ───────────────────────────────────
    # Only balance floating packages (don't move pinned anchors)
    if len(groups) > 1 and floating_pkgs:
        groups = _balance_zones(groups, max_ratio=1.4)

    # ── Step 5: Order zones by route geography ─────────────────────
    centroids_list = [
        {'cluster': c,
         'lat': sum(p['lat'] for p in pts) / len(pts),
         'lng': sum(p['lng'] for p in pts) / len(pts)}
        for c, pts in groups.items()
    ]
    start     = {'lat': geocoded[0]['lat'], 'lng': geocoded[0]['lng']}
    ordered   = []
    remaining = list(centroids_list)
    cur       = start
    while remaining:
        nearest = min(remaining, key=lambda c: _dsq(cur, c))
        ordered.append(nearest['cluster'])
        cur = nearest
        remaining.remove(nearest)
    cluster_letter = {c: chr(65 + seq) for seq, c in enumerate(ordered)}
    return groups, cluster_letter

# ── OSRM chunk size: API handles ~100 waypoints reliably ──
OSRM_MAX_WAYPOINTS = 90
BUILDING_GROUP_METERS = 55  # same building if coords within this distance


def _street_base(address):
    """Strip unit/apt for building-level grouping."""
    if not address:
        return ''
    a = _normalize_addr_key(address)
    a = re.sub(r'\b(APT|APARTMENT|UNIT|STE|SUITE|#|FL|FLOOR|RM|ROOM|BLDG|BUILDING)\s*[\w-]+', '', a, flags=re.I)
    a = re.sub(r',\s*,', ',', a)
    return re.sub(r'\s+', ' ', a).strip(' ,')


def _building_group_key(address, lat=None, lng=None):
    base = _street_base(address)
    if base:
        return base
    if lat is not None and lng is not None:
        return f'COORD:{round(float(lat), 4)}:{round(float(lng), 4)}'
    return ''


def group_packages_by_stop(packages, proximity_m=BUILDING_GROUP_METERS):
    """
    Merge packages at the same building into one routable stop.
    Returns list of {addr_key, lat, lng, address, packages: [...]}.
    """
    groups = []
    for p in packages:
        if not p.get('lat') or not p.get('lng'):
            continue
        addr_key = _building_group_key(p.get('address', ''), p['lat'], p['lng'])
        placed = False
        for g in groups:
            if addr_key and g['addr_key'] == addr_key:
                g['packages'].append(p)
                placed = True
                break
            try:
                dist = geodesic((p['lat'], p['lng']), (g['lat'], g['lng'])).meters
            except Exception:
                dist = 9999
            if dist <= proximity_m:
                if addr_key and g['addr_key'] and addr_key == g['addr_key']:
                    g['packages'].append(p)
                    placed = True
                    break
                if _street_base(p.get('address', '')) == _street_base(g.get('address', '')):
                    g['packages'].append(p)
                    placed = True
                    break
        if not placed:
            groups.append({
                'addr_key': addr_key or f"PT:{p.get('id', id(p))}",
                'lat': p['lat'],
                'lng': p['lng'],
                'address': p.get('address', ''),
                'packages': [p],
            })
    # Packages missing coords — one group each
    for p in packages:
        if p.get('lat') and p.get('lng'):
            continue
        groups.append({
            'addr_key': _building_group_key(p.get('address', '')) or f"NGEO:{p.get('id', id(p))}",
            'lat': p.get('lat'),
            'lng': p.get('lng'),
            'address': p.get('address', ''),
            'packages': [p],
        })
    return groups


def _nn_sort(pkgs, start=None):
    """
    Nearest-neighbor TSP fallback.
    Greedily picks the closest unvisited stop at every step.
    Much better than random order; used when OSRM is unavailable.
    """
    if len(pkgs) <= 1:
        return list(pkgs)
    remaining = list(pkgs)
    if start and start.get('lat') is not None and start.get('lng') is not None:
        cur = {'lat': float(start['lat']), 'lng': float(start['lng'])}
        ordered = []
    else:
        cur = min(remaining, key=lambda p: p.get('lat', 0))
        remaining.remove(cur)
        ordered = [cur]
    while remaining:
        nxt = min(remaining, key=lambda p: _dsq(cur, p))
        remaining.remove(nxt)
        ordered.append(nxt)
        cur = nxt
    return ordered


def _osrm_trip(pkgs, start=None, end=None, timeout=12):
    """
    Run OSRM trip on packages. Optional start/end virtual waypoints fix depot routing.
    Uses input-index mapping (not proximity matching) to avoid order swaps in dense areas.
    """
    all_pts = []
    pkg_indices = []

    if start and start.get('lat') is not None and start.get('lng') is not None:
        all_pts.append({'lat': float(start['lat']), 'lng': float(start['lng']), '_virtual': True})

    for i, p in enumerate(pkgs):
        all_pts.append(p)
        pkg_indices.append(i)

    if end and end.get('lat') is not None and end.get('lng') is not None:
        all_pts.append({'lat': float(end['lat']), 'lng': float(end['lng']), '_virtual': True})

    if len(all_pts) <= 1:
        return list(pkgs), 0, 0

    has_start = start and start.get('lat') is not None
    has_end = end and end.get('lat') is not None
    if has_start and has_end:
        src, dst = 'first', 'last'
    elif has_start:
        src, dst = 'first', 'any'
    elif has_end:
        src, dst = 'any', 'last'
    else:
        src, dst = 'any', 'any'

    coords = ';'.join(f"{p['lng']},{p['lat']}" for p in all_pts)
    url = (f"http://router.project-osrm.org/trip/v1/driving/{coords}"
           f"?roundtrip=false&source={src}&destination={dst}&overview=false")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get('code') != 'Ok' or not data.get('trips'):
        raise ValueError(f"OSRM returned: {data.get('code')}")

    trip = data['trips'][0]
    visit_indices = trip.get('waypoints') or []
    if visit_indices and isinstance(visit_indices[0], dict):
        visit_indices = [w.get('waypoint_index', i) for i, w in enumerate(visit_indices)]
    if not visit_indices:
        visit_indices = list(range(len(all_pts)))

    offset = 1 if has_start else 0
    ordered, seen = [], set()
    for vi in visit_indices:
        if vi < offset or vi >= offset + len(pkgs):
            continue
        pi = vi - offset
        if pi < 0 or pi >= len(pkgs):
            continue
        p = pkgs[pi]
        pid = p.get('id', pi)
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(p)
    for p in pkgs:
        pid = p.get('id', id(p))
        if pid not in seen:
            ordered.append(p)

    dist = trip.get('distance', 0)
    dur = trip.get('duration', 0)
    return ordered, dist, dur


def _street_name_key(address):
    """Street name only (no house number, no unit) — for same-street sequencing."""
    base = _street_base(address)
    if not base:
        return ''
    first = base.split(',')[0].strip()
    m = re.match(r'^[\d-]+\s+(.*)$', first)
    return (m.group(1) if m else first).strip()


def _group_same_street_runs(ordered):
    """Pull stops on the same street together so they're delivered back-to-back.

    Keeps the optimizer's order for first occurrences; later stops on an
    already-visited street are moved up to follow it. Only used within a
    single compact zone so the distance cost is negligible.
    """
    if len(ordered) <= 2:
        return ordered
    result, used = [], set()
    for i, s in enumerate(ordered):
        if i in used:
            continue
        result.append(s)
        used.add(i)
        sk = _street_name_key(s.get('address', ''))
        if not sk:
            continue
        for j in range(i + 1, len(ordered)):
            if j in used:
                continue
            if _street_name_key(ordered[j].get('address', '')) == sk:
                result.append(ordered[j])
                used.add(j)
    return result


def _serpentine_zone_order(zone_lists, start=None, end=None):
    """Chain geographic zones into a sweep (S-curve): finish one area, move to
    the adjacent one, never bounce back. Greedy nearest-centroid ordering.

    - With a start point: sweep outward from the start.
    - With only an end point: chain backwards from the end, then reverse.
    - With neither: sweep from the northernmost zone downward.
    """
    if len(zone_lists) <= 1:
        return zone_lists
    cents = [
        {'lat': sum(s['lat'] for s in zl) / len(zl),
         'lng': sum(s['lng'] for s in zl) / len(zl)}
        for zl in zone_lists
    ]
    remaining = list(range(len(zone_lists)))
    order = []
    reverse_at_end = False

    if start and start.get('lat') is not None:
        cur = {'lat': float(start['lat']), 'lng': float(start['lng'])}
    elif end and end.get('lat') is not None:
        cur = {'lat': float(end['lat']), 'lng': float(end['lng'])}
        reverse_at_end = True
    else:
        first = max(remaining, key=lambda i: cents[i]['lat'])
        remaining.remove(first)
        order.append(first)
        cur = cents[first]

    while remaining:
        nxt = min(remaining, key=lambda i: _dsq(cur, cents[i]))
        remaining.remove(nxt)
        order.append(nxt)
        cur = cents[nxt]

    if reverse_at_end:
        order.reverse()
    return [zone_lists[i] for i in order]


def osrm_optimize_full_route(pkgs, start=None, end=None):
    """
    Globally optimize ALL stops in a single OSRM call (or chunked for large routes).
    This is the correct approach — optimize first, then assign zones.

    For routes > OSRM_MAX_WAYPOINTS:
      - Run OSRM on first chunk to get the anchor order
      - For each subsequent chunk, find the nearest-neighbor handoff from the
        last stop of the previous chunk, then OSRM optimize that chunk
      - Stitch together

    Returns (ordered_pkgs, total_dist_m, total_dur_s)
    """
    n = len(pkgs)
    if n == 0:
        return [], 0, 0
    if n == 1:
        return list(pkgs), 0, 0

    # ── Small route: single OSRM call ───────────────────────────────────
    if n <= OSRM_MAX_WAYPOINTS:
        try:
            return _osrm_trip(pkgs, start=start, end=end)
        except Exception as e:
            log.warning(f'[osrm_full] single-call failed ({e}), falling back to NN')
            return _nn_sort(pkgs, start=start), 0, 0

    # ── Large route: chunked OSRM ─────────────────────────────────────
    # Pre-sort with nearest-neighbor so chunks are geographically contiguous
    pre_sorted   = _nn_sort(pkgs)
    chunk_size   = OSRM_MAX_WAYPOINTS
    chunks       = [pre_sorted[i:i + chunk_size] for i in range(0, n, chunk_size)]
    total_dist, total_dur = 0, 0
    final_order  = []

    for i, chunk in enumerate(chunks):
        try:
            chunk_start = start if i == 0 else None
            chunk_end = end if i == len(chunks) - 1 else None
            ordered_chunk, dist, dur = _osrm_trip(chunk, start=chunk_start, end=chunk_end)
            # If not first chunk, re-anchor: find which end of this chunk
            # is closest to the last stop of the previous chunk, reverse if needed
            if final_order:
                last = final_order[-1]
                if ordered_chunk and _dsq(last, ordered_chunk[-1]) < _dsq(last, ordered_chunk[0]):
                    ordered_chunk = list(reversed(ordered_chunk))
            final_order.extend(ordered_chunk)
            total_dist += dist
            total_dur  += dur
        except Exception as e:
            log.warning(f'[osrm_full] chunk {i} failed ({e}), using NN for chunk')
            final_order.extend(_nn_sort(chunk))

    return final_order, total_dist, total_dur


# Keep for any legacy callers
def osrm_optimize_segment(pkgs):
    return osrm_optimize_full_route(pkgs)


def build_optimized_route(geocoded, start=None, end=None):
    """
    Cluster-first, route-second optimization (zone sweep):
    1. Group packages at same building into one routable stop
    2. K-means geographic zones — compact block-level areas, never split
    3. Serpentine zone chain — finish one zone, move to the adjacent one (S-curve)
    4. Per-zone OSRM TSP, entering where the previous zone exited and
       exiting toward the next zone; same-street stops pulled back-to-back
    5. Zone letters follow drive order (A = first zone driven)
    6. Expand back to per-package list (same building = same zone, consecutive orders)

    Returns (sorted_pkgs, total_dist_m, total_dur_s)
    """
    if not geocoded:
        return [], 0, 0

    addresses = [p.get('address', '') for p in geocoded]
    confidence = get_address_zone_confidence(addresses)
    for p in geocoded:
        key = _normalize_addr_key(p.get('address', ''))
        info = confidence.get(key)
        p['_zone_confidence'] = info['confidence'] if info else 0.0
        p['_zone_count'] = info['count'] if info else 0
        p['_pinned_zone'] = info['zone'] if (info and info['pinned']) else None

    stop_groups = group_packages_by_stop(geocoded)
    routable = []
    for g in stop_groups:
        if not g.get('lat') or not g.get('lng'):
            continue
        routable.append({
            'id': g['packages'][0].get('id'),
            'lat': g['lat'],
            'lng': g['lng'],
            'address': g['address'],
            '_group': g,
        })

    if not routable:
        routable = [dict(p, _group={'packages': [p], 'addr_key': _building_group_key(p.get('address', '')),
                                    'lat': p.get('lat'), 'lng': p.get('lng'), 'address': p.get('address', '')})
                  for p in geocoded if p.get('lat') and p.get('lng')]

    # ── Geographic zones: k-means on stop coords (block-level clusters) ──
    k = calc_num_zones_adaptive(routable)
    if k <= 1 or len(routable) <= 2:
        zone_lists = [routable]
    else:
        assignments = kmeans_geo(routable, k)
        clusters = {}
        for stop_pt, ci in zip(routable, assignments):
            clusters.setdefault(ci, []).append(stop_pt)
        zone_lists = _serpentine_zone_order(
            [clusters[ci] for ci in sorted(clusters)], start=start, end=end
        )

    zone_cents = [
        {'lat': sum(s['lat'] for s in zl) / len(zl),
         'lng': sum(s['lng'] for s in zl) / len(zl)}
        for zl in zone_lists
    ]

    # ── Per-zone OSRM ordering, chained zone-to-zone (the S-sweep) ──
    ordered_stops = []
    total_dist, total_dur = 0, 0
    prev_pt = start
    for zi, zone_stops in enumerate(zone_lists):
        next_hint = zone_cents[zi + 1] if zi + 1 < len(zone_lists) else end
        try:
            if len(zone_stops) > OSRM_MAX_WAYPOINTS:
                ordered, dist, dur = osrm_optimize_full_route(zone_stops, start=prev_pt, end=next_hint)
            else:
                ordered, dist, dur = _osrm_trip(zone_stops, start=prev_pt, end=next_hint)
        except Exception as e:
            log.warning(f'[zone_sweep] zone {zi} OSRM failed ({e}), using NN')
            ordered, dist, dur = _nn_sort(zone_stops, start=prev_pt), 0, 0
        ordered = _group_same_street_runs(ordered)
        for s in ordered:
            s['_zone_seq'] = zi
        ordered_stops.extend(ordered)
        total_dist += dist
        total_dur += dur
        if ordered:
            prev_pt = {'lat': ordered[-1]['lat'], 'lng': ordered[-1]['lng']}

    n_stops = len(ordered_stops)
    result = []
    delivery_order = 1
    zone_local_count = {}
    for si, stop_pt in enumerate(ordered_stops):
        group = stop_pt.get('_group') or stop_pt
        pkgs = sorted(group.get('packages', [stop_pt]),
                      key=lambda x: x.get('scan_order') or x.get('id') or 0)
        zone_seq = stop_pt.get('_zone_seq', 0)
        letter = chr(65 + min(zone_seq, 25))
        zone_local_count[zone_seq] = zone_local_count.get(zone_seq, 0) + 1
        local_num = zone_local_count[zone_seq]
        color = ZONE_COLORS.get(letter, {'hex': '#6b7280', 'emoji': '⚪'})
        for p in pkgs:
            conf = p.get('_zone_confidence', 0.0)
            zone_count = p.get('_zone_count', 0)
            is_pinned = p.get('_pinned_zone') is not None
            if is_pinned:
                zone_status = 'locked'
            elif zone_count >= 1:
                zone_status = 'learning'
            else:
                zone_status = 'new'
            pkg = dict(p)
            pkg.update({
                'zone_letter': letter,
                'zone_num': local_num,
                'zone_label_full': f'{letter}-{local_num}',
                'zone_color': color['hex'],
                'zone_emoji': color['emoji'],
                'bag_num': math.ceil(local_num / BAG_SIZE),
                'bag_label': f'{letter}-Bag{math.ceil(local_num / BAG_SIZE)}',
                'delivery_order': delivery_order,
                'load_position': max(0, n_stops - si),
                'zone_confidence': round(conf * 100),
                'zone_status': zone_status,
                'zone_deliveries': zone_count,
                'stop_group_key': group.get('addr_key'),
                'packages_at_stop': len(pkgs),
            })
            result.append(pkg)
            delivery_order += 1

    # Ungeocoded packages not in routable — append at end
    routed_ids = {p.get('id') for p in result}
    for p in geocoded:
        if p.get('id') not in routed_ids and p.get('id') is not None:
            pkg = dict(p)
            pkg.update({
                'zone_letter': '?', 'zone_num': 0, 'zone_label_full': '?',
                'zone_color': '#6b7280', 'zone_emoji': '⚪',
                'delivery_order': delivery_order,
                'packages_at_stop': 1,
            })
            result.append(pkg)
            delivery_order += 1

    return result, total_dist, total_dur

def assign_delivery_zones(sorted_pkgs):
    """
    Legacy wrapper used by import-route and manual lock paths.
    Clusters pre-sorted packages into zones (no per-zone OSRM).
    """
    geocoded = [p for p in sorted_pkgs if p.get('lat') and p.get('lng')]
    ungeoced = [p for p in sorted_pkgs if not (p.get('lat') and p.get('lng'))]
    n = len(geocoded)
    if n == 0:
        _mark_unknown(ungeoced)
        return sorted_pkgs
    groups, cluster_letter = cluster_packages_geo(geocoded)
    result = []
    for cluster_idx in sorted(groups, key=lambda c: cluster_letter.get(c,'Z')):
        letter = cluster_letter.get(cluster_idx, 'A')
        color  = ZONE_COLORS.get(letter, {'hex':'#6b7280','emoji':'⚪'})
        for local_num, p in enumerate(
            sorted(groups[cluster_idx], key=lambda x: x.get('delivery_order', 0)), 1
        ):
            p.update({
                'zone_letter':    letter, 'zone_num':       local_num,
                'zone_label_full':f'{letter}-{local_num}',
                'zone_color':     color['hex'], 'zone_emoji': color['emoji'],
                'bag_num':        math.ceil(local_num / BAG_SIZE),
                'bag_label':      f'{letter}-Bag{math.ceil(local_num / BAG_SIZE)}',
            })
            result.append(p)
    _mark_unknown(ungeoced)
    return result + ungeoced

def assign_vehicle_zones(sorted_pkgs, vehicle_type):
    """
    Assign vehicle cargo zone to each package based on its delivery zone letter.
    Zone A (first delivery cluster) → near door (load last).
    Zone C/D (last cluster) → deepest (load first).
    All packages in the same delivery zone go to the same vehicle spot.
    """
    v_zones = VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv_midsize'])
    v_count = len(v_zones)
    if not sorted_pkgs:
        return sorted_pkgs

    # Get unique delivery zone letters in order
    unique_letters = []
    seen_l = set()
    for p in sorted_pkgs:
        letter = p.get('zone_letter', '?')
        if letter != '?' and letter not in seen_l:
            seen_l.add(letter)
            unique_letters.append(letter)
    unique_letters.sort()   # A, B, C…

    n_dlv_zones = len(unique_letters)

    # Map each delivery zone letter → vehicle cargo zone
    # Zone A (first delivery) → last vehicle zone (near door)
    # Zone Z (last delivery) → first vehicle zone (deepest)
    letter_to_vzone = {}
    for seq, letter in enumerate(unique_letters):
        if n_dlv_zones > 1:
            v_idx = int(round(seq / (n_dlv_zones - 1) * (v_count - 1)))
        else:
            v_idx = 0
        v_idx = v_count - 1 - v_idx   # invert: seq=0 (Zone A) → last v_zone (door)
        v_idx = max(0, min(v_idx, v_count - 1))
        letter_to_vzone[letter] = v_zones[v_idx]

    for p in sorted_pkgs:
        letter = p.get('zone_letter', '?')
        vz = letter_to_vzone.get(letter, v_zones[-1])
        p.update({'vehicle_zone_id': vz['id'], 'vehicle_zone_label': vz['label'],
                  'vehicle_zone_icon': vz['icon'], 'vehicle_zone_desc': vz['desc']})
    return sorted_pkgs
# ── ZONE AUTO-LOCK HELPERS ────────────────────────────────────────
ZONE_LOCK_THRESHOLD = 20   # packages before auto-lock triggers
ZONE_STABLE_MILES   = 0.15 # centroid must move < this between scans to be "stable"

def compute_centroids(sorted_pkgs):
    """Extract zone centroids (avg lat/lng per zone letter) from assigned packages."""
    zone_pts = {}
    for p in sorted_pkgs:
        letter = p.get('zone_letter')
        if letter and letter != '?' and p.get('lat') and p.get('lng'):
            zone_pts.setdefault(letter, []).append(p)
    return [
        {
            'letter': letter,
            'lat':    sum(p['lat'] for p in pts) / len(pts),
            'lng':    sum(p['lng'] for p in pts) / len(pts),
            'color':  pts[0].get('zone_color',  '#3b82f6'),
            'emoji':  pts[0].get('zone_emoji',  '⚪'),
        }
        for letter, pts in sorted(zone_pts.items())
    ]


def reconstruct_scan_pkg_meta(items, vehicle_type='suv_midsize'):
    """Rebuild zone colors, nums, and vehicle spots from locked scan_items."""
    pkgs = []
    for item in items:
        pkgs.append({
            'id': item['id'],
            'tracking': _ss_val(item, 'tracking', ''),
            'name': _ss_val(item, 'customer_name', ''),
            'address': _ss_val(item, 'address', ''),
            'lat': _ss_val(item, 'dest_lat'),
            'lng': _ss_val(item, 'dest_lng'),
            'zone_letter': _ss_val(item, 'zone_letter', '?'),
            'delivery_order': _ss_val(item, 'delivery_order', 0),
            'scan_order': _ss_val(item, 'scan_order', 0),
        })
    zone_counters = {}
    for p in pkgs:
        letter = p.get('zone_letter') or '?'
        zone_counters[letter] = zone_counters.get(letter, 0) + 1
        p['zone_num'] = zone_counters[letter]
        ci = ZONE_COLORS.get(letter, {'hex': '#6b7280', 'emoji': '⚪'})
        p['zone_color'] = ci['hex']
        p['zone_emoji'] = ci['emoji']
        p['zone_label_full'] = f"{letter}-{p['zone_num']}" if letter != '?' else '?'
    return assign_vehicle_zones(pkgs, vehicle_type)


def build_route_zone_context(stops, route=None, vehicle_type='suv_midsize', db=None):
    """
    Enrich delivery stops with zone/sticker metadata for dashboard + maps.
    Returns (stops_list, zone_summary, zone_centroids, current_zone, next_stop).
    """
    stops_list = [dict(s) for s in stops] if stops else []

    if db:
        for s in stops_list:
            if s.get('tracking') and (not s.get('scan_order') or not s.get('zone_letter')):
                item = db.execute(
                    "SELECT scan_order, zone_letter FROM scan_items WHERE tracking=? ORDER BY id DESC LIMIT 1",
                    (s['tracking'],)
                ).fetchone()
                if item:
                    if not s.get('scan_order'):
                        s['scan_order'] = _ss_val(item, 'scan_order')
                    if not s.get('zone_letter'):
                        s['zone_letter'] = _ss_val(item, 'zone_letter')

    zone_counters = {}
    unique_letters = []
    v_zones = VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv_midsize'])
    for s in stops_list:
        letter = s.get('zone_letter') or '?'
        if letter not in zone_counters:
            zone_counters[letter] = 0
            if letter != '?':
                unique_letters.append(letter)
        if not s.get('zone_num'):
            zone_counters[letter] += 1
            s['zone_num'] = zone_counters[letter]
        ci = ZONE_COLORS.get(letter, {'hex': '#6b7280', 'emoji': '⚪'})
        s['zone_color'] = s.get('zone_color') or ci['hex']
        s['zone_emoji'] = s.get('zone_emoji') or ci['emoji']
        s['zone_label_full'] = f"{letter}-{s.get('zone_num', 1)}" if letter != '?' else '?'
        s['scan_order'] = s.get('scan_order') or s.get('stop_number', 0)
        if not s.get('vehicle_zone_label') and letter in unique_letters:
            idx = unique_letters.index(letter)
            vz = v_zones[min(idx, len(v_zones) - 1)]
            s['vehicle_zone_label'] = vz.get('label', '')

    zone_summary = {}
    for s in stops_list:
        letter = s.get('zone_letter') or '?'
        if letter == '?':
            continue
        if letter not in zone_summary:
            idx = unique_letters.index(letter) if letter in unique_letters else 0
            vz = v_zones[min(idx, len(v_zones) - 1)]
            zone_summary[letter] = {
                'letter': letter,
                'color': s['zone_color'],
                'emoji': s['zone_emoji'],
                'count': 0,
                'delivered': 0,
                'pending': 0,
                'vehicle_spot': s.get('vehicle_zone_label') or vz.get('label', ''),
            }
        zone_summary[letter]['count'] += 1
        if s.get('status') == 'delivered':
            zone_summary[letter]['delivered'] += 1
        elif s.get('status') in ('pending', 'en_route'):
            zone_summary[letter]['pending'] += 1

    zone_summary_list = [zone_summary[l] for l in sorted(zone_summary.keys())]

    zone_centroids = []
    if route and _ss_val(route, 'zone_centroids'):
        try:
            zone_centroids = json.loads(route['zone_centroids'])
        except Exception:
            pass
    if not zone_centroids:
        zone_centroids = compute_centroids([
            {
                'zone_letter': s.get('zone_letter'),
                'lat': s.get('dest_lat'),
                'lng': s.get('dest_lng'),
                'zone_color': s.get('zone_color'),
                'zone_emoji': s.get('zone_emoji'),
            }
            for s in stops_list if s.get('dest_lat')
        ])

    current_zone = None
    next_stop = None
    for s in stops_list:
        if s.get('status') in ('pending', 'en_route'):
            if not next_stop:
                next_stop = s
            if not current_zone and s.get('zone_letter'):
                current_zone = s.get('zone_letter')

    return stops_list, zone_summary_list, zone_centroids, current_zone, next_stop


def centroids_stable(old_c, new_c):
    """True when all zone centroids moved < ZONE_STABLE_MILES since last scan."""
    if not old_c or len(old_c) != len(new_c):
        return False
    old_map = {c['letter']: c for c in old_c}
    new_map = {c['letter']: c for c in new_c}
    if set(old_map) != set(new_map):
        return False
    for letter, oc in old_map.items():
        nc = new_map[letter]
        if geodesic((oc['lat'], oc['lng']), (nc['lat'], nc['lng'])).miles > ZONE_STABLE_MILES:
            return False
    return True

def assign_zones_from_centroids(sorted_pkgs, centroids):
    """
    Fast locked-zone assignment: snap each package to its nearest centroid.
    No re-clustering. Zones never change after lock.
    """
    for p in sorted_pkgs:
        if p.get('lat') and p.get('lng'):
            nearest = min(centroids, key=lambda c: _dsq(p, c))
            p['zone_letter'] = nearest['letter']
            p['zone_color']  = nearest['color']
            p['zone_emoji']  = nearest['emoji']
        else:
            p['zone_letter'] = '?'
            p['zone_color']  = '#6b7280'
            p['zone_emoji']  = '⚪'

    # Number packages within each zone by delivery_order
    zone_groups = {}
    for p in sorted_pkgs:
        zone_groups.setdefault(p['zone_letter'], []).append(p)

    for letter, grp in zone_groups.items():
        for num, p in enumerate(
            sorted(grp, key=lambda x: x.get('delivery_order', 0)), 1
        ):
            p['zone_num']        = num
            p['zone_label_full'] = f'{letter}-{num}' if letter != '?' else '?'

    return sorted_pkgs


TWILIO_SID   = os.environ.get('TWILIO_SID', '')
TWILIO_TOKEN = os.environ.get('TWILIO_TOKEN', '')
TWILIO_PHONE = os.environ.get('TWILIO_PHONE', '')
STRIPE_SECRET      = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUB_KEY     = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
STRIPE_PRICE_ID    = os.environ.get('STRIPE_PRICE_ID', 'price_1TZeWUEQpiT0nKEdHs158Phk')
stripe.api_key     = STRIPE_SECRET
APPROACH_RADIUS_MILES = 0.5
GEOFENCE_RADIUS_MILES  = 0.028
_geocache = {}   # in-memory: address -> (lat, lng)  — pre-seeded from address_intel on startup

# ─── ADDRESS INTELLIGENCE ─────────────────────────────────────

def _table_exists(db, name):
    """Check if a SQLite table exists (safe for migrations)."""
    try:
        return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None
    except Exception:
        return False

def _normalize_addr_key(address):
    """Consistent lookup key for address_intel."""
    return re.sub(r'\s+', ' ', address.strip().upper())

def _extract_zip(address):
    m = re.search(r'\b(\d{5})\b', address)
    return m.group(1) if m else None

def upsert_address_intel(address, lat, lng, zone_letter=None):
    """
    Persist a geocoded address to the address_intel table.
    Also updates _geocache immediately so the current session benefits.
    """
    if not address or not lat or not lng:
        return
    key      = _normalize_addr_key(address)
    zip_code = _extract_zip(address)
    now      = datetime.now().isoformat()
    _geocache[key]     = (lat, lng)
    _geocache[address] = (lat, lng)
    try:
        db       = get_db()
        existing = db.execute("SELECT id, zone_history FROM address_intel WHERE address=?", (key,)).fetchone()
        if existing:
            try:
                hist = json.loads(existing['zone_history'] or '[]')
            except Exception:
                hist = []
            if zone_letter and zone_letter not in ('?', None):
                hist.append(zone_letter)
                hist = hist[-30:]
            db.execute(
                """UPDATE address_intel
                   SET lat=?, lng=?, zip_code=COALESCE(?,zip_code), zone_history=?, updated_at=?
                   WHERE address=?""",
                (lat, lng, zip_code, json.dumps(hist), now, key)
            )
        else:
            zone_history = json.dumps([zone_letter]) if zone_letter and zone_letter != '?' else '[]'
            db.execute(
                """INSERT INTO address_intel (address, lat, lng, zip_code, zone_history, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (key, lat, lng, zip_code, zone_history, now, now)
            )
        db.commit()
        db.close()
    except Exception as e:
        log.warning(f'[address_intel] upsert error: {e}')

def record_address_delivery(address, zone_letter=None):
    """
    Increment delivery count and record zone assignment for a confirmed delivery.
    Builds the zone intelligence used for future route clustering.
    """
    if not address:
        return
    key = _normalize_addr_key(address)
    now = datetime.now().isoformat()
    try:
        db  = get_db()
        row = db.execute(
            "SELECT delivery_count, zone_history FROM address_intel WHERE address=?", (key,)
        ).fetchone()
        if row:
            count = (row['delivery_count'] or 0) + 1
            try:
                hist = json.loads(row['zone_history'] or '[]')
            except Exception:
                hist = []
            if zone_letter and zone_letter not in ('?', None):
                hist.append(zone_letter)
                hist = hist[-30:]
            db.execute(
                """UPDATE address_intel
                   SET delivery_count=?, zone_history=?, last_delivered=?, updated_at=?
                   WHERE address=?""",
                (count, json.dumps(hist), now, now, key)
            )
            db.commit()
        db.close()
    except Exception as e:
        log.warning(f'[address_intel] record_delivery error: {e}')

_UNIT_LABEL_RE = re.compile(
    r'(?:\b(?:apt|apartment|unit|suite|ste|fl|floor|rm|room|bldg|building|no)\b\.?|#)\s*#?\s*([A-Za-z]?\d+[A-Za-z]?(?:-\d+)?|\d*[A-Za-z])\b',
    re.I,
)
# Bare number between street and city: "123 Main St, 4B, Detroit MI" or "1310 Pallister St 807"
_UNIT_TRAILING_RE = re.compile(
    r'\b(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Blvd|Boulevard|Ln|Lane|Ct|Court|Pl|Place|Way|Pkwy|Hwy|Ter|Trl|Cir)\.?\s*,?\s+(\d{1,4}[A-Za-z]?)\s*(?:,|$)',
    re.I,
)


def extract_unit_number(address):
    """Pull the apt/unit identifier out of a full address string.

    Returns the unit string ('4B', '206', ...) or ''. The address itself is
    left untouched — geocoding and grouping already handle unit suffixes.
    """
    if not address:
        return ''
    m = _UNIT_LABEL_RE.search(address)
    if m and not re.fullmatch(r'\d{5}', m.group(1)):  # 'FL 33101' is a state+zip, not Floor 33101
        return m.group(1).upper()
    m = _UNIT_TRAILING_RE.search(address)
    if m and not re.fullmatch(r'\d{5}', m.group(1)):
        return m.group(1).upper()
    return ''


def get_known_units(address):
    """Return list of unit numbers previously confirmed at this street address."""
    if not address:
        return []
    key = _normalize_addr_key(_street_base(address))
    if not key:
        return []
    try:
        db = get_db()
        row = db.execute(
            "SELECT known_units FROM address_intel WHERE address=? OR address LIKE ? LIMIT 1",
            (key, key + '%'),
        ).fetchone()
        db.close()
        if row and row['known_units']:
            units = json.loads(row['known_units'])
            return [str(u) for u in units if u][:12]
    except Exception as e:
        log.warning(f'[building_memory] get_known_units error: {e}')
    return []


def is_known_multi_unit(address):
    """True if this street address is a known multi-unit building."""
    if not address:
        return False
    base = _street_base(address)
    if not base:
        return False
    if len(get_known_units(address)) >= 1:
        return True
    try:
        db = get_db()
        hit = db.execute(
            "SELECT 1 FROM buildings WHERE UPPER(address) LIKE ? LIMIT 1",
            (base + '%',),
        ).fetchone()
        db.close()
        if hit:
            return True
    except Exception:
        pass
    return bool(re.search(r'\b(apt|apartment|unit|suite|ste|bldg|building|tower|plaza|lofts?|manor|terrace apartments)\b', address, re.I))


def remember_unit_number(address, unit):
    """Append a confirmed unit number to address_intel.known_units (building memory)."""
    if not address or not unit:
        return
    unit = str(unit).strip().upper()
    base_key = _normalize_addr_key(_street_base(address))
    full_key = _normalize_addr_key(address)
    try:
        db = get_db()
        row = db.execute(
            "SELECT id, known_units FROM address_intel WHERE address IN (?,?) LIMIT 1",
            (base_key, full_key),
        ).fetchone()
        now = datetime.now().isoformat()
        if row:
            try:
                units = json.loads(row['known_units'] or '[]')
            except Exception:
                units = []
            if unit not in units:
                units.append(unit)
                db.execute(
                    "UPDATE address_intel SET known_units=?, updated_at=? WHERE id=?",
                    (json.dumps(units[-50:]), now, row['id']),
                )
                db.commit()
        db.close()
    except Exception as e:
        log.warning(f'[building_memory] remember_unit error: {e}')


def get_historical_centroids_for_zips(zip_codes):
    """
    Pull all historically-delivered lat/lng points for a set of zip codes.
    Returns [{lat, lng}, ...] — used as k-means seed candidates for new routes.
    Only includes addresses actually delivered at least once (verified ground truth).
    """
    if not zip_codes:
        return []
    try:
        db  = get_db()
        ph  = ','.join('?' * len(zip_codes))
        rows = db.execute(
            f"SELECT lat, lng FROM address_intel WHERE zip_code IN ({ph}) AND delivery_count > 0",
            list(zip_codes)
        ).fetchall()
        db.close()
        return [{'lat': r['lat'], 'lng': r['lng']} for r in rows if r['lat'] and r['lng']]
    except Exception as e:
        log.warning(f'[address_intel] centroid query error: {e}')
        return []

def get_address_zone_confidence(addresses):
    """
    For a list of raw address strings, return a confidence map:
      { normalized_addr: { 'zone': 'B', 'confidence': 0.85, 'count': 6,
                           'pinned': True, 'lat': x, 'lng': y } }

    Pinned = delivered 3+ times AND top zone appears >= 60% of the time.
    These addresses act as anchors for the clustering step —
    they don't move; floating addresses cluster around them.
    """
    if not addresses:
        return {}
    from collections import Counter
    keys = [_normalize_addr_key(a) for a in addresses if a]
    if not keys:
        return {}
    try:
        db  = get_db()
        ph  = ','.join('?' * len(keys))
        rows = db.execute(
            f"""SELECT address, lat, lng, delivery_count, zone_history
                FROM address_intel WHERE address IN ({ph})""",
            keys
        ).fetchall()
        db.close()
    except Exception as e:
        log.warning(f'[zone_confidence] query error: {e}')
        return {}

    result = {}
    for r in rows:
        try:
            hist  = json.loads(r['zone_history'] or '[]')
        except Exception:
            hist  = []
        count = r['delivery_count'] or 0
        if not hist:
            continue
        counter      = Counter(hist)
        top_zone, top_freq = counter.most_common(1)[0]
        confidence   = top_freq / len(hist)
        pinned       = count >= 3 and confidence >= 0.60
        result[r['address']] = {
            'zone':       top_zone,
            'confidence': round(confidence, 2),
            'count':      count,
            'pinned':     pinned,
            'lat':        r['lat'],
            'lng':        r['lng'],
        }
    return result


def _balance_zones(groups, max_ratio=1.5):
    """
    After k-means, redistribute packages from over-full zones to under-full ones.
    A zone is over-full if it has more than (avg * max_ratio) packages.
    Moves border packages (furthest from their zone centroid) to the nearest
    under-full zone. Produces more even load distribution.
    """
    if len(groups) <= 1:
        return groups

    total = sum(len(v) for v in groups.values())
    avg   = total / len(groups)
    target_max = math.ceil(avg * max_ratio)

    # Compute current centroids
    def centroid(pkgs):
        if not pkgs: return {'lat': 0, 'lng': 0}
        return {'lat': sum(p['lat'] for p in pkgs) / len(pkgs),
                'lng': sum(p['lng'] for p in pkgs) / len(pkgs)}

    max_passes = 5
    for _ in range(max_passes):
        changed = False
        centroids = {k: centroid(v) for k, v in groups.items()}
        over_full = [k for k, v in groups.items() if len(v) > target_max]
        if not over_full:
            break
        for big_k in over_full:
            pkgs  = groups[big_k]
            c     = centroids[big_k]
            # Sort by distance from centroid desc (border packages first)
            pkgs.sort(key=lambda p: _dsq(p, c), reverse=True)
            under_keys = [k for k, v in groups.items() if k != big_k and len(v) < avg]
            if not under_keys:
                break
            while len(groups[big_k]) > target_max and under_keys:
                pkg = groups[big_k][0]  # furthest from centroid
                # Find nearest under-full zone
                target_k = min(under_keys, key=lambda k: _dsq(pkg, centroids[k]))
                groups[big_k].remove(pkg)
                groups[target_k].append(pkg)
                changed = True
                if len(groups[target_k]) >= avg:
                    under_keys = [k for k, v in groups.items() if k != big_k and len(v) < avg]
        if not changed:
            break
    return groups


def nearest_intel_zone(lat, lng, radius_miles=0.25):
    """
    Given a lat/lng, return the most common historical zone letter among
    nearby addresses in address_intel (within radius_miles, min 3 deliveries total).
    Returns zone_letter string or None.
    """
    try:
        db   = get_db()
        rows = db.execute(
            "SELECT lat, lng, zone_history FROM address_intel WHERE delivery_count >= 2"
        ).fetchall()
        db.close()
        nearby_zones = []
        for r in rows:
            if r['lat'] and r['lng']:
                if geodesic((lat, lng), (r['lat'], r['lng'])).miles <= radius_miles:
                    try:
                        hist = json.loads(r['zone_history'] or '[]')
                        nearby_zones.extend(hist)
                    except Exception:
                        pass
        if len(nearby_zones) >= 2:
            from collections import Counter
            return Counter(nearby_zones).most_common(1)[0][0]
    except Exception as e:
        log.warning(f'[address_intel] nearest_intel_zone error: {e}')
    return None

# ─── DB ────────────────────────────────────────────────────────

# ── PostgreSQL connection pool (one pool per worker process) ──
_pg_pool      = None
_pg_pool_lock = None

def _get_pg_pool():
    """Lazy-init a simple per-worker connection pool for PostgreSQL."""
    global _pg_pool, _pg_pool_lock
    import threading
    if _pg_pool_lock is None:
        _pg_pool_lock = threading.Lock()
    with _pg_pool_lock:
        if _pg_pool is None:
            import urllib.parse
            url = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
            p   = urllib.parse.urlparse(url)
            # Build a small pool: min 1, max 5 connections per worker
            _pg_pool = {
                'host':     p.hostname,
                'port':     p.port or 5432,
                'database': p.path.lstrip('/'),
                'user':     p.username,
                'password': p.password,
            }
    return _pg_pool

def get_db():
    if USE_PG:
        cfg  = _get_pg_pool()
        retry = 0
        while retry < 3:
            try:
                conn = pg8000.connect(
                    host=cfg['host'], port=cfg['port'],
                    database=cfg['database'], user=cfg['user'],
                    password=cfg['password'], ssl_context=True,
                    timeout=10
                )
                conn.autocommit = False
                return DBWrapper(conn, pg=True)
            except Exception as e:
                retry += 1
                if retry >= 3:
                    log.error(f'DB connect failed after 3 retries: {e}')
                    raise
                import time as _t; _t.sleep(0.5 * retry)
    else:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect(DB, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=-8000')   # 8MB page cache
        conn.execute('PRAGMA temp_store=MEMORY')
        return DBWrapper(conn, pg=False)

def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS drivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            company TEXT,
            pin TEXT,
            current_lat REAL,
            current_lng REAL,
            last_seen TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER,
            driver_name TEXT,
            name TEXT,
            date TEXT,
            blast_sent INTEGER DEFAULT 0,
            blast_sent_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS stops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id INTEGER,
            stop_number INTEGER,
            address TEXT,
            unit TEXT,
            customer_name TEXT,
            phone TEXT,
            tracking TEXT,
            notes TEXT,
            drop_spot TEXT,
            dest_lat REAL,
            dest_lng REAL,
            driver_lat REAL,
            driver_lng REAL,
            status TEXT DEFAULT 'pending',
            sms_blast_sent INTEGER DEFAULT 0,
            approach_sms_sent INTEGER DEFAULT 0,
            token TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS buildings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT UNIQUE NOT NULL,
            access_code TEXT,
            buzzer_notes TEXT,
            interior_directions TEXT,
            access_type TEXT DEFAULT 'code',
            lat REAL, lng REAL,
            confirmed_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS pin_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT UNIQUE NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            corrected_by TEXT,
            corrected_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS residents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            unit TEXT NOT NULL,
            phone TEXT NOT NULL,
            backup_phone TEXT,
            drop_spot TEXT,
            door_notes TEXT,
            sms_consent INTEGER DEFAULT 0,
            sms_consent_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS driver_onboarding (
            driver_id INTEGER PRIMARY KEY,
            completed_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip TEXT NOT NULL,
            attempted_at TEXT NOT NULL
        );
    ''')
    db.commit()
    # Only insert default driver if NO drivers exist yet
    try:
        count = db.execute("SELECT COUNT(*) FROM drivers").fetchone()[0]
        if count == 0:
            init_pin = str(secrets.randbelow(9000) + 1000)
            db.execute("INSERT INTO drivers (name, phone, company, pin) VALUES (?,?,?,?)",
                       ('Director X', '3135550000', 'SpeedX', init_pin))
            db.commit()
            log.info(f'Default driver created with PIN: {init_pin}')
    except: pass
    # Safe migrations — add columns if they don't exist yet
    for migration in [
        "ALTER TABLE stops ADD COLUMN drop_spot TEXT",
        "ALTER TABLE residents ADD COLUMN sms_consent INTEGER DEFAULT 0",
        "ALTER TABLE residents ADD COLUMN sms_consent_at TEXT",
        "ALTER TABLE residents ADD COLUMN customer_name TEXT",
        "ALTER TABLE drivers ADD COLUMN onboarded INTEGER DEFAULT 0",
        "ALTER TABLE drivers ADD COLUMN is_beta INTEGER DEFAULT 0",
        "ALTER TABLE drivers ADD COLUMN vehicle_type TEXT DEFAULT 'suv_midsize'",
        "ALTER TABLE drivers ADD COLUMN vehicle_capacity INTEGER DEFAULT 100",
        "ALTER TABLE drivers ADD COLUMN assigned_zips TEXT",
        "ALTER TABLE drivers ADD COLUMN pay_rate REAL DEFAULT 1.50",
        "ALTER TABLE routes ADD COLUMN route_type TEXT DEFAULT 'standard'",
        "ALTER TABLE scan_sessions ADD COLUMN zones_locked INTEGER DEFAULT 0",
        "ALTER TABLE scan_sessions ADD COLUMN zone_centroids TEXT",
        "ALTER TABLE scan_sessions ADD COLUMN locked_at TEXT",
        "ALTER TABLE scan_sessions ADD COLUMN phase TEXT DEFAULT 'scanning'",
        "ALTER TABLE scan_sessions ADD COLUMN prev_centroids TEXT",
        "ALTER TABLE scan_items ADD COLUMN zone_letter TEXT",
        "ALTER TABLE scan_items ADD COLUMN delivery_order INTEGER",
        "ALTER TABLE scan_items ADD COLUMN scan_order INTEGER",
        "ALTER TABLE stops ADD COLUMN zone_letter TEXT",
        "ALTER TABLE stops ADD COLUMN delivered_at TEXT",
        "ALTER TABLE routes ADD COLUMN est_distance_miles REAL",
        "ALTER TABLE routes ADD COLUMN est_duration_mins REAL",
        "ALTER TABLE routes ADD COLUMN route_started_at TEXT",
        "ALTER TABLE routes ADD COLUMN first_delivery_at TEXT",
        "ALTER TABLE routes ADD COLUMN zone_centroids TEXT",
        "ALTER TABLE stops ADD COLUMN scan_order INTEGER",
        "ALTER TABLE stops ADD COLUMN zone_num INTEGER",
        "ALTER TABLE stops ADD COLUMN zone_color TEXT",
        "ALTER TABLE stops ADD COLUMN zone_emoji TEXT",
        "ALTER TABLE stops ADD COLUMN vehicle_zone_label TEXT",
        "ALTER TABLE scan_sessions ADD COLUMN route_start_lat REAL",
        "ALTER TABLE scan_sessions ADD COLUMN route_start_lng REAL",
        "ALTER TABLE scan_sessions ADD COLUMN route_end_lat REAL",
        "ALTER TABLE scan_sessions ADD COLUMN route_end_lng REAL",
        "ALTER TABLE scan_sessions ADD COLUMN route_end_mode TEXT",
        "ALTER TABLE stops ADD COLUMN package_count INTEGER DEFAULT 1",
        "ALTER TABLE stops ADD COLUMN package_list TEXT",
        # ── Unit number extraction + building memory ──
        "ALTER TABLE scan_items ADD COLUMN unit TEXT",
        "ALTER TABLE address_intel ADD COLUMN known_units TEXT DEFAULT '[]'",
        # ── QR Building Access feature ──
        "ALTER TABLE buildings ADD COLUMN building_code TEXT",
        "ALTER TABLE buildings ADD COLUMN name TEXT",
        "ALTER TABLE buildings ADD COLUMN general_access_code TEXT",
        "ALTER TABLE buildings ADD COLUMN package_room_notes TEXT",
        "ALTER TABLE buildings ADD COLUMN lockbox_notes TEXT",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_buildings_code ON buildings (building_code)",
        """CREATE TABLE IF NOT EXISTS delivery_instructions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            building_id INTEGER NOT NULL,
            unit_number TEXT NOT NULL,
            customer_notes TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_delivery_instructions_building ON delivery_instructions (building_id)",
        "CREATE TABLE IF NOT EXISTS pin_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT UNIQUE NOT NULL, lat REAL NOT NULL, lng REAL NOT NULL, corrected_by TEXT, corrected_at TEXT DEFAULT CURRENT_TIMESTAMP)",
        "CREATE TABLE IF NOT EXISTS login_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, ip TEXT NOT NULL, attempted_at TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts (ip, attempted_at)",
        # ── Performance indexes for scale ──
        "CREATE INDEX IF NOT EXISTS idx_stops_route_id ON stops (route_id)",
        "CREATE INDEX IF NOT EXISTS idx_stops_status ON stops (status)",
        "CREATE INDEX IF NOT EXISTS idx_stops_route_status ON stops (route_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_stops_delivered_at ON stops (delivered_at)",
        "CREATE INDEX IF NOT EXISTS idx_routes_driver_id ON routes (driver_id)",
        "CREATE INDEX IF NOT EXISTS idx_routes_driver_date ON routes (driver_id, date)",
        "CREATE INDEX IF NOT EXISTS idx_routes_date ON routes (date)",
        "CREATE INDEX IF NOT EXISTS idx_drivers_pin ON drivers (pin)",
        "CREATE INDEX IF NOT EXISTS idx_scan_sessions_driver_date ON scan_sessions (driver_id, date, status)",
        "CREATE INDEX IF NOT EXISTS idx_scan_items_session ON scan_items (session_id)",
        "CREATE INDEX IF NOT EXISTS idx_stops_token ON stops (token)",
        "CREATE INDEX IF NOT EXISTS idx_live_sessions_token ON live_sessions (token)",
        "CREATE INDEX IF NOT EXISTS idx_residents_address ON residents (address)",
        "CREATE INDEX IF NOT EXISTS idx_buildings_address ON buildings (address)",
        "CREATE INDEX IF NOT EXISTS idx_pin_corrections_address ON pin_corrections (address)",
        """CREATE TABLE IF NOT EXISTS live_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            driver_id INTEGER,
            driver_name TEXT,
            driver_lat REAL,
            driver_lng REAL,
            last_seen TEXT,
            status TEXT DEFAULT 'active',
            viewed_at TEXT,
            view_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "ALTER TABLE live_sessions ADD COLUMN viewed_at TEXT",
        "ALTER TABLE live_sessions ADD COLUMN view_count INTEGER DEFAULT 0",
        """CREATE TABLE IF NOT EXISTS scan_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT DEFAULT 'scanning',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS scan_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            tracking TEXT,
            customer_name TEXT,
            address TEXT,
            zip_code TEXT,
            dest_lat REAL,
            dest_lng REAL,
            raw_json TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        # ── Address Intelligence: persistent spatial memory ──
        """CREATE TABLE IF NOT EXISTS address_intel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT UNIQUE NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            zip_code TEXT,
            delivery_count INTEGER DEFAULT 0,
            last_delivered TEXT,
            zone_history TEXT DEFAULT '[]',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS label_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking TEXT UNIQUE NOT NULL,
            customer_name TEXT,
            address TEXT,
            zip_code TEXT,
            read_count INTEGER DEFAULT 1,
            last_source TEXT DEFAULT 'confirm',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_label_memory_tracking ON label_memory (tracking)",
        "CREATE INDEX IF NOT EXISTS idx_address_intel_zip ON address_intel (zip_code)",
        "CREATE INDEX IF NOT EXISTS idx_address_intel_latng ON address_intel (lat, lng)",
        """CREATE TABLE IF NOT EXISTS route_manual_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            packages INTEGER DEFAULT 0,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        # ── Speed X POD (proof of delivery) ──
        "ALTER TABLE stops ADD COLUMN pod_photo_1 TEXT",
        "ALTER TABLE stops ADD COLUMN pod_photo_2 TEXT",
        "ALTER TABLE stops ADD COLUMN pod_photo_3 TEXT",
        "ALTER TABLE stops ADD COLUMN pod_captured_at TEXT",
        # ── Payroll (manager operations layer) ──
        """CREATE TABLE IF NOT EXISTS payroll_days (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            work_date TEXT NOT NULL,
            stops INTEGER DEFAULT 0,
            rate_per_stop REAL DEFAULT 0,
            area TEXT,
            source TEXT DEFAULT 'manual',
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_payroll_days_driver_date ON payroll_days (driver_id, work_date)",
        """CREATE TABLE IF NOT EXISTS payroll_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            work_date TEXT,
            kind TEXT DEFAULT 'claim',
            amount REAL DEFAULT 0,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_payroll_adjustments_driver_date ON payroll_adjustments (driver_id, work_date)",
        # Per-driver default stop rate so payroll prefills sensibly
        "ALTER TABLE drivers ADD COLUMN default_rate REAL DEFAULT 0",
        # ── Companies & managers (multi-tenant ops portal) ──
        """CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS managers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            phone TEXT,
            pin TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_managers_pin ON managers (pin)",
        "ALTER TABLE drivers ADD COLUMN company_id INTEGER",
        "CREATE INDEX IF NOT EXISTS idx_drivers_company ON drivers (company_id)",
        """CREATE TABLE IF NOT EXISTS driver_checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER NOT NULL,
            check_date TEXT NOT NULL,
            status TEXT DEFAULT 'unknown',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(driver_id, check_date)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_checkins_date ON driver_checkins (check_date)",
        """CREATE TABLE IF NOT EXISTS manager_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            body TEXT NOT NULL,
            sent_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_manager_messages_company ON manager_messages (company_id)",
        "ALTER TABLE driver_checkins ADD COLUMN assignment TEXT",
    ]:
        try:
            db.execute(migration)
            db.commit()
        except:
            try: db._conn.rollback()
            except: pass

    # ── Seed in-memory geocache from address_intel on startup ──
    try:
        rows = db.execute("SELECT address, lat, lng FROM address_intel WHERE lat IS NOT NULL AND lng IS NOT NULL").fetchall()
        for r in rows:
            _geocache[r['address']] = (r['lat'], r['lng'])
        if rows:
            log.info(f'[address_intel] Seeded geocache with {len(rows)} known addresses')
    except Exception as e:
        log.warning(f'[address_intel] Failed to seed geocache: {e}')

    _seed_companies_and_managers(db)
    db.close()

def _seed_companies_and_managers(db):
    """Bootstrap Rolling Logistics + default manager; link existing drivers."""
    try:
        if not _table_exists(db, 'companies'):
            return
        row = db.execute("SELECT id FROM companies WHERE slug = ?", ('rolling-logistics',)).fetchone()
        if not row:
            db.execute("INSERT INTO companies (name, slug) VALUES (?, ?)",
                       ('Rolling Logistics', 'rolling-logistics'))
            db.commit()
            cid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            mgr_pin = os.environ.get('MANAGER_PIN', '5678')
            db.execute(
                "INSERT INTO managers (company_id, name, pin) VALUES (?, ?, ?)",
                (cid, 'Rolling Logistics Manager', mgr_pin)
            )
            db.commit()
            log.info(f'[manager] Created Rolling Logistics (id={cid}), manager PIN={mgr_pin}')
        else:
            cid = row['id']
        # Only auto-link drivers whose company text mentions Rolling.
        # Everyone else is curated by the manager via the roster page —
        # this avoids sweeping legacy/test drivers into the live team.
        db.execute(
            """UPDATE drivers SET company_id = ?
               WHERE company_id IS NULL
                 AND LOWER(COALESCE(company, '')) LIKE '%rolling%'""",
            (cid,)
        )
        db.commit()
    except Exception as e:
        log.warning(f'[manager] Seed failed: {e}')
        try: db._conn.rollback()
        except: pass

# ─── HELPERS ───────────────────────────────────────────────────

_WORD_TO_NUM = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10','eleven':'11',
    'twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15',
    'sixteen':'16','seventeen':'17','eighteen':'18','nineteen':'19','twenty':'20',
}
_WORD_NUM_RE = re.compile(r'\b(' + '|'.join(_WORD_TO_NUM.keys()) + r')\b', re.IGNORECASE)

def _normalize_street_numbers(addr):
    """Convert spelled-out numbers to digits (Eight Mile -> 8 Mile). Nominatim fails on word numbers."""
    return _WORD_NUM_RE.sub(lambda m: _WORD_TO_NUM[m.group(1).lower()], addr)

def _census_geocode(address):
    """US Census Bureau geocoder — free, no API key, highest accuracy for US addresses."""
    try:
        r = requests.get(
            'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress',
            params={'address': address, 'benchmark': 'Public_AR_Current', 'format': 'json'},
            timeout=10
        )
        matches = r.json().get('result', {}).get('addressMatches', [])
        if matches:
            c = matches[0]['coordinates']
            return float(c['y']), float(c['x'])  # lat, lng
    except Exception as e:
        log.warning(f'Census geocode failed for {address}: {e}')
    return None, None

def geocode_address(address):
    # 0. Check in-memory cache first (already loaded from address_intel on startup)
    if address in _geocache:
        cached = _geocache[address]
        if cached[0]:   # valid hit
            return cached
    key = _normalize_addr_key(address)
    if key in _geocache and _geocache[key][0]:
        _geocache[address] = _geocache[key]   # alias for future hits
        return _geocache[key]

    # Normalize spelled-out numbers (Eight Mile -> 8 Mile)
    normalized = _normalize_street_numbers(address)
    # Strip apt/unit suffixes before geocoding
    clean = re.sub(r'\s+(Apt|Unit|Suite|Ste|#)\s*[\w-]+', '', normalized, flags=re.IGNORECASE).strip()

    # 1. Try US Census Bureau (most accurate for US addresses, free, no key)
    lat, lng = _census_geocode(clean)
    if lat and lng:
        log.info(f'Census geocode hit: {address} -> {lat:.5f}, {lng:.5f}')
        _geocache[address] = (lat, lng)
        upsert_address_intel(address, lat, lng)   # persist for future routes
        return lat, lng

    # 2. Fall back to Nominatim
    try:
        geo = Nominatim(user_agent='unit-delivery-app', timeout=8)
        loc = geo.geocode(clean) or geo.geocode(normalized) or geo.geocode(address)
        if loc:
            log.info(f'Nominatim fallback hit: {address} -> {loc.latitude:.5f}, {loc.longitude:.5f}')
            _geocache[address] = (loc.latitude, loc.longitude)
            upsert_address_intel(address, loc.latitude, loc.longitude)   # persist
            return loc.latitude, loc.longitude
    except Exception as e:
        log.warning(f'Nominatim geocode failed for {address}: {e}')

    _geocache[address] = (None, None)
    return None, None

TEXTBELT_KEY = os.environ.get('TEXTBELT_KEY', '')

def send_sms(to_phone, message, media_url=None):
    """
    Send SMS (or MMS when media_url is provided).
    - Textbelt: SMS only (no MMS support) — used when TEXTBELT_KEY is set
    - Twilio: MMS when media_url provided, SMS otherwise
    """
    # Use Textbelt if key provided (no A2P registration needed, SMS only)
    if TEXTBELT_KEY:
        try:
            resp = requests.post('https://textbelt.com/text', {
                'phone': to_phone,
                'message': message,
                'key': TEXTBELT_KEY
            }, timeout=10).json()
            if resp.get('success'):
                log.info(f'Textbelt SMS sent to {to_phone}')
                return True, 'textbelt'
            else:
                log.error(f'Textbelt failed: {resp}')
                return False, resp.get('error', 'unknown')
        except Exception as e:
            log.error(f'Textbelt error: {e}')
            return False, str(e)

    # Fallback to Twilio (supports MMS via media_url)
    if not TWILIO_SID or not TWILIO_TOKEN:
        log.info(f'[SMS MOCK] To: {to_phone} | {message[:80]}')
        return True, 'mock'
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        kwargs = dict(body=message, from_=TWILIO_PHONE, to=to_phone)
        if media_url:
            kwargs['media_url'] = [media_url]
        msg = client.messages.create(**kwargs)
        log.info(f'{"MMS" if media_url else "SMS"} sent to {to_phone}: {msg.sid}')
        return True, msg.sid
    except Exception as e:
        log.error(f'SMS failed to {to_phone}: {e}')
        return False, str(e)

def miles_away(lat1, lng1, lat2, lng2):
    try:
        if None in (lat1, lng1, lat2, lng2): return 999
        return geodesic((lat1, lng1), (lat2, lng2)).miles
    except: return 999

def send_imessage_to_driver(phone, message):
    """Send iMessage to driver via Mac mini AppleScript — no Twilio needed."""
    import subprocess
    try:
        clean = phone.replace(' ','').replace('-','').replace('(','').replace(')','')
        if not clean.startswith('+'): clean = '+1' + clean.lstrip('1')
        script = f"""tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{clean}" of targetService
    send "{message}" to targetBuddy
end tell"""
        result = subprocess.run(['osascript', '-e', script],
                                capture_output=True, text=True, timeout=8)
        if result.returncode == 0:
            log.info(f'iMessage sent to driver {clean}')
            return True
        else:
            log.warning(f'iMessage failed: {result.stderr.strip()}')
            return False
    except Exception as e:
        log.error(f'send_imessage_to_driver error: {e}')
        return False

def get_base_url():
    # Always force HTTPS in production — HTTP links are unclickable in SMS on many carriers
    explicit = os.environ.get('BASE_URL', '').rstrip('/')
    if explicit:
        return explicit
    url = request.host_url.rstrip('/')
    # Force https:// — Render proxy may pass http:// internally
    if url.startswith('http://'):
        url = 'https://' + url[7:]
    return url

def parse_speedx_screenshot(text):
    """Parse Speed X app screenshot OCR text into structured stops."""
    stops = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    tracking_pat = re.compile(r'(SPXDTW\w+)', re.IGNORECASE)
    stop_num_pat = re.compile(r'Stop[:\s]+?(\d+)', re.IGNORECASE)
    # Address: starts with number, contains city/state/zip
    addr_pat     = re.compile(r'^(\d+\s+.+?),\s*([A-Za-z\s]+),\s*([A-Z]{2})[,\s]+(\d{5})', re.IGNORECASE)
    # Loose address for two-line format
    street_pat   = re.compile(r'^\d+\s+[A-Za-z]', re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for tracking number as anchor
        tracking_match = tracking_pat.search(line)
        if tracking_match:
            tracking = tracking_match.group(1)
            # Look back up to 4 lines for address + customer
            address = ''
            customer = ''
            stop_num = ''
            for j in range(max(0, i-4), i):
                m = addr_pat.match(lines[j])
                if m:
                    street = m.group(1).strip()
                    city   = m.group(2).strip()
                    state  = m.group(3).strip()
                    zipcode = m.group(4).strip()
                    address = f"{street}, {city}, {state} {zipcode}"
                elif street_pat.match(lines[j]) and j+1 < len(lines):
                    # Two-line address — combine with next
                    next_line = lines[j+1]
                    combined = lines[j] + ',' + next_line
                    m2 = addr_pat.match(combined)
                    if m2:
                        street  = m2.group(1).strip()
                        city    = m2.group(2).strip()
                        state   = m2.group(3).strip()
                        zipcode = m2.group(4).strip()
                        address = f"{street}, {city}, {state} {zipcode}"
                # Customer name — no digits, no SPXDTW, reasonable length
                if (not re.search(r'\d', lines[j]) and
                    'SPXDTW' not in lines[j].upper() and
                    'Stop' not in lines[j] and
                    'parcel' not in lines[j].lower() and
                    'arrival' not in lines[j].lower() and
                    3 < len(lines[j]) < 40):
                    customer = lines[j]
            # Look forward for stop number
            for j in range(i, min(i+3, len(lines))):
                sn = stop_num_pat.search(lines[j])
                if sn:
                    stop_num = sn.group(1)
                    break
            if address:
                # Clean address — remove USA suffix
                address = re.sub(r',?\s*USA\s*$', '', address, flags=re.IGNORECASE).strip()
                stops.append({
                    'address':  address,
                    'name':     customer,
                    'tracking': tracking,
                    'stop_num': stop_num
                })
        i += 1
    return stops

def parse_stops_from_text(text):
    """Try Speed X parser first, fall back to generic address extraction."""
    # Try SpeedX format first
    speedx_stops = parse_speedx_screenshot(text)
    if speedx_stops:
        return speedx_stops

    # Generic fallback
    stops = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    addr_pattern = re.compile(r'^\d+\s+[A-Za-z].*,(\s*\w+,)?\s*[A-Z]{2}\s+\d{5}', re.IGNORECASE)
    loose_pattern = re.compile(r'^\d+\s+[A-Za-z][A-Za-z\s]+(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl|Cir|Hwy|Pkwy|Terr?|Trail|Loop)[\.\s,]', re.IGNORECASE)
    for i, line in enumerate(lines):
        if addr_pattern.match(line) or loose_pattern.match(line):
            name = lines[i-1] if i > 0 and not lines[i-1][0].isdigit() else ''
            stops.append({'address': line, 'name': name})
    return stops

def format_phone(phone):
    digits = ''.join(c for c in str(phone) if c.isdigit())
    if len(digits) == 10: return f'+1{digits}'
    if len(digits) == 11 and digits[0] == '1': return f'+{digits}'
    return phone

# ─── ROUTES ────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ─── DRIVER AUTH ───────────────────────────────────────────────

@app.route('/driver/login', methods=['GET', 'POST'])
def driver_login():
    error = None
    ip = get_real_ip()
    if request.method == 'POST':
        if is_rate_limited(ip):
            return render_template('driver_login.html', error='Too many attempts. Try again in 5 minutes.')
        pin = request.form.get('pin', '').strip()
        db = get_db()
        driver = db.execute("SELECT * FROM drivers WHERE pin=?", (pin,)).fetchone()
        if driver:
            session['driver_id'] = driver['id']
            session['driver_name'] = driver['name']
            onboarded = db.execute(
                "SELECT 1 FROM driver_onboarding WHERE driver_id=?",
                (driver['id'],)
            ).fetchone()
            db.close()
            if not onboarded:
                return redirect(url_for('driver_walkthrough'))
            # Return to a QR building gate if the driver scanned before logging in
            dest = session.pop('post_login_redirect', None)
            if dest:
                return redirect(dest)
            return redirect(url_for('driver_dashboard'))
        db.close()
        record_attempt(ip)
        error = 'Invalid PIN'
    return render_template('driver_login.html', error=error)

@app.route('/driver/walkthrough')
def driver_walkthrough():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    return render_template('driver_walkthrough.html', driver=session['driver_name'])

@app.route('/driver/walkthrough/complete', methods=['POST'])
def driver_walkthrough_complete():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    try:
        db.execute(
            "INSERT INTO driver_onboarding (driver_id) VALUES (?)",
            (session['driver_id'],)
        )
        db.commit()
    except Exception as e:
        log.error(f'Walkthrough complete error: {e}')
        try: db._conn.rollback()
        except: pass
    db.close()
    return redirect(url_for('driver_dashboard'))

@app.route('/driver/logout')
def driver_logout():
    session.clear()
    return redirect(url_for('driver_login'))

# ─── DRIVER DASHBOARD ──────────────────────────────────────────

@app.route('/driver')
def driver_dashboard():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    driver_row = db.execute("SELECT * FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    pay_rate = float(driver_row['pay_rate']) if driver_row and driver_row['pay_rate'] else 1.50

    route = db.execute(
        "SELECT * FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()
    stops = []
    if route:
        stops = db.execute(
            "SELECT * FROM stops WHERE route_id=? ORDER BY stop_number",
            (route['id'],)
        ).fetchall()

    # Today earnings
    today_delivered = db.execute(
        """SELECT COUNT(*) FROM stops s
           JOIN routes r ON s.route_id = r.id
           WHERE r.driver_id=? AND r.date=? AND s.status='delivered'""",
        (session['driver_id'], today)
    ).fetchone()[0]
    today_total = db.execute(
        """SELECT COUNT(*) FROM stops s
           JOIN routes r ON s.route_id = r.id
           WHERE r.driver_id=? AND r.date=?""",
        (session['driver_id'], today)
    ).fetchone()[0]

    # Week earnings (Mon-Sun of current week)
    from datetime import date
    today_date = date.today()
    week_start = (today_date - timedelta(days=today_date.weekday())).strftime('%Y-%m-%d')
    week_stop_count = db.execute(
        """SELECT COUNT(*) FROM stops s
           JOIN routes r ON s.route_id = r.id
           WHERE r.driver_id=? AND r.date >= ? AND s.status='delivered'""",
        (session['driver_id'], week_start)
    ).fetchone()[0]

    # Month earnings (1st of current month → today)
    month_start = today_date.replace(day=1).strftime('%Y-%m-%d')
    month_stop_count = db.execute(
        """SELECT COUNT(*) FROM stops s
           JOIN routes r ON s.route_id = r.id
           WHERE r.driver_id=? AND r.date >= ? AND s.status='delivered'""",
        (session['driver_id'], month_start)
    ).fetchone()[0]

    # Weekly route history (last 7 days)
    week_history = db.execute(
        """SELECT r.date, COUNT(s.id) as total,
           SUM(CASE WHEN s.status='delivered' THEN 1 ELSE 0 END) as delivered
           FROM routes r
           LEFT JOIN stops s ON s.route_id = r.id
           WHERE r.driver_id=? AND r.date >= ?
           GROUP BY r.date ORDER BY r.date DESC""",
        (session['driver_id'], week_start)
    ).fetchall()

    db.close()

    today_earnings  = round(today_delivered  * pay_rate, 2)
    week_earnings   = round(week_stop_count  * pay_rate, 2)
    month_earnings  = round(month_stop_count * pay_rate, 2)

    vehicle_type = _ss_val(driver_row, 'vehicle_type', 'suv_midsize') or 'suv_midsize'
    zone_summary = []
    zone_centroids = []
    current_zone = None
    checkin_status = 'unknown'
    assignment_today = None
    try:
        dbc = get_db()
        crow = dbc.execute(
            "SELECT status, assignment FROM driver_checkins WHERE driver_id=? AND check_date=?",
            (session['driver_id'], today)
        ).fetchone()
        if crow:
            checkin_status = crow['status']
            assignment_today = crow['assignment'] if 'assignment' in crow.keys() else None
        dbc.close()
    except Exception:
        pass

    next_stop = None
    finish_estimate = None
    if route and stops:
        db2 = get_db()
        stops, zone_summary, zone_centroids, current_zone, next_stop = build_route_zone_context(
            stops, route, vehicle_type, db2
        )
        finish_estimate = _route_finish_estimate(db2, route['id'])
        db2.close()

    return render_template('driver_dashboard.html',
        route=route, stops=stops, driver=session['driver_name'],
        gmaps_key=GOOGLE_MAPS_KEY, mapbox_token=MAPBOX_TOKEN,
        pay_rate=pay_rate,
        today_delivered=today_delivered,
        today_total=today_total,
        today_earnings=today_earnings,
        week_stop_count=week_stop_count,
        week_earnings=week_earnings,
        month_stop_count=month_stop_count,
        month_earnings=month_earnings,
        week_history=week_history,
        zone_summary=zone_summary,
        zone_centroids=zone_centroids,
        current_zone=current_zone,
        next_stop=next_stop,
        vehicle_type=vehicle_type,
        finish_estimate=finish_estimate,
        checkin_status=checkin_status,
        assignment_today=assignment_today,
    )


@app.route('/driver/checkin', methods=['POST'])
def driver_checkin():
    """Driver sets their own attendance for today (green/red dot for manager)."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    status = request.form.get('status', 'in').strip()
    if status not in ('in', 'out'):
        status = 'in'
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now().isoformat()
    db = get_db()
    existing = db.execute(
        "SELECT id FROM driver_checkins WHERE driver_id=? AND check_date=?",
        (session['driver_id'], today)
    ).fetchone()
    if existing:
        db.execute("UPDATE driver_checkins SET status=?, updated_at=? WHERE id=?",
                   (status, now, existing['id']))
    else:
        db.execute(
            "INSERT INTO driver_checkins (driver_id, check_date, status, updated_at) VALUES (?,?,?,?)",
            (session['driver_id'], today, status, now)
        )
    db.commit()
    db.close()
    return redirect(url_for('driver_dashboard'))

# ─── LIVE EARNINGS API ─────────────────────────────────
@app.route('/api/driver/earnings')
def api_driver_earnings():
    """Returns current driver earnings — called live from dashboard."""
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    from datetime import date
    db       = get_db()
    driver   = db.execute("SELECT pay_rate FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    pay_rate = float(driver['pay_rate']) if driver and driver['pay_rate'] else 1.50
    today    = date.today().strftime('%Y-%m-%d')
    today_delivered = db.execute(
        """SELECT COUNT(*) FROM stops s JOIN routes r ON s.route_id=r.id
           WHERE r.driver_id=? AND r.date=? AND s.status='delivered'""",
        (session['driver_id'], today)
    ).fetchone()[0]
    today_total = db.execute(
        """SELECT COUNT(*) FROM stops s JOIN routes r ON s.route_id=r.id
           WHERE r.driver_id=? AND r.date=?""",
        (session['driver_id'], today)
    ).fetchone()[0]
    week_start = (date.today() - timedelta(days=date.today().weekday())).strftime('%Y-%m-%d')
    week_delivered = db.execute(
        """SELECT COUNT(*) FROM stops s JOIN routes r ON s.route_id=r.id
           WHERE r.driver_id=? AND r.date>=? AND s.status='delivered'""",
        (session['driver_id'], week_start)
    ).fetchone()[0]
    month_start = date.today().replace(day=1).strftime('%Y-%m-%d')
    month_delivered = db.execute(
        """SELECT COUNT(*) FROM stops s JOIN routes r ON s.route_id=r.id
           WHERE r.driver_id=? AND r.date>=? AND s.status='delivered'""",
        (session['driver_id'], month_start)
    ).fetchone()[0]
    db.close()
    return jsonify({
        'pay_rate':        pay_rate,
        'today_delivered': today_delivered,
        'today_total':     today_total,
        'today_earnings':  round(today_delivered  * pay_rate, 2),
        'week_earnings':   round(week_delivered   * pay_rate, 2),
        'month_earnings':  round(month_delivered  * pay_rate, 2),
    })

# ─── TEMP DEBUG ────────────────────────────────────────
@app.route('/driver/debug-dashboard')
def debug_dashboard():
    try:
        db = get_db()
        today = datetime.now().strftime('%Y-%m-%d')
        route = db.execute(
            "SELECT * FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
            (1, today)
        ).fetchone()
        db.close()
        return jsonify({'ok': True, 'route': dict(route) if route else None})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

# ─── PACKAGE SCAN ──────────────────────────────────────────────

def _get_or_create_scan_session(db, driver_id):
    """Get today's open scan session or create one."""
    today = datetime.now().strftime('%Y-%m-%d')
    ss = db.execute(
        "SELECT * FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (driver_id, today)
    ).fetchone()
    if ss:
        return ss['id']
    db.execute(
        "INSERT INTO scan_sessions (driver_id, date, status) VALUES (?,?,?)",
        (driver_id, today, 'scanning')
    )
    db.commit()
    row = db.execute(
        "SELECT id FROM scan_sessions WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
        (driver_id, today)
    ).fetchone()
    return row['id']


def _ss_val(row, key, default=None):
    """Safe column read for sqlite Row / PG dict (missing migration columns)."""
    if not row:
        return default
    try:
        if hasattr(row, 'keys') and key not in row.keys():
            return default
        val = row[key]
        return val if val is not None else default
    except (KeyError, IndexError, TypeError):
        return default


def _scan_order_for_item(item, fallback_index):
    """Permanent warehouse sticker number (1-based). Set at first scan, never changes."""
    so = _ss_val(item, 'scan_order')
    if so is not None and int(so) > 0:
        return int(so)
    return fallback_index + 1


def _scan_order_map(items):
    return {item['id']: _scan_order_for_item(item, i) for i, item in enumerate(items)}


def _attach_scan_orders(packages, order_map):
    for p in packages:
        p['scan_order'] = order_map.get(p['id'], p.get('scan_order') or 0)
    return packages


def _ensure_unique_scan_orders(db, session_id):
    """Fix duplicate or missing warehouse sticker numbers."""
    items = db.execute(
        "SELECT id, scan_order FROM scan_items WHERE session_id=? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    used = set()
    fixes = []
    next_free = 1
    for item in items:
        so = _ss_val(item, 'scan_order')
        if so is not None and int(so) > 0 and int(so) not in used:
            used.add(int(so))
            next_free = max(next_free, int(so) + 1)
            continue
        while next_free in used:
            next_free += 1
        fixes.append((next_free, item['id']))
        used.add(next_free)
        next_free += 1
    for new_so, item_id in fixes:
        db.execute("UPDATE scan_items SET scan_order=? WHERE id=?", (new_so, item_id))
    if fixes:
        db.commit()


def _parse_route_endpoints(data):
    """Extract start/end depot from optimize request JSON."""
    data = data or {}
    start = end = None
    if data.get('start_lat') is not None and data.get('start_lng') is not None:
        start = {'lat': float(data['start_lat']), 'lng': float(data['start_lng'])}
    end_mode = (data.get('end_mode') or 'none').strip().lower()
    if end_mode == 'return_start' and start:
        end = dict(start)
    elif data.get('end_lat') is not None and data.get('end_lng') is not None:
        end = {'lat': float(data['end_lat']), 'lng': float(data['end_lng'])}
    elif data.get('end_address'):
        coords = geocode_address(data['end_address'].strip())
        if coords:
            end = {'lat': coords[0], 'lng': coords[1]}
    return start, end, end_mode


def _persist_optimized_scan_items(db, session_id, sorted_pkgs):
    """Save locked zone + delivery order onto scan_items for build-route."""
    for p in sorted_pkgs:
        try:
            db.execute(
                """UPDATE scan_items SET zone_letter=?, delivery_order=?
                   WHERE id=? AND session_id=?""",
                (p.get('zone_letter', '?'), p.get('delivery_order', 0), p['id'], session_id)
            )
        except Exception as e:
            log.warning(f'[scan] persist zone/order failed for item {p.get("id")}: {e}')
    db.commit()


def _unlock_stale_scan_session(db, ss):
    """
    Auto-lock from the old flow set zones_locked without phase='optimized'.
    Reset so drivers can scan → optimize → build cleanly.
    """
    if not ss:
        return ss
    phase = _ss_val(ss, 'phase', 'scanning') or 'scanning'
    if _ss_val(ss, 'zones_locked', 0) and phase != 'optimized':
        db.execute(
            """UPDATE scan_sessions
               SET zones_locked=0, zone_centroids=NULL, prev_centroids=NULL, phase='scanning'
               WHERE id=?""",
            (ss['id'],)
        )
        db.commit()
        return db.execute("SELECT * FROM scan_sessions WHERE id=?", (ss['id'],)).fetchone()
    return ss


@app.route('/driver/scan/label-memory', methods=['GET'])
def scan_label_memory():
    """Sync learned label reads to the phone for instant offline recall."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    try:
        db = get_db()
        rows = db.execute(
            """SELECT tracking, customer_name, address, zip_code, read_count, updated_at
               FROM label_memory ORDER BY updated_at DESC LIMIT 800"""
        ).fetchall()
        db.close()
        items = [{
            'tracking': r['tracking'],
            'name': r['customer_name'] or '',
            'address': r['address'] or '',
            'zip': r['zip_code'] or '',
            'read_count': r['read_count'] or 1,
        } for r in rows]
        return jsonify({'ok': True, 'items': items})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/driver/scan/infer-address', methods=['POST'])
def scan_infer_address():
    """Suggest full addresses from partial/warped label fragments + delivery history."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    partial = (data.get('address') or '').strip()
    ocr_text = (data.get('ocr_text') or '').strip()
    fr = _extract_address_fragments(partial, ocr_text)
    hints = []
    if fr.get('street_num'):
        hints.append(f"#{fr['street_num']}")
    if fr.get('street_name'):
        hints.append(fr['street_name'])
    elif fr.get('street_tokens'):
        hints.append(fr['street_tokens'][0])
    if fr.get('unit'):
        hints.append(f"Apt {fr['unit']}")
    if fr.get('city'):
        hints.append(fr['city'])
    if fr.get('zip'):
        hints.append(fr['zip'])
    suggestions = infer_address_suggestions(
        tracking=(data.get('tracking') or '').strip(),
        partial_address=partial,
        ocr_text=ocr_text,
        name=(data.get('name') or '').strip(),
        zip_code=(data.get('zip') or '').strip(),
        limit=6,
    )
    return jsonify({
        'ok': True,
        'suggestions': suggestions,
        'detected': hints,
    })


@app.route('/driver/scan', methods=['GET'])
def scan_packages():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    ss_id = _get_or_create_scan_session(db, session['driver_id'])
    ss = db.execute("SELECT * FROM scan_sessions WHERE id=?", (ss_id,)).fetchone()
    ss = _unlock_stale_scan_session(db, ss)
    items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=? ORDER BY id ASC",
        (ss_id,)
    ).fetchall()
    zones_locked = bool(_ss_val(ss, 'zones_locked', 0)) if ss else False
    phase = _ss_val(ss, 'phase', 'scanning') or 'scanning'
    db.close()
    return render_template(
        'scan.html',
        items=items,
        session_id=ss_id,
        driver=session['driver_name'],
        zones_locked=zones_locked,
        phase=phase,
    )


@app.route('/driver/scan/process', methods=['POST'])
def scan_process():
    """
    Receive label photo, run Claude Vision, return parsed JSON.
    If the tracking number already exists in the session (re-scan / lookup),
    returns mode='lookup' with the package's current zone assignment.
    """
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401

    # Fast path: phone already read the label locally (no photo upload / no AI wait)
    if request.is_json:
        data = request.get_json(silent=True) or {}
        tracking = (data.get('tracking') or '').strip()
        ocr_text = (data.get('ocr_text') or '').strip()

        # Instant recall — learned from past confirmed scans
        if tracking:
            mem = lookup_label_memory(tracking)
            if mem and mem.get('address'):
                mem['source'] = 'memory'
                return _finish_scan_label_result(mem)

        if ocr_text:
            result = parse_label_text(ocr_text)
            result['tracking'] = result.get('tracking') or tracking
            if result['tracking']:
                mem = lookup_label_memory(result['tracking'])
                if mem and mem.get('address'):
                    if not result.get('name'):
                        result['name'] = mem.get('name') or ''
                    if not result.get('address'):
                        result['address'] = mem.get('address') or ''
                    result['zip'] = result.get('zip') or mem.get('zip') or ''
        else:
            result = {
                'tracking': tracking,
                'name':     (data.get('name') or '').strip(),
                'address':  (data.get('address') or '').strip(),
                'zip':      (data.get('zip') or '').strip(),
            }

        if not result.get('address') and not result.get('tracking'):
            return jsonify({'ok': False, 'error': 'No address or tracking in request'})
        return _finish_scan_label_result(result)

    file = request.files.get('photo')
    if not file:
        return jsonify({'ok': False, 'error': 'No photo received'})
    img_bytes = file.read()
    if not _vision_available():
        return jsonify({'ok': False, 'error': 'Vision AI not configured — add GEMINI_API_KEY (free) on server'})
    if not img_bytes:
        return jsonify({'ok': False, 'error': 'Empty photo received — try capturing again'})
    try:
        result = extract_package_label(img_bytes)
    except Exception as e:
        log.error(f'scan_process label read failed: {e}')
        err = str(e)
        if 'Gemini vision failed' in err or '404' in err or 'not found' in err.lower():
            err = 'AI temporarily unavailable — use on-phone read or confirm manually'
        return jsonify({'ok': False, 'error': err, 'ai_failed': True})
    if not result or not isinstance(result, dict):
        return jsonify({'ok': False, 'error': 'Could not read label — try a clearer, well-lit photo'})
    # Empty extraction = unreadable label (rather than a hard error)
    if not (result.get('address') or '').strip() and not (result.get('tracking') or '').strip():
        return jsonify({'ok': False, 'error': 'No address or tracking found — move closer and fill the frame with the label'})

    # Merge with learned memory when AI got tracking but weak address
    trk = (result.get('tracking') or '').strip()
    if trk:
        mem = lookup_label_memory(trk)
        if mem and mem.get('address'):
            if not (result.get('address') or '').strip():
                result['address'] = mem['address']
            if not (result.get('name') or '').strip():
                result['name'] = mem.get('name') or ''
            result['zip'] = result.get('zip') or mem.get('zip') or ''

    return _finish_scan_label_result(result)


def _finish_scan_label_result(result):
    """Duplicate / lookup / confirm response after label fields are known."""
    tracking = (result.get('tracking') or '').strip()
    if tracking:
        db = get_db()
        today = datetime.now().strftime('%Y-%m-%d')
        ss = db.execute(
            "SELECT * FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
            (session['driver_id'], today)
        ).fetchone()
        if ss:
            existing = db.execute(
                "SELECT * FROM scan_items WHERE session_id=? AND tracking=? LIMIT 1",
                (ss['id'], tracking)
            ).fetchone()
            if existing:
                ss = _unlock_stale_scan_session(db, ss)
                zones_locked = bool(_ss_val(ss, 'zones_locked', 0))
                phase = _ss_val(ss, 'phase', 'scanning') or 'scanning'
                # During scan phase, tell client it's a duplicate — not a zone lookup
                if not zones_locked or phase != 'optimized':
                    db.close()
                    return jsonify({
                        'ok': True,
                        'mode': 'duplicate',
                        'tracking': tracking,
                        'address': (f"{existing['customer_name']} — " if existing['customer_name'] else '') + existing['address'],
                        'data': result,
                    })
                # Package already in session — look up its zone + vehicle spot
                stored_cents = json.loads(ss['zone_centroids']) if ss['zone_centroids'] else None
                lookup_zone  = {}
                if stored_cents and existing['dest_lat'] and existing['dest_lng']:
                    pkg_pt  = {'lat': existing['dest_lat'], 'lng': existing['dest_lng']}
                    nearest = min(stored_cents, key=lambda c: _dsq(pkg_pt, c))
                    letter  = nearest['letter']
                    # Get driver vehicle type for vehicle zone lookup
                    drv = db.execute("SELECT vehicle_type FROM drivers WHERE id=?",
                                     (session['driver_id'],)).fetchone()
                    vtype  = drv['vehicle_type'] if drv and drv['vehicle_type'] else 'suv_midsize'
                    vzones = VEHICLE_ZONES.get(vtype, VEHICLE_ZONES['suv_midsize'])
                    # All letters in order to map delivery zone → vehicle zone
                    all_letters = sorted({c['letter'] for c in stored_cents})
                    n_dlv = len(all_letters)
                    seq   = all_letters.index(letter) if letter in all_letters else 0
                    if n_dlv > 1:
                        v_idx = int(round(seq / (n_dlv - 1) * (len(vzones) - 1)))
                    else:
                        v_idx = 0
                    v_idx = len(vzones) - 1 - v_idx
                    v_idx = max(0, min(v_idx, len(vzones) - 1))
                    vz    = vzones[v_idx]
                    lookup_zone = {
                        'zone_letter':       letter,
                        'zone_color':        nearest['color'],
                        'zone_emoji':        nearest['emoji'],
                        'vehicle_zone_label': vz['label'],
                        'vehicle_zone_desc':  vz['desc'],
                        'zones_locked':       zones_locked,
                    }
                db.close()
                return jsonify({
                    'ok':      True,
                    'mode':    'lookup',
                    'tracking': tracking,
                    'address': (f"{existing['customer_name']} — " if existing['customer_name'] else '') + existing['address'],
                    'data':    result,
                    **lookup_zone
                })
        db.close()

    return jsonify({'ok': True, 'mode': 'confirm', 'data': result})


@app.route('/driver/scan/add', methods=['POST'])
def scan_add():
    """
    Add a confirmed package to the scan session.
    If zones are already locked (import-first path), match tracking number to
    pre-loaded stop and return instant zone assignment.
    Otherwise geocode immediately for live-sort calibration.
    """
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    tracking = data.get('tracking', '').strip()
    name = data.get('name', '').strip()
    address = data.get('address', '').strip()
    zip_code = data.get('zip', '').strip()
    if not address:
        return jsonify({'ok': False, 'error': 'Address is required'})
    unit = (data.get('unit') or '').strip().upper() or extract_unit_number(address)

    # ─ Import-first fast path: zones locked, match by tracking ─
    db0 = get_db()
    today0 = datetime.now().strftime('%Y-%m-%d')
    ss0 = db0.execute(
        "SELECT * FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today0)
    ).fetchone()
    if ss0 and ss0['zones_locked'] and tracking:
        # Look up the tracking number in pre-loaded stops
        pre = db0.execute(
            "SELECT * FROM scan_items WHERE session_id=? AND tracking=? LIMIT 1",
            (ss0['id'], tracking)
        ).fetchone()
        if pre:
            # Mark as confirmed (reuse existing row, just return zone info)
            stored_cents = json.loads(ss0['zone_centroids']) if ss0['zone_centroids'] else []
            lat, lng = pre['dest_lat'], pre['dest_lng']
            pkg = {'lat': lat, 'lng': lng, 'delivery_order': 1}
            zone_info = {}
            if stored_cents and lat and lng:
                nearest = min(stored_cents, key=lambda c: _dsq(pkg, c))
                zone_info = {
                    'zone_letter': nearest['letter'],
                    'zone_color':  nearest['color'],
                    'zone_emoji':  nearest['emoji'],
                }
            count = db0.execute("SELECT COUNT(*) FROM scan_items WHERE session_id=?", (ss0['id'],)).fetchone()[0]
            db0.close()
            return jsonify({
                'ok': True, 'count': count, 'geocoded': bool(lat),
                'new_item_id': pre['id'], 'zip_warning': None,
                'zones_locked': True, 'preloaded_match': True,
                **zone_info
            })
    db0.close()

    # Geocode immediately so live-sort works right away
    lat, lng = None, None
    try:
        coords = geocode_address(address)
        if coords:
            lat, lng = coords
    except Exception as e:
        log.warning(f'Live geocode failed for {address}: {e}')

    db = get_db()
    ss_id = _get_or_create_scan_session(db, session['driver_id'])

    # Block duplicate adds (double-tap while geocoding)
    addr_key = _normalize_addr_key(address)
    trk_key = re.sub(r'\s+', '', tracking.upper()) if tracking else ''
    dup = None
    if trk_key:
        dup = db.execute(
            "SELECT id, scan_order, dest_lat FROM scan_items WHERE session_id=? AND REPLACE(UPPER(COALESCE(tracking,'')),' ','')=? LIMIT 1",
            (ss_id, trk_key),
        ).fetchone()
    if not dup and addr_key:
        dup = db.execute(
            "SELECT id, scan_order, dest_lat FROM scan_items WHERE session_id=? AND UPPER(TRIM(address))=? LIMIT 1",
            (ss_id, addr_key),
        ).fetchone()
    if dup:
        count = db.execute("SELECT COUNT(*) FROM scan_items WHERE session_id=?", (ss_id,)).fetchone()[0]
        db.close()
        return jsonify({
            'ok': True, 'duplicate': True, 'count': count,
            'scan_order': dup['scan_order'], 'new_item_id': dup['id'],
            'geocoded': bool(dup['dest_lat']), 'zip_warning': None,
        })

    # Check zip against driver's assigned zips
    zip_warning = None
    driver_row = db.execute("SELECT assigned_zips FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    if driver_row and driver_row['assigned_zips'] and zip_code:
        assigned = [z.strip() for z in driver_row['assigned_zips'].split(',') if z.strip()]
        if assigned and zip_code not in assigned:
            zip_warning = f'ZIP {zip_code} is outside your assigned zone ({driver_row["assigned_zips"]})'

    next_order = db.execute(
        "SELECT COALESCE(MAX(scan_order), 0) + 1 FROM scan_items WHERE session_id=?",
        (ss_id,),
    ).fetchone()[0]
    db.execute(
        """INSERT INTO scan_items (session_id, tracking, customer_name, address, zip_code, raw_json, dest_lat, dest_lng, scan_order, unit)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (ss_id, tracking, name, address, zip_code, json.dumps(data), lat, lng, next_order, unit)
    )
    db.commit()
    new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.close()
    if tracking and address:
        upsert_label_memory(tracking, name, address, zip_code, source='confirm')

    # Building memory: no unit found on a known multi-unit building → flag now, not at the door
    unit_warning = None
    known_units = []
    if not unit:
        try:
            known_units = get_known_units(address)
            if known_units or is_known_multi_unit(address):
                unit_warning = 'No unit number found — this looks like a multi-unit building. Enter it now before you leave.'
        except Exception as e:
            log.warning(f'[building_memory] scan_add check failed: {e}')

    return jsonify({'ok': True, 'count': next_order, 'scan_order': next_order, 'geocoded': lat is not None,
                   'new_item_id': new_id, 'zip_warning': zip_warning,
                   'unit': unit, 'unit_warning': unit_warning, 'known_units': known_units})


@app.route('/driver/scan/live-sort', methods=['GET'])
def scan_live_sort():
    """Return all scanned packages sorted in optimal route order with geo coords."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    ss = db.execute(
        "SELECT id FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()
    if not ss:
        db.close()
        return jsonify({'ok': True, 'items': [], 'sorted': False})

    items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=? ORDER BY id ASC",
        (ss['id'],)
    ).fetchall()
    # Get driver vehicle type + zone lock state
    driver   = db.execute("SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    ss_full  = db.execute("SELECT * FROM scan_sessions WHERE id=?", (ss['id'],)).fetchone()
    vehicle_type    = (driver['vehicle_type'] if driver and driver['vehicle_type'] else 'suv_midsize')
    ss_full         = _unlock_stale_scan_session(db, ss_full) if ss_full else None
    zones_locked    = bool(_ss_val(ss_full, 'zones_locked', 0)) if ss_full else False
    stored_cents    = json.loads(ss_full['zone_centroids']) if ss_full and ss_full['zone_centroids'] else None
    prev_cents      = json.loads(ss_full['prev_centroids'])  if ss_full and ss_full['prev_centroids']  else None
    current_phase   = _ss_val(ss_full, 'phase', 'scanning') or 'scanning'
    db.close()

    if not items:
        return jsonify({'ok': True, 'items': [], 'sorted': False, 'zones_locked': False,
                        'phase': 'scanning',
                        'vehicle_type': vehicle_type,
                        'vehicle_label': VEHICLE_LABELS.get(vehicle_type, 'Vehicle'),
                        'vehicle_zones': VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv_midsize'])})

    packages = []
    order_map = _scan_order_map(items)
    for item in items:
        item_unit = item['unit'] if 'unit' in item.keys() else None
        packages.append({
            'id':         item['id'],
            'tracking':   item['tracking'],
            'name':       item['customer_name'],
            'address':    item['address'],
            'unit':       item_unit or extract_unit_number(item['address']),
            'lat':        item['dest_lat'],
            'lng':        item['dest_lng'],
            'scan_order': order_map[item['id']],
        })

    # Split into geocoded and non-geocoded
    geocoded   = [p for p in packages if p['lat'] and p['lng']]
    ungeoced   = [p for p in packages if not (p['lat'] and p['lng'])]

    route_miles      = None
    route_drive_mins = None
    naive_miles      = None
    savings_miles    = None
    just_locked      = False

    # ── SCAN PHASE: data collection only — no zones, no optimization ──
    if not zones_locked and current_phase != 'optimized':
        sorted_pkgs = []
        for i, p in enumerate(packages):
            sorted_pkgs.append({
                **p,
                'delivery_order': i + 1,
                'scan_order':     order_map[p['id']],
            })
        total = len(sorted_pkgs)
        return jsonify({
            'ok':              True,
            'items':           sorted_pkgs,
            'sorted':          False,
            'total':           total,
            'zone_summary':    [],
            'zones_locked':    False,
            'just_locked':     False,
            'vehicle_type':    vehicle_type,
            'vehicle_label':   VEHICLE_LABELS.get(vehicle_type, 'Vehicle'),
            'vehicle_zones':   VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv_midsize']),
            'route_miles':     None,
            'route_drive_mins': None,
            'naive_miles':     None,
            'savings_miles':   None,
            'est_total_mins':  None,
            'phase':           'scanning',
        })

    # ─ Naive distance (sequential scan order) for savings calc ─
    if len(geocoded) >= 2:
        naive_dist_m = sum(
            geodesic((geocoded[i]['lat'], geocoded[i]['lng']),
                     (geocoded[i+1]['lat'], geocoded[i+1]['lng'])).meters
            for i in range(len(geocoded) - 1)
        )
        naive_miles = round(naive_dist_m * 0.000621371, 2)

    # ─ OPTIMIZED PHASE: zones locked — run route optimization once ─
    start = end = None
    if ss_full and _ss_val(ss_full, 'route_start_lat') is not None:
        start = {'lat': float(ss_full['route_start_lat']), 'lng': float(ss_full['route_start_lng'])}
    if ss_full and _ss_val(ss_full, 'route_end_lat') is not None:
        end = {'lat': float(ss_full['route_end_lat']), 'lng': float(ss_full['route_end_lng'])}
    sorted_pkgs, total_dist_m, total_dur_s = build_optimized_route(geocoded, start=start, end=end)

    # Add ungeocoded at end
    seen_ids = {p['id'] for p in sorted_pkgs}
    for p in ungeoced:
        if p['id'] not in seen_ids:
            p.update({'zone_letter':'?','zone_num':0,'zone_label_full':'?',
                      'zone_color':'#6b7280','zone_emoji':'⚪',
                      'bag_num':0,'bag_label':'?',
                      'delivery_order': len(sorted_pkgs)+1,
                      'load_position': 0})
            sorted_pkgs.append(p)

    total = len(sorted_pkgs)

    if total_dist_m > 0:
        route_miles      = round(total_dist_m * 0.000621371, 2)
        route_drive_mins = round(total_dur_s / 60, 1)
        savings_miles    = round(naive_miles - route_miles, 2) if naive_miles else None

    if zones_locked and stored_cents:
        # Re-snap to locked centroids to keep zone letters stable
        sorted_pkgs = assign_zones_from_centroids(sorted_pkgs, stored_cents)

    # ─ Assign vehicle cargo zones based on delivery zone letter ─
    sorted_pkgs = assign_vehicle_zones(sorted_pkgs, vehicle_type)
    _attach_scan_orders(sorted_pkgs, order_map)

    # ─ Build zone summary ─
    zone_summary = {}
    for p in sorted_pkgs:
        letter = p.get('zone_letter', '?')
        if letter not in zone_summary:
            zone_summary[letter] = {
                'letter':       letter,
                'count':        0,
                'color':        p.get('zone_color', '#6b7280'),
                'emoji':        p.get('zone_emoji', '⚪'),
                'vehicle_spot': p.get('vehicle_zone_label', ''),
                'load_order':   len(zone_summary) + 1,
            }
        zone_summary[letter]['count'] += 1
    # Add bag counts per zone
    for letter, zs in zone_summary.items():
        n_bags = math.ceil(zs['count'] / BAG_SIZE)
        zs['n_bags'] = n_bags
        zs['bags']   = [f'{letter}-Bag{b}' for b in range(1, n_bags + 1)]
    zone_list = sorted(zone_summary.values(), key=lambda z: z['letter'], reverse=True)
    for i, z in enumerate(zone_list):
        z['load_order'] = i + 1

    # ETA projection
    est_total_mins = None
    if route_drive_mins is not None:
        est_total_mins = round(route_drive_mins + (total * 3), 1)

    return jsonify({
        'ok':              True,
        'items':           sorted_pkgs,
        'sorted':          len(geocoded) >= 2,
        'total':           total,
        'zone_summary':    zone_list,
        'zones_locked':    zones_locked,
        'just_locked':     just_locked,
        'vehicle_type':    vehicle_type,
        'vehicle_label':   VEHICLE_LABELS.get(vehicle_type, 'Vehicle'),
        'vehicle_zones':   VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv_midsize']),
        'route_miles':     route_miles,
        'route_drive_mins': route_drive_mins,
        'naive_miles':     naive_miles,
        'savings_miles':   savings_miles,
        'est_total_mins':  est_total_mins,
        'phase':           current_phase,
    })


@app.route('/driver/scan/optimize', methods=['POST'])
def scan_optimize():
    """
    Run route optimization ONCE when driver is done scanning.
    Locks zones permanently. Returns loading plan.
    Packages will NEVER be re-sorted after this point.
    """
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401

    req_data = request.get_json(silent=True) or {}
    start, end, end_mode = _parse_route_endpoints(req_data)

    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    ss = db.execute(
        "SELECT * FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()

    if not ss:
        db.close()
        return jsonify({'ok': False, 'error': 'No active scan session'})

    items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=? ORDER BY id ASC",
        (ss['id'],)
    ).fetchall()

    driver = db.execute("SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    vehicle_type = (driver['vehicle_type'] if driver and driver['vehicle_type'] else 'suv_midsize')

    if not items:
        db.close()
        return jsonify({'ok': False, 'error': 'No packages to optimize'})

    _ensure_unique_scan_orders(db, ss['id'])
    items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=? ORDER BY id ASC",
        (ss['id'],)
    ).fetchall()
    db.close()

    # Build packages list
    packages = []
    for item in items:
        packages.append({
            'id': item['id'],
            'tracking': item['tracking'],
            'name': item['customer_name'],
            'address': item['address'],
            'lat': item['dest_lat'],
            'lng': item['dest_lng'],
        })

    geocoded = [p for p in packages if p['lat'] and p['lng']]
    ungeoced = [p for p in packages if not (p['lat'] and p['lng'])]

    # Run optimization ONCE — group buildings, honor start/end depot
    sorted_pkgs, total_dist_m, total_dur_s = build_optimized_route(geocoded, start=start, end=end)

    # Add ungeocoded at end
    seen_ids = {p['id'] for p in sorted_pkgs}
    n = len(sorted_pkgs)
    for p in ungeoced:
        if p['id'] not in seen_ids:
            p.update({'zone_letter':'?', 'zone_num':0, 'zone_label_full':'?',
                      'zone_color':'#6b7280', 'zone_emoji':'⚪',
                      'bag_num':0, 'bag_label':'?',
                      'delivery_order': n + 1,
                      'load_position': 0})
            sorted_pkgs.append(p)
            n += 1

    # Assign vehicle zones
    sorted_pkgs = assign_vehicle_zones(sorted_pkgs, vehicle_type)
    order_map = _scan_order_map(items)
    _attach_scan_orders(sorted_pkgs, order_map)

    # Compute centroids for locked zone reference
    centroids = compute_centroids(sorted_pkgs)

    # Build loading plan — zones that deliver LAST load FIRST (deepest in car)
    zone_summary = {}
    for p in sorted_pkgs:
        letter = p.get('zone_letter', '?')
        if letter == '?':
            continue
        if letter not in zone_summary:
            zone_summary[letter] = {
                'letter': letter,
                'count': 0,
                'color': p.get('zone_color', '#6b7280'),
                'emoji': p.get('zone_emoji', '⚪'),
                'vehicle_spot': p.get('vehicle_zone_label', ''),
                'vehicle_desc': p.get('vehicle_zone_desc', ''),
            }
        zone_summary[letter]['count'] += 1

    # Loading order: last delivery zone loads first (deepest)
    letters_in_delivery_order = sorted(zone_summary.keys())  # A=first delivery, last letter=last delivery
    loading_plan = []
    for i, letter in enumerate(reversed(letters_in_delivery_order)):
        z = dict(zone_summary[letter])
        z['load_order'] = i + 1
        z['n_bags'] = math.ceil(z['count'] / BAG_SIZE)
        loading_plan.append(z)

    # Lock zones permanently in DB + persist per-item zone/order
    db2 = get_db()
    _persist_optimized_scan_items(db2, ss['id'], sorted_pkgs)
    db2.execute(
        """UPDATE scan_sessions
           SET zones_locked=1, zone_centroids=?, locked_at=?, phase='optimized',
               route_start_lat=?, route_start_lng=?, route_end_lat=?, route_end_lng=?, route_end_mode=?
           WHERE id=?""",
        (
            json.dumps(centroids), datetime.now().isoformat(),
            start['lat'] if start else None, start['lng'] if start else None,
            end['lat'] if end else None, end['lng'] if end else None,
            end_mode,
            ss['id'],
        )
    )
    db2.commit()
    db2.close()

    route_miles = round(total_dist_m * 0.000621371, 2) if total_dist_m > 0 else None
    route_drive_mins = round(total_dur_s / 60, 1) if total_dur_s > 0 else None

    return jsonify({
        'ok': True,
        'total': len(sorted_pkgs),
        'geocoded': len(geocoded),
        'n_zones': len(zone_summary),
        'loading_plan': loading_plan,
        'route_miles': route_miles,
        'route_drive_mins': route_drive_mins,
        'vehicle_type': vehicle_type,
        'vehicle_label': VEHICLE_LABELS.get(vehicle_type, 'Vehicle'),
        'items': sorted_pkgs,
    })


@app.route('/driver/vehicle-setup', methods=['POST'])
def vehicle_setup():
    """Save driver's vehicle type preference."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    vehicle_type = data.get('vehicle_type', '').strip()
    if vehicle_type not in VEHICLE_ZONES:
        return jsonify({'ok': False, 'error': 'Invalid vehicle type'})
    db = get_db()
    db.execute("UPDATE drivers SET vehicle_type=? WHERE id=?", (vehicle_type, session['driver_id']))
    db.commit()
    db.close()
    session['vehicle_type'] = vehicle_type
    return jsonify({'ok': True, 'vehicle_type': vehicle_type, 'label': VEHICLE_LABELS.get(vehicle_type)})


@app.route('/driver/vehicle-setup', methods=['GET'])
def vehicle_setup_get():
    """Return current vehicle type + all options."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    db = get_db()
    driver = db.execute("SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    db.close()
    vehicle_type = (driver['vehicle_type'] if driver and driver['vehicle_type'] else 'suv_midsize')
    # Fallback to suv_midsize if stored value is no longer a valid key
    if vehicle_type not in VEHICLE_LABELS:
        vehicle_type = 'suv_midsize'
    return jsonify({
        'ok':     True,
        'current': vehicle_type,
        'label':   VEHICLE_LABELS.get(vehicle_type, 'SUV'),
        'options': [{'value': k, 'label': v} for k, v in VEHICLE_LABELS.items()],
        'zones':   VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv_midsize']),
    })


@app.route('/driver/scan/validate-addresses', methods=['POST'])
def scan_validate_addresses():
    """Flag bad/informal addresses before route optimization."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    try:
        return _scan_validate_addresses_impl()
    except Exception:
        log.exception('scan_validate_addresses failed')
        return jsonify({'ok': False, 'error': 'Could not validate addresses'}), 500


def _scan_validate_addresses_impl():
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    ss = db.execute(
        "SELECT id FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today),
    ).fetchone()
    if not ss:
        db.close()
        return jsonify({'ok': True, 'flagged': [], 'total': 0})
    items = db.execute(
        "SELECT id, tracking, customer_name, address, zip_code, dest_lat, dest_lng, scan_order FROM scan_items WHERE session_id=? ORDER BY scan_order ASC, id ASC",
        (ss['id'],),
    ).fetchall()
    db.close()
    rows = [dict(i) for i in items]
    centroid = _route_centroid_from_items(rows)
    flagged = []
    issue_labels = {
        'sender_address': 'Looks like sender/warehouse address',
        'informal_street': 'Informal street — missing St/Ave/Rd',
        'not_geocoded': 'Could not verify on map',
        'far_from_route': 'Far from other stops on your route',
        'empty': 'Missing address',
    }
    for item in rows:
        issues = _address_quality_issues(
            item.get('address'), item.get('dest_lat'), item.get('dest_lng'), centroid,
        )
        if not issues:
            continue
        suggestions = []
        try:
            suggestions = infer_address_suggestions(
                tracking=item.get('tracking') or '',
                partial_address=item.get('address') or '',
                name=item.get('customer_name') or '',
                zip_code=item.get('zip_code') or '',
                limit=3,
            )
        except Exception as e:
            log.warning(f'validate-addresses suggestions failed id={item.get("id")}: {e}')
        flagged.append({
            'id': item['id'],
            'scan_order': item.get('scan_order'),
            'tracking': item.get('tracking') or '',
            'name': item.get('customer_name') or '',
            'address': item.get('address') or '',
            'issues': issues,
            'issue_labels': [issue_labels.get(i, i) for i in issues],
            'suggestions': suggestions,
        })
    return jsonify({'ok': True, 'flagged': flagged, 'total': len(rows)})


@app.route('/driver/scan/item/<int:item_id>', methods=['PATCH'])
def scan_update_item(item_id):
    """Fix an address on a scanned package (before or after optimize)."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    address = (data.get('address') or '').strip()
    name = (data.get('name') or '').strip()
    if not address:
        return jsonify({'ok': False, 'error': 'Address is required'})
    if _is_sender_address(address):
        return jsonify({'ok': False, 'error': 'That looks like a sender address — enter the delivery address'})
    zip_code = _extract_zip(address) or (data.get('zip') or '').strip()
    lat, lng = None, None
    try:
        coords = geocode_address(address)
        if coords:
            lat, lng = coords
    except Exception as e:
        log.warning(f'scan_update_item geocode: {e}')
    db = get_db()
    row = db.execute("SELECT session_id FROM scan_items WHERE id=?", (item_id,)).fetchone()
    if not row:
        db.close()
        return jsonify({'ok': False, 'error': 'Package not found'})
    unit = (data.get('unit') or '').strip().upper() or extract_unit_number(address)
    db.execute(
        """UPDATE scan_items SET address=?, customer_name=?, zip_code=?, dest_lat=?, dest_lng=?, unit=?
           WHERE id=?""",
        (address, name, zip_code, lat, lng, unit, item_id),
    )
    db.commit()
    db.close()
    tracking = (data.get('tracking') or '').strip()
    if tracking:
        upsert_label_memory(tracking, name, address, zip_code, source='fix')
    return jsonify({
        'ok': True, 'geocoded': bool(lat), 'address': address, 'name': name,
        'issues': _address_quality_issues(address, lat, lng),
    })


@app.route('/driver/scan/item/<int:item_id>/unit', methods=['POST'])
def scan_set_unit(item_id):
    """Set/fix the unit number on a scanned package (warehouse, before route build)."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    unit = (data.get('unit') or '').strip().upper()
    if not unit:
        return jsonify({'ok': False, 'error': 'Unit number is required'})
    db = get_db()
    row = db.execute("SELECT address FROM scan_items WHERE id=?", (item_id,)).fetchone()
    if not row:
        db.close()
        return jsonify({'ok': False, 'error': 'Package not found'})
    db.execute("UPDATE scan_items SET unit=? WHERE id=?", (unit, item_id))
    db.commit()
    db.close()
    remember_unit_number(row['address'], unit)
    return jsonify({'ok': True, 'unit': unit})


@app.route('/driver/scan/remove/<int:item_id>', methods=['POST'])
def scan_remove(item_id):
    if 'driver_id' not in session:
        return jsonify({'ok': False}), 401
    db = get_db()
    db.execute("DELETE FROM scan_items WHERE id=?", (item_id,))
    db.commit()
    db.close()
    return jsonify({'ok': True})


@app.route('/driver/scan/import-route', methods=['POST'])
def scan_import_route():
    """
    Import-first path: driver uploads Speed X screenshots BEFORE scanning.
    Extracts all stops, geocodes, clusters, locks zones immediately.
    From scan #1, every barcode scan is an instant lookup — no re-clustering ever.
    """
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401

    files = request.files.getlist('photos')
    if not files:
        return jsonify({'ok': False, 'error': 'No photos received'})

    db = get_db()
    ss_id = _get_or_create_scan_session(db, session['driver_id'])

    # Clear any existing items so import is a clean slate
    db.execute("DELETE FROM scan_items WHERE session_id=?", (ss_id,))
    db.execute(
        """UPDATE scan_sessions
           SET zones_locked=0, zone_centroids=NULL, prev_centroids=NULL,
               phase='scanning', status='scanning'
           WHERE id=?""",
        (ss_id,)
    )
    db.commit()

    if not _vision_available():
        db.close()
        return jsonify({'ok': False, 'error': 'Vision AI not configured — add GEMINI_API_KEY (free) on server'})

    # Extract stops from all uploaded screenshots and PDFs
    all_stops = []
    api_error = None
    for f in files:
        try:
            fname = (f.filename or '').lower()
            if fname.endswith('.pdf'):
                pdf_bytes = f.read()
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ''
                        if text.strip():
                            # Text-based PDF — parse directly
                            pdf_stops = parse_stops_from_text(text)
                            all_stops.extend(pdf_stops)
                        else:
                            # Image-based PDF (e.g. created by our screenshot converter)
                            # Render the page as an image and send to Claude
                            try:
                                pil_img = page.to_image(resolution=150).original
                                buf = io.BytesIO()
                                pil_img.save(buf, format='JPEG', quality=85)
                                page_stops = extract_stops_from_image(buf.getvalue())
                                all_stops.extend(page_stops)
                            except Exception as render_err:
                                log.warning(f'PDF page render failed: {render_err}')
                                # Can't render — tell user to upload original screenshots
                                raise ValueError(
                                    'PDF is image-only and cannot be rendered on this server. '
                                    'Please upload the original Speed X screenshots directly instead.'
                                )
            else:
                img_bytes = f.read()
                if not img_bytes:
                    continue
                stops = extract_stops_from_image(img_bytes)
                all_stops.extend(stops)
        except ValueError as ve:
            db.close()
            return jsonify({'ok': False, 'error': str(ve)})
        except Exception as e:
            api_error = f'{type(e).__name__}: {str(e)}'
            log.error(f'import-route extract error: {api_error}')

    if not all_stops:
        db.close()
        if api_error:
            return jsonify({'ok': False, 'error': f'Vision API error: {api_error}'})
        return jsonify({'ok': False, 'error': 'No stops found — make sure these are Speed X delivery screenshots showing package cards'})

    # Deduplicate by tracking number
    seen_tracking = set()
    unique_stops = []
    for s in all_stops:
        t = s.get('tracking', '').strip()
        if t and t not in seen_tracking:
            seen_tracking.add(t)
            unique_stops.append(s)
        elif not t:
            unique_stops.append(s)

    # Geocode all stops (batch)
    geocoded_stops = []
    for s in unique_stops:
        addr = s.get('address', '').strip()
        lat, lng = None, None
        if addr:
            try:
                coords = geocode_address(addr)
                if coords:
                    lat, lng = coords
            except Exception as e:
                log.warning(f'import geocode failed {addr}: {e}')
        s['lat'] = lat
        s['lng'] = lng
        geocoded_stops.append(s)

    # Insert all stops into scan_items (scan_order = sticker # written at warehouse)
    for i, s in enumerate(geocoded_stops):
        db.execute(
            """INSERT INTO scan_items
               (session_id, tracking, customer_name, address, zip_code, raw_json, dest_lat, dest_lng, scan_order)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                ss_id,
                s.get('tracking', '').strip(),
                s.get('name', '').strip(),
                s.get('address', '').strip(),
                s.get('zip', '').strip(),
                json.dumps(s),
                s.get('lat'),
                s.get('lng'),
                i + 1,
            )
        )
    db.commit()

    driver = db.execute("SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    vehicle_type = (driver['vehicle_type'] if driver and driver['vehicle_type'] else 'suv_midsize')

    db_items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=? ORDER BY id ASC", (ss_id,)
    ).fetchall()
    geo_pkgs = [
        {'id': r['id'], 'lat': r['dest_lat'], 'lng': r['dest_lng'],
         'address': r['address'], 'tracking': r['tracking'] or '',
         'name': r['customer_name'] or ''}
        for r in db_items if r['dest_lat'] and r['dest_lng']
    ]

    if len(geo_pkgs) < 1:
        db.close()
        return jsonify({
            'ok': True, 'imported': len(unique_stops),
            'geocoded': 0, 'zones_locked': False,
            'message': 'Imported but addresses could not be geocoded — check and re-import',
        })

    # Optimize + lock zones (same pipeline as manual scan → optimize)
    sorted_pkgs, _, _ = build_optimized_route(geo_pkgs)
    ungeoced = [
        {'id': r['id'], 'address': r['address'], 'tracking': r['tracking'] or '',
         'name': r['customer_name'] or ''}
        for r in db_items if not (r['dest_lat'] and r['dest_lng'])
    ]
    n = len(sorted_pkgs)
    for p in ungeoced:
        p.update({'zone_letter': '?', 'delivery_order': n + 1})
        sorted_pkgs.append(p)
        n += 1
    sorted_pkgs = assign_vehicle_zones(sorted_pkgs, vehicle_type)
    centroids = compute_centroids(sorted_pkgs)
    _persist_optimized_scan_items(db, ss_id, sorted_pkgs)

    db.execute(
        "UPDATE scan_sessions SET zones_locked=1, zone_centroids=?, locked_at=?, phase='optimized' WHERE id=?",
        (json.dumps(centroids), datetime.now().isoformat(), ss_id)
    )
    db.commit()

    zone_counts = {}
    for p in sorted_pkgs:
        l = p.get('zone_letter', '?')
        if l != '?':
            zone_counts[l] = zone_counts.get(l, 0) + 1

    db.close()
    return jsonify({
        'ok':           True,
        'imported':     len(unique_stops),
        'geocoded':     len(geo_pkgs),
        'zones_locked': True,
        'n_zones':      len(centroids),
        'zone_counts':  zone_counts,
        'centroids':    centroids,
    })


@app.route('/driver/scan/lock-zones', methods=['POST'])
def scan_lock_zones():
    """Backward-compat alias — same as /driver/scan/optimize."""
    return scan_optimize()


@app.route('/driver/scan/test-vision', methods=['GET'])
def test_vision():
    """Quick API sanity check — confirms vision provider + key are working."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    if not _vision_available():
        return jsonify({'ok': False, 'error': 'No vision key — set GEMINI_API_KEY (free at aistudio.google.com)'})
    try:
        if GEMINI_API_KEY:
            model = _resolve_gemini_models()[0]
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            r = requests.post(url, headers={
                'x-goog-api-key': GEMINI_API_KEY, 'Content-Type': 'application/json',
            }, json={
                'contents': [{'parts': [{'text': 'Reply with exactly: VISION_API_OK'}]}],
                'generationConfig': {'maxOutputTokens': 32},
            }, timeout=30)
            if not r.ok:
                return jsonify({'ok': False, 'error': r.text[:300]})
            text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            return jsonify({'ok': True, 'response': text, 'provider': 'gemini', 'model': model})
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        resp = client.messages.create(
            model='claude-haiku-4-5', max_tokens=50,
            messages=[{'role': 'user', 'content': 'Reply with exactly: VISION_API_OK'}]
        )
        return jsonify({'ok': True, 'response': resp.content[0].text.strip(),
                        'provider': 'anthropic', 'model': 'claude-haiku-4-5'})
    except Exception as e:
        return jsonify({'ok': False, 'error': f'{type(e).__name__}: {str(e)}'})

def _extract_loading_scan(img_bytes):
    """
    Vision extraction tuned for the Speed X 'Loading Scan' screen format.
    Handles the two-line address layout where unit/apt appears on line 2.
    """
    if not _vision_available():
        raise ValueError('Vision AI not configured')
    prompt = '''This is a Speed X "Loading Scan" delivery app screenshot.
Extract EVERY delivery stop visible. Return ONLY a raw JSON array, no markdown, no code blocks.

SCREEN LAYOUT:
- Each stop card has a blue left border
- Top-left: ADDRESS (often split across 2 lines due to screen width)
- Top-right: "1 parcel" label and CUSTOMER NAME in blue
- Middle: TRACKING NUMBER (starts with SPXDTW or YWORD or similar)
- Bottom-right: "Stop: ##"

CRITICAL ADDRESS PARSING RULES:
The address is split across 1 or 2 lines. Examples of how it appears:

  Line 1: "287 Alfred"          Line 2: "St,Detroit,MI,48201-3122,USA"
  → Street = "287 Alfred St", City = "Detroit", State = "MI", ZIP = "48201"

  Line 1: "124 Alfred St"       Line 2: "206,DETROIT,MI,48201,USA"
  → Street = "124 Alfred St", Unit = "206", City = "Detroit", State = "MI", ZIP = "48201"
  NOTE: When line 2 starts with a NUMBER before a city name, that number is the UNIT/APT.

  Line 1: "66 Winder St Apt"    Line 2: "338,Detroit,MI,48201,USA"
  → Street = "66 Winder St", Unit = "338", City = "Detroit", ZIP = "48201"

  Line 1: "3402 Brush St Apt"   Line 2: "5,Detroit,MI,48201,USA"
  → Street = "3402 Brush St", Unit = "5", City = "Detroit", ZIP = "48201"

  Line 1: "3439 woodward ave apt"  Line 2: "409,DETROIT,MI,48201-2791,USA"
  → Street = "3439 Woodward Ave", Unit = "409", City = "Detroit", ZIP = "48201"
  NOTE: "apt" / "Apt" / "Apartment" is a LABEL, not the unit. The number after it is the unit.

  Line 1: "3501 woodward ave Apartment"  Line 2: "531,detroit,MI,48201,USA"
  → Street = "3501 Woodward Ave", Unit = "531", City = "Detroit", ZIP = "48201"

  Line 1: "2900 Brush St"       Line 2: "225,DETROIT,MI,48201-3156,U..."
  → Street = "2900 Brush St", Unit = "225", City = "Detroit", ZIP = "48201"

  Line 1: "4830 Cass Ave apt"   Line 2: "324,Detroit,MI,48201,USA"
  → Street = "4830 Cass Ave", Unit = "324", City = "Detroit", ZIP = "48201"

RULE: If line 2 starts with digits followed by a comma and then a city name → those digits = unit number.
RULE: Use only the 5-digit ZIP (drop anything after a dash, e.g. 48201-3122 → 48201).
RULE: Normalize city to Title Case (Detroit not DETROIT).
RULE: Do NOT include "USA" or ",USA" in the address field.
RULE: Truncated names (ending in "...") — include what is visible.
RULE: Tracking numbers — copy EXACTLY, full length (e.g. SPXDTW013662605280007185 or YWORD010179392388).

OUTPUT FORMAT — JSON array of objects:
[
  {
    "stop_num": "46",
    "address": "624 Eliot St, Detroit, MI 48201",
    "unit": "",
    "name": "Joann Whern",
    "tracking": "YWORD010179392388"
  },
  {
    "stop_num": "48",
    "address": "314 Elliot St, Detroit, MI 48201",
    "unit": "4",
    "name": "Abhigail Ash",
    "tracking": "SPXDTW013662605260017951"
  }
]

Include EVERY stop card visible on screen. Do not skip any.'''
    text = _vision_extract_text(prompt, img_bytes, max_tokens=4096)
    log.info(f'[loading_scan] vision raw (first 500): {text[:500]}')
    try:
        stops = _parse_json_response(text, expect='array')
    except (json.JSONDecodeError, ValueError):
        log.warning(f'[loading_scan] No JSON array in response: {text[:300]}')
        return []
    _UNIT_WORDS = {'apt', 'apartment', 'unit', 'suite', 'ste', 'floor', 'fl', '#', 'no', 'num', 'usa', 'u'}
    result = []
    for s in stops:
        addr = (s.get('address') or '').strip()
        if not addr:
            continue
        raw_unit = str(s.get('unit') or '').strip()
        unit = raw_unit if raw_unit.lower() not in _UNIT_WORDS and raw_unit != '' else ''
        result.append({
            'address':  addr,
            'unit':     unit,
            'name':     (s.get('name') or '').strip(),
            'tracking': (s.get('tracking') or '').strip(),
            'stop_num': str(s.get('stop_num') or '').strip(),
            'phone':    '',
        })
    log.info(f'[loading_scan] Extracted {len(result)} stops')
    return result


@app.route('/driver/scan/screenshots-to-pdf', methods=['POST'])
def screenshots_to_pdf():
    """Convert uploaded images directly into a multi-page PDF — one image per page."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401

    files = request.files.getlist('photos')
    if not files:
        return jsonify({'ok': False, 'error': 'No images provided'})

    from PIL import Image as _PilImage
    import io as _io

    images = []
    for f in files:
        try:
            raw = f.read()
            if not raw:
                continue
            img = _PilImage.open(_io.BytesIO(raw))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            log.warning(f'screenshots-to-pdf: skipping unreadable image: {e}')

    if not images:
        return jsonify({'ok': False, 'error': 'No valid images found — make sure you uploaded JPG or PNG files'})

    try:
        buf = _io.BytesIO()
        images[0].save(
            buf,
            format='PDF',
            save_all=True,
            append_images=images[1:],
            resolution=150
        )
        buf.seek(0)
        today_file = datetime.now().strftime('%Y%m%d')
        return send_file(
            buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'speedx_route_{today_file}.pdf'
        )
    except Exception as e:
        log.error(f'screenshots-to-pdf error: {traceback.format_exc()}')
        return jsonify({'ok': False, 'error': f'PDF build failed: {str(e)}'})


@app.route('/driver/scan/clear', methods=['POST'])
def scan_clear():
    if 'driver_id' not in session:
        return jsonify({'ok': False}), 401
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    ss = db.execute(
        "SELECT id FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()
    if ss:
        db.execute("DELETE FROM scan_items WHERE session_id=?", (ss['id'],))
        db.execute(
            "UPDATE scan_sessions SET status='cleared', zones_locked=0, zone_centroids=NULL, prev_centroids=NULL, phase='scanning' WHERE id=?",
            (ss['id'],)
        )
        db.commit()
    db.close()
    return jsonify({'ok': True})



@app.route('/driver/scan/quick-navigate', methods=['POST'])
def scan_quick_navigate():
    """
    Scan a package label -> geocode -> create stop -> drop into in-app navigation.
    Keeps ALL data inside UNIT: precise coords, delivery tracking, address intel.
    No Apple Maps. No handing off. Everything stays in the pipeline.
    """
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data    = request.get_json() or {}
    address  = (data.get('address') or '').strip()
    tracking = (data.get('tracking') or '').strip()
    name     = (data.get('name') or '').strip()
    if not address:
        return jsonify({'ok': False, 'error': 'No address provided'})

    db   = get_db()
    today = datetime.now().strftime('%Y-%m-%d')

    # Geocode immediately — this is what captures the precise coord for address_intel
    lat, lng = None, None
    try:
        coords = geocode_address(address)
        if coords:
            lat, lng = coords
            # Write to address_intel now — even before delivery is confirmed
            upsert_address_intel(address, lat, lng)
    except Exception as ge:
        log.warning(f'quick-navigate geocode failed: {ge}')

    # Use or create today's route for this driver
    route = db.execute(
        "SELECT * FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()

    if not route:
        route_name = f"Quick Route {today}"
        if USE_PG:
            route_id = db.execute(
                "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (%s,%s,%s,%s) RETURNING id",
                (session['driver_id'], session['driver_name'], route_name, today)
            ).fetchone()['id']
        else:
            db.execute(
                "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (?,?,?,?)",
                (session['driver_id'], session['driver_name'], route_name, today)
            )
            db.commit()
            route_id = db.execute(
                "SELECT id FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
                (session['driver_id'], today)
            ).fetchone()['id']
    else:
        route_id = route['id']
    db.commit()

    # Get next stop number
    last = db.execute(
        "SELECT MAX(stop_number) as mx FROM stops WHERE route_id=?", (route_id,)
    ).fetchone()
    stop_num = (last['mx'] or 0) + 1

    import secrets as _sec
    token = _sec.token_urlsafe(12)

    db.execute(
        """INSERT INTO stops
           (route_id, stop_number, address, customer_name, tracking, dest_lat, dest_lng, status, token)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (route_id, stop_num, address, name, tracking, lat, lng, 'en_route', token)
    )
    db.commit()

    stop_id = db.execute(
        "SELECT id FROM stops WHERE route_id=? AND stop_number=? LIMIT 1",
        (route_id, stop_num)
    ).fetchone()['id']
    db.close()

    return jsonify({
        'ok':      True,
        'stop_id': stop_id,
        'redirect': url_for('stop_active', stop_id=stop_id),
        'geocoded': bool(lat and lng),
    })

@app.route('/driver/scan/build-route', methods=['POST'])
def scan_build_route():
    """Create route from locked scan session using optimized stop order."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    ss = db.execute(
        "SELECT * FROM scan_sessions WHERE driver_id=? AND date=? AND status='scanning' ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()
    if not ss:
        db.close()
        return jsonify({'ok': False, 'error': 'No scan session found'})
    ss = _unlock_stale_scan_session(db, ss)
    if not _ss_val(ss, 'zones_locked', 0):
        db.close()
        return jsonify({'ok': False, 'error': 'Tap OPTIMIZE to lock zones before building route'})
    ss_id = ss['id']
    items = db.execute(
        """SELECT * FROM scan_items WHERE session_id=?
           ORDER BY COALESCE(delivery_order, 9999), id ASC""",
        (ss_id,)
    ).fetchall()
    if not items:
        db.close()
        return jsonify({'ok': False, 'error': 'No packages scanned yet'})

    driver_row = db.execute(
        "SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)
    ).fetchone()
    vehicle_type = _ss_val(driver_row, 'vehicle_type', 'suv_midsize') or 'suv_midsize'
    pkg_meta = {p['id']: p for p in reconstruct_scan_pkg_meta(items, vehicle_type)}
    centroids = compute_centroids(list(pkg_meta.values()))
    stored_cents = json.loads(ss['zone_centroids']) if _ss_val(ss, 'zone_centroids') else centroids

    # Create a new route
    route_name = f'Scan Route {today}'
    db.execute(
        "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (?,?,?,?)",
        (session['driver_id'], session['driver_name'], route_name, today)
    )
    db.commit()
    route = db.execute(
        "SELECT id FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
        (session['driver_id'], today)
    ).fetchone()
    route_id = route['id']

    try:
        db.execute(
            "UPDATE routes SET zone_centroids=? WHERE id=?",
            (json.dumps(stored_cents or centroids), route_id)
        )
    except Exception:
        pass

    # Geocode items missing coords, then group same-building into one delivery stop
    geocoded = 0
    failed_addresses = []
    for item in items:
        if not item['dest_lat'] and item['address']:
            coords = geocode_address(item['address'])
            if coords:
                lat, lng = coords
                geocoded += 1
                db.execute(
                    "UPDATE scan_items SET dest_lat=?, dest_lng=? WHERE id=?",
                    (lat, lng, item['id'])
                )
                item = dict(item)
                item['dest_lat'] = lat
                item['dest_lng'] = lng
            else:
                failed_addresses.append(item['address'])

    stop_groups = []
    group_index = {}
    for item in items:
        meta = pkg_meta.get(item['id'], {})
        lat = _ss_val(item, 'dest_lat') or meta.get('lat')
        lng = _ss_val(item, 'dest_lng') or meta.get('lng')
        gkey = meta.get('stop_group_key') or _building_group_key(item['address'], lat, lng)
        placed = False
        for gi, g in enumerate(stop_groups):
            if gkey and g['key'] == gkey:
                g['items'].append((item, meta))
                placed = True
                break
            if lat and lng and g.get('lat') and g.get('lng'):
                try:
                    if geodesic((lat, lng), (g['lat'], g['lng'])).meters <= BUILDING_GROUP_METERS:
                        if _street_base(item['address']) == _street_base(g['address']):
                            g['items'].append((item, meta))
                            placed = True
                            break
                except Exception:
                    pass
        if not placed:
            stop_groups.append({
                'key': gkey or f"item:{item['id']}",
                'address': item['address'],
                'lat': lat, 'lng': lng,
                'items': [(item, meta)],
            })

    import secrets as _sec
    stop_num = 0
    for group in stop_groups:
        stop_num += 1
        items_in = group['items']
        primary_item, primary_meta = items_in[0]
        lat = group.get('lat') or _ss_val(primary_item, 'dest_lat')
        lng = group.get('lng') or _ss_val(primary_item, 'dest_lng')
        if lat and lng:
            geocoded += 1 if len(items_in) == 1 else 0

        package_list = []
        scan_orders = []
        names = []
        trackings = []
        for it, meta in items_in:
            so = meta.get('scan_order') or _ss_val(it, 'scan_order')
            package_list.append({
                'tracking': _ss_val(it, 'tracking', ''),
                'name': _ss_val(it, 'customer_name', ''),
                'scan_order': so,
                'address': _ss_val(it, 'address', ''),
            })
            if so:
                scan_orders.append(int(so))
            nm = (_ss_val(it, 'customer_name', '') or '').strip()
            if nm:
                names.append(nm)
            tr = (_ss_val(it, 'tracking', '') or '').strip()
            if tr:
                trackings.append(tr)

        customer_name = names[0] if names else ''
        if len(names) > 1:
            customer_name = f"{names[0]} +{len(names) - 1}"
        tracking = trackings[0] if trackings else ''
        if len(trackings) > 1:
            tracking = f"{trackings[0]} (+{len(trackings) - 1})"

        primary_unit = ''
        if 'unit' in primary_item.keys() and primary_item['unit']:
            primary_unit = primary_item['unit']
        if not primary_unit:
            primary_unit = extract_unit_number(primary_item['address'])

        stickers = ', '.join(f'#{n}' for n in sorted(scan_orders))
        notes = f"{len(items_in)} packages · Stickers: {stickers}" if len(items_in) > 1 else ''
        zone_letter = primary_meta.get('zone_letter') or _ss_val(primary_item, 'zone_letter')
        scan_order = min(scan_orders) if scan_orders else stop_num
        zone_num = primary_meta.get('zone_num')
        zone_color = primary_meta.get('zone_color')
        zone_emoji = primary_meta.get('zone_emoji')
        vehicle_zone_label = primary_meta.get('vehicle_zone_label', '')
        token = _sec.token_urlsafe(12)
        pkg_count = len(items_in)
        pkg_json = json.dumps(package_list)

        try:
            db.execute(
                """INSERT INTO stops
                   (route_id, stop_number, address, unit, customer_name, tracking, notes, dest_lat, dest_lng,
                    status, token, zone_letter, scan_order, zone_num, zone_color, zone_emoji,
                    vehicle_zone_label, package_count, package_list)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (route_id, stop_num, primary_item['address'], primary_unit, customer_name, tracking, notes,
                 lat, lng, 'pending', token, zone_letter, scan_order, zone_num, zone_color,
                 zone_emoji, vehicle_zone_label, pkg_count, pkg_json)
            )
        except Exception:
            db.execute(
                """INSERT INTO stops
                   (route_id, stop_number, address, unit, customer_name, tracking, notes, dest_lat, dest_lng,
                    status, token, zone_letter)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (route_id, stop_num, primary_item['address'], primary_unit, customer_name, tracking, notes,
                 lat, lng, 'pending', token, zone_letter)
            )
    db.commit()

    # Route order already set by optimize — compute stats from locked sequence
    geocoded_stops = db.execute(
        "SELECT dest_lat, dest_lng FROM stops WHERE route_id=? AND dest_lat IS NOT NULL ORDER BY stop_number",
        (route_id,)
    ).fetchall()
    if len(geocoded_stops) >= 2:
        try:
            total_dist_m = sum(
                geodesic(
                    (geocoded_stops[i]['dest_lat'], geocoded_stops[i]['dest_lng']),
                    (geocoded_stops[i + 1]['dest_lat'], geocoded_stops[i + 1]['dest_lng'])
                ).meters
                for i in range(len(geocoded_stops) - 1)
            )
            dist_miles = round(total_dist_m * 0.000621371, 2)
            dur_mins   = round((total_dist_m / 1000) / 25 * 60, 1)  # ~25 mph urban avg
            db.execute(
                "UPDATE routes SET est_distance_miles=?, est_duration_mins=? WHERE id=?",
                (dist_miles, dur_mins, route_id)
            )
            db.commit()
        except Exception as e:
            log.warning(f'Route stats on scan build failed: {e}')

    # Mark scan session as built
    db.execute("UPDATE scan_sessions SET status='built' WHERE id=?", (ss_id,))
    db.commit()
    db.close()
    return jsonify({
        'ok': True,
        'route_id': route_id,
        'total': len(items),
        'stops': stop_num,
        'geocoded': geocoded,
        'failed': failed_addresses,
        'redirect': url_for('route_detail', route_id=route_id)
    })


# ─── END PACKAGE SCAN ──────────────────────────────────────────

# ─── ROUTE IMPORT ──────────────────────────────────────────────

@app.route('/driver/route/new', methods=['GET', 'POST'])
def route_new():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    # Redirect GET requests to the scan page — it has the full import flow built in
    if request.method == 'GET':
        return redirect(url_for('scan_packages'))

    if request.method == 'POST':
        db = get_db()
        today = datetime.now().strftime('%Y-%m-%d')
        route_name = request.form.get('route_name', f'Route {today}')

        # Create route
        if USE_PG:
            route_id = db.execute(
                "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (%s,%s,%s,%s) RETURNING id",
                (session['driver_id'], session['driver_name'], route_name, today)
            ).fetchone()['id']
        else:
            db.execute(
                "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (?,?,?,?)",
                (session['driver_id'], session['driver_name'], route_name, today)
            )
            db.commit()
            route_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        # Parse route files (CSV, PDF, or multiple image screenshots)
        route_files = request.files.getlist('route_files')
        stops_added  = 0
        # Accumulate all stops from all files — keyed by tracking# to dedupe
        collected    = {}  # tracking_or_addr -> stop dict

        import_errors = []

        for route_file in route_files:
            if not route_file or not route_file.filename: continue
            fname = route_file.filename.lower()

            # ── IMAGE / SCREENSHOT — Claude Vision ──
            if fname.endswith(('.png', '.jpg', '.jpeg', '.heic', '.webp', '.gif')):
                try:
                    img_bytes = route_file.read()
                    if len(img_bytes) == 0:
                        import_errors.append(f'{fname}: file is empty')
                        continue
                    log.info(f'Processing image: {fname}, size: {len(img_bytes)} bytes')
                    stops_from_img = extract_stops_from_image(img_bytes)
                    log.info(f'Claude returned {len(stops_from_img)} stops from {fname}')
                    if not stops_from_img:
                        import_errors.append(f'{fname}: no stops found — make sure it is a Speed X screenshot')
                    for s in stops_from_img:
                        key = s.get('tracking') or s.get('address', '')
                        if key and key not in collected:
                            collected[key] = s
                except Exception as img_err:
                    log.error(f'Image processing error on {fname}: {img_err}')
                    import_errors.append(f'{fname}: error — {str(img_err)[:80]}')
                    continue

            # ── PDF ──
            elif fname.endswith('.pdf'):
                with pdfplumber.open(io.BytesIO(route_file.read())) as pdf:
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                for row_data in table:
                                    if not row_data or not any(row_data): continue
                                    flat     = [str(c).strip() if c else '' for c in row_data]
                                    addr     = next((c for c in flat if re.search(r'\d+.*(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl)', c, re.I)), '')
                                    tracking = next((c for c in flat if c.upper().startswith('SPX')), '')
                                    if addr:
                                        key = tracking or addr
                                        if key not in collected:
                                            collected[key] = {'address': addr, 'name': flat[1] if len(flat)>1 else '', 'tracking': tracking}
                        else:
                            text = page.extract_text() or ''
                            for s in parse_stops_from_text(text):
                                key = s.get('tracking') or s.get('address','')
                                if key and key not in collected:
                                    collected[key] = s

            # ── CSV ──
            elif fname.endswith('.csv'):
                content = route_file.read().decode('utf-8', errors='ignore')
                reader  = csv.DictReader(io.StringIO(content))
                for row in reader:
                    raw_addr = row.get('Address','').strip()
                    city     = row.get('City','').strip()
                    state    = row.get('State','').strip()
                    zipcode  = row.get('ZIP','').strip()
                    if not raw_addr: continue
                    unit = ''
                    if '#' in raw_addr:
                        parts    = raw_addr.split('#')
                        raw_addr = parts[0].strip()
                        unit     = parts[1].strip()
                    full_addr = f"{raw_addr}, {city}, {state} {zipcode}".strip(', ')
                    tracking  = row.get('Tracking Number','').strip()
                    key       = tracking or full_addr
                    if key not in collected:
                        collected[key] = {
                            'address':  full_addr,
                            'name':     row.get('Recipient','').strip(),
                            'tracking': tracking,
                            'unit':     unit,
                            'stop_num': row.get('Stop', '')
                        }

        # ── Bulk insert all collected stops, sorted by stop number ──
        sorted_stops = sorted(
            collected.values(),
            key=lambda s: int(s.get('stop_num') or 0)
        )
        for idx, s in enumerate(sorted_stops):
            full_addr = s.get('address', '')
            if not full_addr: continue
            street    = full_addr.split(',')[0].strip()
            resident  = db.execute("SELECT * FROM residents WHERE LOWER(address) LIKE LOWER(?)", (f'%{street}%',)).fetchone()
            drop_spot  = resident['drop_spot']  if resident else ''
            door_notes = resident['door_notes'] if resident else ''
            unit       = s.get('unit', '') or (resident['unit'] if resident and resident.get('unit') else '')
            # Phone priority: 1) parsed from SpeedX screenshot, 2) resident profile, 3) stop history
            phone = format_phone(s.get('phone','')) if s.get('phone') else ''
            if not phone:
                phone = resident['phone'] if resident else ''
            if not phone:
                hist = db.execute(
                    """SELECT phone, customer_name FROM stops
                       WHERE LOWER(address) LIKE LOWER(?) AND phone != ''
                       ORDER BY id DESC LIMIT 1""",
                    (f'%{street}%',)
                ).fetchone()
                if hist:
                    phone = hist['phone'] or ''
                    if not s.get('name') and hist['customer_name']:
                        s['name'] = hist['customer_name']
            # Building access
            building   = db.execute("SELECT * FROM buildings WHERE LOWER(address) LIKE LOWER(?)", (f'%{street}%',)).fetchone()
            access_note = ''
            if building:
                parts = []
                if building['access_code']:        parts.append(f"Code: {building['access_code']}")
                if building['buzzer_notes']:        parts.append(f"Buzzer: {building['buzzer_notes']}")
                if building['interior_directions']: parts.append(building['interior_directions'])
                access_note = ' | '.join(parts)
            notes     = ' | '.join(filter(None, [door_notes, access_note]))
            # Coord priority: 1) pin_corrections (human-verified), 2) address_intel cache, 3) geocode now
            correction = db.execute("SELECT lat, lng FROM pin_corrections WHERE address=?", (full_addr,)).fetchone()
            if correction:
                saved_lat, saved_lng = correction['lat'], correction['lng']
            else:
                # Check address_intel for previously geocoded coords
                intel_row = db.execute("SELECT lat, lng FROM address_intel WHERE address=?", (_normalize_addr_key(full_addr),)).fetchone()
                if intel_row and intel_row['lat']:
                    saved_lat, saved_lng = intel_row['lat'], intel_row['lng']
                else:
                    # Geocode immediately so address_intel gets populated at import time
                    _lat, _lng = geocode_address(full_addr)
                    saved_lat, saved_lng = _lat, _lng
            stop_num   = s.get('stop_num') or idx + 1
            token      = secrets.token_urlsafe(12)
            db.execute(
                "INSERT INTO stops (route_id, stop_number, address, unit, customer_name, tracking, phone, drop_spot, notes, token, dest_lat, dest_lng) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (route_id, stop_num, full_addr, unit, s.get('name',''), s.get('tracking',''), phone, drop_spot, notes, token, saved_lat, saved_lng)
            )
            stops_added += 1

        db.commit()

        if stops_added == 0:
            error_msg = 'No stops could be imported.'
            if import_errors:
                error_msg += ' Details: ' + ' | '.join(import_errors)
            elif not route_files or all(not f.filename for f in route_files):
                error_msg = 'No file was uploaded.'
            db.execute('DELETE FROM routes WHERE id=?', (route_id,))
            db.commit()
            db.close()
            return render_template('route_new.html', error=error_msg)

        db.close()
        return redirect(url_for('route_detail', route_id=route_id))

    return render_template('route_new.html')

@app.route('/driver/test-import', methods=['POST'])
def test_import():
    """Diagnostic endpoint — returns raw Claude output for an uploaded screenshot."""
    if 'driver_id' not in session:
        return jsonify({'error': 'not logged in'}), 401
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'no file'}), 400
    try:
        img_bytes = f.read()
        original_size = len(img_bytes)
        compressed = compress_for_api(img_bytes)
        compressed_size = len(compressed)
        stops = extract_stops_from_image(img_bytes)
        return jsonify({
            'original_size_kb': round(original_size / 1024),
            'compressed_size_kb': round(compressed_size / 1024),
            'stops_found': len(stops),
            'stops': stops,
            'vision_key_set': _vision_available(),
            'vision_provider': 'gemini' if GEMINI_API_KEY else ('anthropic' if ANTHROPIC_KEY else 'none'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/driver/route/manual', methods=['GET', 'POST'])
def route_manual():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))

    if request.method == 'POST':
        db = get_db()
        today = datetime.now().strftime('%Y-%m-%d')
        route_name = request.form.get('route_name', f'Route {today}')
        if USE_PG:
            route_id = db.execute(
                "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (%s,%s,%s,%s) RETURNING id",
                (session['driver_id'], session['driver_name'], route_name, today)
            ).fetchone()['id']
        else:
            db.execute(
                "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (?,?,?,?)",
                (session['driver_id'], session['driver_name'], route_name, today)
            )
            db.commit()
            route_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.commit()

        addresses = request.form.getlist('address')
        phones    = request.form.getlist('phone')
        names     = request.form.getlist('name')
        units     = request.form.getlist('unit')

        for i, addr in enumerate(addresses):
            if not addr.strip(): continue
            token = secrets.token_urlsafe(12)
            clean_addr = addr.strip()
            street_m   = clean_addr.split(',')[0].strip()
            # Auto-fill from residents table
            resident_m = db.execute(
                "SELECT * FROM residents WHERE LOWER(address) LIKE LOWER(?)", (f'%{street_m}%',)
            ).fetchone()
            auto_phone = phones[i].strip() if i < len(phones) and phones[i].strip() else ''
            auto_name  = names[i].strip()  if i < len(names)  and names[i].strip()  else ''
            auto_unit  = units[i].strip()  if i < len(units)  and units[i].strip()  else ''
            auto_drop  = ''
            auto_notes = ''
            if resident_m:
                if not auto_phone: auto_phone = resident_m['phone'] or ''
                if not auto_unit:  auto_unit  = resident_m['unit']  or ''
                auto_drop  = resident_m['drop_spot']  or ''
                auto_notes = resident_m['door_notes'] or ''
            # Fallback to stop history for phone + name
            if not auto_phone or not auto_name:
                hist_m = db.execute(
                    """SELECT phone, customer_name FROM stops
                       WHERE LOWER(address) LIKE LOWER(?) AND (phone != '' OR customer_name != '')
                       ORDER BY id DESC LIMIT 1""",
                    (f'%{street_m}%',)
                ).fetchone()
                if hist_m:
                    if not auto_phone: auto_phone = hist_m['phone'] or ''
                    if not auto_name:  auto_name  = hist_m['customer_name'] or ''
            # Building access codes
            building_m = db.execute(
                "SELECT * FROM buildings WHERE LOWER(address) LIKE LOWER(?)", (f'%{street_m}%',)
            ).fetchone()
            if building_m:
                parts = []
                if building_m['access_code']:  parts.append(f"Code: {building_m['access_code']}")
                if building_m['buzzer_notes']: parts.append(f"Buzzer: {building_m['buzzer_notes']}")
                if building_m['interior_directions']: parts.append(building_m['interior_directions'])
                if parts: auto_notes = ' | '.join(filter(None, [auto_notes, ' | '.join(parts)]))
            correction = db.execute(
                "SELECT lat, lng FROM pin_corrections WHERE address=?",
                (clean_addr,)
            ).fetchone()
            saved_lat = correction['lat'] if correction else None
            saved_lng = correction['lng'] if correction else None
            db.execute(
                "INSERT INTO stops (route_id, stop_number, address, unit, customer_name, phone, drop_spot, notes, token, dest_lat, dest_lng) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (route_id, i+1, clean_addr,
                 auto_unit, auto_name,
                 format_phone(auto_phone) if auto_phone else '',
                 auto_drop, auto_notes, token, saved_lat, saved_lng)
            )
        db.commit()
        db.close()
        return redirect(url_for('route_detail', route_id=route_id))

    return render_template('route_manual.html')

@app.route('/driver/route/<int:route_id>')
def route_detail(route_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    route = db.execute("SELECT * FROM routes WHERE id=?", (route_id,)).fetchone()
    stops = db.execute("SELECT * FROM stops WHERE route_id=? ORDER BY stop_number", (route_id,)).fetchall()
    driver_row = db.execute(
        "SELECT pay_rate, vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)
    ).fetchone()
    vehicle_type = _ss_val(driver_row, 'vehicle_type', 'suv_midsize') or 'suv_midsize'
    stops, zone_summary, zone_centroids, current_zone, next_stop = build_route_zone_context(
        stops, route, vehicle_type, db
    )
    db.close()
    pay_rate   = float(driver_row['pay_rate']) if driver_row and driver_row['pay_rate'] else 1.50
    total      = len(stops)
    with_phone = sum(1 for s in stops if s.get('phone'))
    delivered  = sum(1 for s in stops if s.get('status') == 'delivered')
    potential  = round(total * pay_rate, 2)
    earned     = round(delivered * pay_rate, 2)
    return render_template('route_detail.html', route=route, stops=stops, total=total,
                           with_phone=with_phone, mapbox_token=MAPBOX_TOKEN,
                           pay_rate=pay_rate, potential=potential, earned=earned,
                           delivered=delivered,
                           zone_summary=zone_summary, zone_centroids=zone_centroids,
                           current_zone=current_zone, next_stop=next_stop)

@app.route('/driver/route/<int:route_id>/add-stop', methods=['POST'])
def route_add_stop(route_id):
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    data    = request.get_json()
    address = data.get('address', '').strip()
    name    = data.get('name', '').strip()
    phone   = format_phone(data.get('phone', '').strip()) if data.get('phone', '').strip() else ''
    if not address:
        return jsonify({'ok': False, 'error': 'Address required'})
    db = get_db()
    # Get next stop number
    last = db.execute("SELECT MAX(stop_number) as mx FROM stops WHERE route_id=?", (route_id,)).fetchone()
    next_num = (last['mx'] or 0) + 1
    # Check resident profile
    street    = address.split(',')[0].strip()
    resident  = db.execute("SELECT * FROM residents WHERE address LIKE ?", (f'%{street}%',)).fetchone()
    if not phone and resident:   phone     = resident['phone'] or ''
    drop_spot  = resident['drop_spot']  if resident else ''
    door_notes = resident['door_notes'] if resident else ''
    # Pin correction
    correction = db.execute("SELECT lat, lng FROM pin_corrections WHERE address=?", (address,)).fetchone()
    saved_lat  = correction['lat'] if correction else None
    saved_lng  = correction['lng'] if correction else None
    token      = secrets.token_urlsafe(12)
    db.execute(
        "INSERT INTO stops (route_id, stop_number, address, customer_name, phone, drop_spot, notes, token, dest_lat, dest_lng) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (route_id, next_num, address, name, phone, drop_spot, door_notes, token, saved_lat, saved_lng)
    )
    db.commit()
    db.close()
    return jsonify({'ok': True})

@app.route('/driver/route/<int:route_id>/stop/<int:stop_id>/phone', methods=['POST'])
def update_stop_phone(route_id, stop_id):
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    phone = format_phone(request.form.get('phone', '').strip())
    db = get_db()
    db.execute("UPDATE stops SET phone=? WHERE id=? AND route_id=?", (phone, stop_id, route_id))
    db.commit()
    db.close()
    return jsonify({'ok': True, 'phone': phone})

@app.route('/driver/stop/<int:stop_id>/edit', methods=['GET', 'POST'])
def stop_edit(stop_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop:
        db.close()
        return redirect(url_for('driver_dashboard'))

    if request.method == 'POST':
        address   = request.form.get('address', '').strip()
        unit      = request.form.get('unit', '').strip()
        name      = request.form.get('name', '').strip()
        phone     = format_phone(request.form.get('phone', '').strip()) if request.form.get('phone', '').strip() else ''
        notes     = request.form.get('notes', '').strip()
        drop_spot = request.form.get('drop_spot', '').strip()
        pin_lat   = request.form.get('pin_lat', '').strip()
        pin_lng   = request.form.get('pin_lng', '').strip()

        # Re-geocode if address changed
        lat, lng = stop['dest_lat'], stop['dest_lng']
        if address != stop['address']:
            lat, lng = geocode_address(address)
        # Override with manually dragged pin if provided
        if pin_lat and pin_lng:
            try:
                lat, lng = float(pin_lat), float(pin_lng)
                # Save pin correction permanently
                db.execute('''
                    INSERT INTO pin_corrections (address, lat, lng, corrected_by, corrected_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(address) DO UPDATE SET
                        lat=excluded.lat, lng=excluded.lng,
                        corrected_by=excluded.corrected_by,
                        corrected_at=excluded.corrected_at
                ''', (address, lat, lng, session.get('driver_name','driver'), datetime.now().isoformat()))
                upsert_address_intel(address, lat, lng)
            except (ValueError, TypeError):
                pass

        db.execute(
            "UPDATE stops SET address=?, unit=?, customer_name=?, phone=?, notes=?, drop_spot=?, dest_lat=?, dest_lng=?, status='pending', approach_sms_sent=0 WHERE id=?",
            (address, unit, name, phone, notes, drop_spot, lat, lng, stop_id)
        )

        # Persist customer preferences to residents table for future routes
        if phone or drop_spot or notes:
            street = address.split(',')[0].strip()
            existing_res = db.execute(
                "SELECT id FROM residents WHERE LOWER(address) LIKE LOWER(?)", (f'%{street}%',)
            ).fetchone()
            if existing_res:
                db.execute(
                    "UPDATE residents SET customer_name=?, phone=COALESCE(NULLIF(?,\'\'),phone), drop_spot=COALESCE(NULLIF(?,\'\'),drop_spot), door_notes=COALESCE(NULLIF(?,\'\'),door_notes) WHERE id=?",
                    (name or None, phone or None, drop_spot or None, notes or None, existing_res['id'])
                )
            elif phone and address:
                db.execute(
                    "INSERT OR IGNORE INTO residents (address, unit, phone, customer_name, drop_spot, door_notes) VALUES (?,?,?,?,?,?)",
                    (address, unit, phone, name, drop_spot, notes)
                )

        db.commit()
        route_id = stop['route_id']
        db.close()
        return redirect(url_for('route_detail', route_id=route_id))

    route_id = stop['route_id']
    # Load existing resident preferences for this address
    street = (stop['address'] or '').split(',')[0].strip()
    resident = db.execute(
        "SELECT * FROM residents WHERE LOWER(address) LIKE LOWER(?)", (f'%{street}%',)
    ).fetchone() if street else None
    db.close()
    return render_template('stop_edit.html', stop=stop, route_id=route_id,
                           resident=resident, mapbox_token=MAPBOX_TOKEN)

@app.route('/driver/route/<int:route_id>/blast', methods=['POST'])
def route_blast(route_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    route = db.execute("SELECT * FROM routes WHERE id=?", (route_id,)).fetchone()
    stops = db.execute(
        "SELECT * FROM stops WHERE route_id=? AND phone != '' AND phone IS NOT NULL ORDER BY stop_number",
        (route_id,)
    ).fetchall()

    sent = 0
    failed = 0
    for stop in stops:
        if stop['sms_blast_sent']: continue
        track_url = f"{get_base_url()}/track/{stop['token']}"
        name_part = f"Hi {stop['customer_name'].split()[0]}! " if stop['customer_name'] else "Hi! "
        msg = (f"{name_part}Your SpeedX delivery is out today. "
               f"Your driver will notify you when they're heading to your stop.\n"
               f"Track here: {track_url}")
        mms_img = f"{get_base_url()}/static/speedx_mms.jpg" if (TWILIO_SID and not TEXTBELT_KEY) else None
        ok, _ = send_sms(format_phone(stop['phone']), msg, media_url=mms_img)
        if ok:
            db.execute("UPDATE stops SET sms_blast_sent=1 WHERE id=?", (stop['id'],))
            sent += 1
        else:
            failed += 1

    db.execute("UPDATE routes SET blast_sent=1, blast_sent_at=? WHERE id=?",
               (datetime.now().isoformat(), route_id))
    db.commit()
    db.close()
    return redirect(url_for('route_detail', route_id=route_id, blast_sent=sent, blast_failed=failed))

# ─── PER-STOP DELIVERY ─────────────────────────────────────────

@app.route('/driver/stop/<int:stop_id>/start', methods=['POST'])
def stop_start(stop_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop:
        db.close()
        return redirect(url_for('driver_dashboard'))

    # Geocode if needed
    if not stop['dest_lat']:
        lat, lng = geocode_address(stop['address'])
        db.execute("UPDATE stops SET dest_lat=?, dest_lng=?, status='en_route' WHERE id=?",
                   (lat, lng, stop_id))
    else:
        db.execute("UPDATE stops SET status='en_route' WHERE id=?", (stop_id,))

    db.commit()
    db.close()
    return redirect(url_for('stop_active', stop_id=stop_id))

@app.route('/driver/stop/<int:stop_id>')
def stop_active(stop_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop:
        db.close()
        return redirect(url_for('driver_dashboard'))
    route = db.execute("SELECT * FROM routes WHERE id=?", (stop['route_id'],)).fetchone()
    driver_row = db.execute(
        "SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)
    ).fetchone()
    vehicle_type = _ss_val(driver_row, 'vehicle_type', 'suv_midsize') or 'suv_midsize'
    all_stops = db.execute(
        "SELECT * FROM stops WHERE route_id=? ORDER BY stop_number", (stop['route_id'],)
    ).fetchall()
    stops_enriched, zone_summary, zone_centroids, current_zone, next_stop = build_route_zone_context(
        all_stops, route, vehicle_type, db
    )
    stop_dict = next((s for s in stops_enriched if s['id'] == stop_id), dict(stop))
    zone_letter = stop_dict.get('zone_letter')
    zone_stops = [s for s in stops_enriched if s.get('zone_letter') == zone_letter] if zone_letter else []
    zone_position = next((i + 1 for i, s in enumerate(zone_stops) if s['id'] == stop_id), 0)
    zone_delivered = sum(1 for s in zone_stops if s.get('status') == 'delivered')
    # Lazy geocode — if no pin yet, geocode now so map loads correctly
    if not stop['dest_lat']:
        lat, lng = geocode_address(stop['address'])
        if lat and lng:
            db.execute("UPDATE stops SET dest_lat=?, dest_lng=? WHERE id=?", (lat, lng, stop_id))
            db.commit()
            stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
            stop_dict = next((s for s in stops_enriched if s['id'] == stop_id), dict(stop))
    db.close()
    # Building memory: quick-pick units when the unit number is missing
    known_units = []
    multi_unit = False
    if not stop['unit']:
        try:
            known_units = get_known_units(stop['address'])
            multi_unit = bool(known_units) or is_known_multi_unit(stop['address'])
        except Exception as _e:
            log.warning(f'[building_memory] stop_active lookup failed: {_e}')
    return render_template('stop_active.html', stop=stop, stop_meta=stop_dict,
        zone_summary=zone_summary, zone_stops=zone_stops,
        zone_position=zone_position, zone_delivered=zone_delivered,
        current_zone=current_zone, gmaps_key=GOOGLE_MAPS_KEY, mapbox_token=MAPBOX_TOKEN,
        known_units=known_units, multi_unit=multi_unit)

@app.route('/driver/stop/<int:stop_id>/pin', methods=['POST'])
def stop_pin(stop_id):
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    lat, lng = data.get('lat'), data.get('lng')
    if not lat or not lng:
        return jsonify({'error': 'missing coords'}), 400
    db = get_db()
    stop = db.execute("SELECT address FROM stops WHERE id=?", (stop_id,)).fetchone()
    db.execute("UPDATE stops SET dest_lat=?, dest_lng=?, approach_sms_sent=0 WHERE id=?", (lat, lng, stop_id))
    if stop:
        # Save to pin_corrections (survives future routes)
        db.execute('''
            INSERT INTO pin_corrections (address, lat, lng, corrected_by, corrected_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                lat=excluded.lat, lng=excluded.lng,
                corrected_by=excluded.corrected_by,
                corrected_at=excluded.corrected_at
        ''', (stop['address'], lat, lng, session.get('driver_name', 'driver'), datetime.now().isoformat()))
        # Sync human-verified coords to address_intel (highest quality signal)
        try:
            upsert_address_intel(stop['address'], lat, lng)
            log.info(f'[address_intel] Pin correction saved: {stop["address"]} -> {lat:.5f},{lng:.5f}')
        except Exception as _e:
            log.warning(f'[address_intel] pin sync failed: {_e}')
    db.commit()
    db.close()
    return jsonify({'ok': True})

def _finalize_stop_delivery(db, stop, stop_id, now_iso):
    """Shared post-delivery bookkeeping (address intel, building memory, route timing)."""
    route_id = stop['route_id'] if stop else None
    if stop and stop['address']:
        try:
            zone_letter = stop['zone_letter'] if 'zone_letter' in stop.keys() else None
            record_address_delivery(stop['address'], zone_letter=zone_letter)
        except Exception as _ae:
            log.warning(f'[address_intel] delivery record failed: {_ae}')
        if stop['unit']:
            remember_unit_number(stop['address'], stop['unit'])
    if route_id:
        route = db.execute("SELECT * FROM routes WHERE id=?", (route_id,)).fetchone()
        if route and not route['first_delivery_at']:
            db.execute("UPDATE routes SET first_delivery_at=?, route_started_at=? WHERE id=?",
                       (now_iso, now_iso, route_id))
    return route_id

def _next_pending_stop_id(db, route_id):
    row = db.execute(
        """SELECT id FROM stops WHERE route_id=? AND status='pending'
           ORDER BY stop_number ASC LIMIT 1""",
        (route_id,)
    ).fetchone()
    return row['id'] if row else None

@app.route('/driver/stop/<int:stop_id>/delivered', methods=['POST'])
def stop_delivered(stop_id):
    """Mark delivered without POD photos (skip-photos escape hatch)."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop:
        db.close()
        return redirect(url_for('driver_dashboard'))
    now_iso = datetime.now().isoformat()
    db.execute("UPDATE stops SET status='delivered', delivered_at=? WHERE id=?", (now_iso, stop_id))
    db.commit()
    route_id = _finalize_stop_delivery(db, stop, stop_id, now_iso)
    db.commit()
    next_id = _next_pending_stop_id(db, route_id) if route_id else None
    db.close()
    if next_id:
        return redirect(url_for('stop_active', stop_id=next_id))
    return redirect(url_for('route_detail', route_id=route_id) if route_id else url_for('driver_dashboard'))

@app.route('/driver/stop/<int:stop_id>/deliver', methods=['POST'])
def stop_deliver(stop_id):
    """Mark delivered with POD photo thumbnails (JSON body)."""
    if 'driver_id' not in session:
        return jsonify({'error': 'not logged in'}), 401
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop:
        db.close()
        return jsonify({'error': 'stop not found'}), 404

    data = request.get_json(silent=True) or {}
    photos = [data.get('pod_photo_1'), data.get('pod_photo_2'), data.get('pod_photo_3')]
    now_iso = datetime.now().isoformat()

    try:
        db.execute(
            """UPDATE stops SET status='delivered', delivered_at=?, pod_photo_1=?, pod_photo_2=?,
               pod_photo_3=?, pod_captured_at=? WHERE id=?""",
            (now_iso, photos[0], photos[1], photos[2], now_iso, stop_id)
        )
        db.commit()
    except Exception as e:
        log.error(f'POD storage failed for stop {stop_id}: {e}')
        db.execute("UPDATE stops SET status='delivered', delivered_at=? WHERE id=?", (now_iso, stop_id))
        db.commit()

    route_id = _finalize_stop_delivery(db, stop, stop_id, now_iso)
    db.commit()
    next_id = _next_pending_stop_id(db, route_id) if route_id else None
    db.close()
    return jsonify({'ok': True, 'next_stop_id': next_id, 'route_id': route_id})

@app.route('/driver/stop/<int:stop_id>/pod')
def stop_pod(stop_id):
    """Return POD thumbnails for a stop."""
    if 'driver_id' not in session:
        return jsonify({'error': 'not logged in'}), 401
    db = get_db()
    stop = db.execute(
        "SELECT id, route_id, stop_number, address, customer_name, status, "
        "pod_photo_1, pod_photo_2, pod_photo_3, pod_captured_at FROM stops WHERE id=?",
        (stop_id,)
    ).fetchone()
    db.close()
    if not stop:
        return jsonify({'error': 'stop not found'}), 404
    return jsonify({
        'stop_id': stop['id'],
        'route_id': stop['route_id'],
        'stop_number': stop['stop_number'],
        'address': stop['address'],
        'customer_name': stop['customer_name'],
        'status': stop['status'],
        'pod_photo_1': stop['pod_photo_1'],
        'pod_photo_2': stop['pod_photo_2'],
        'pod_photo_3': stop['pod_photo_3'],
        'pod_captured_at': stop['pod_captured_at'],
        'has_pod': bool(stop['pod_photo_1'] or stop['pod_photo_2'] or stop['pod_photo_3']),
    })

@app.route('/driver/stop/<int:stop_id>/unit', methods=['POST'])
def stop_set_unit(stop_id):
    """Tap-to-add unit number from the stop_active screen. Saves to building memory."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    unit = (data.get('unit') or '').strip().upper()
    if not unit:
        return jsonify({'ok': False, 'error': 'Unit number is required'})
    db = get_db()
    stop = db.execute("SELECT address FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop:
        db.close()
        return jsonify({'ok': False, 'error': 'Stop not found'})
    db.execute("UPDATE stops SET unit=? WHERE id=?", (unit, stop_id))
    db.commit()
    db.close()
    remember_unit_number(stop['address'], unit)
    return jsonify({'ok': True, 'unit': unit})


@app.route('/driver/stop/<int:stop_id>/failed', methods=['POST'])
def stop_failed(stop_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    db.execute("UPDATE stops SET status='failed' WHERE id=?", (stop_id,))
    db.commit()
    route_id = stop['route_id'] if stop else None
    next_stop = None
    if route_id:
        next_stop = db.execute(
            """SELECT id FROM stops
               WHERE route_id=? AND status='pending'
               ORDER BY stop_number ASC LIMIT 1""",
            (route_id,)
        ).fetchone()
    db.close()
    if next_stop:
        return redirect(url_for('stop_active', stop_id=next_stop['id']))
    return redirect(url_for('route_detail', route_id=route_id) if route_id else url_for('driver_dashboard'))

@app.route('/driver/stop/<int:stop_id>/undo', methods=['POST'])
def stop_undo(stop_id):
    """Reset a failed or delivered stop back to pending so the driver can retry."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if stop:
        db.execute("UPDATE stops SET status='pending', delivered_at=NULL WHERE id=?", (stop_id,))
        db.commit()
    route_id = stop['route_id'] if stop else None
    db.close()
    return redirect(url_for('route_detail', route_id=route_id) if route_id else url_for('driver_dashboard'))

@app.route('/driver/route/<int:route_id>/eta', methods=['GET'])
def route_eta(route_id):
    """Live ETA: based on first delivery time + avg time per stop."""
    if 'driver_id' not in session:
        return jsonify({'ok': False}), 401
    db = get_db()
    route = db.execute("SELECT * FROM routes WHERE id=?", (route_id,)).fetchone()
    if not route:
        db.close()
        return jsonify({'ok': False, 'error': 'Route not found'})

    stops = db.execute(
        "SELECT * FROM stops WHERE route_id=? ORDER BY stop_number ASC", (route_id,)
    ).fetchall()
    db.close()

    total      = len(stops)
    delivered  = [s for s in stops if s['status'] == 'delivered' and s['delivered_at']]
    remaining  = [s for s in stops if s['status'] not in ('delivered', 'failed')]
    n_done     = len(delivered)
    n_remaining= len(remaining)

    result = {
        'ok':          True,
        'total':       total,
        'done':        n_done,
        'remaining':   n_remaining,
        'pct':         round(n_done / total * 100) if total else 0,
        'eta_time':    None,
        'eta_mins':    None,
        'avg_mins_per_stop': None,
        'started_at':  route['first_delivery_at'],
    }

    if n_done >= 1 and route['first_delivery_at']:
        start = datetime.fromisoformat(route['first_delivery_at'])
        now   = datetime.now()
        elapsed_mins = (now - start).total_seconds() / 60

        if n_done >= 2:
            # Average based on actual pace
            avg = elapsed_mins / n_done
        else:
            # First stop just delivered — use OSRM estimate if available (3 min/stop fallback)
            est = route['est_duration_mins']
            avg = (est / total) if est and total else 3.5

        result['avg_mins_per_stop'] = round(avg, 1)
        eta_mins = avg * n_remaining
        result['eta_mins'] = round(eta_mins)
        eta_dt = now + timedelta(minutes=eta_mins)
        result['eta_time'] = eta_dt.strftime('%I:%M %p').lstrip('0')

    return jsonify(result)


@app.route('/driver/route/<int:route_id>/clear', methods=['POST'])
def route_clear(route_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    # Verify this route belongs to the logged-in driver
    route = db.execute(
        "SELECT * FROM routes WHERE id=? AND driver_id=?",
        (route_id, session['driver_id'])
    ).fetchone()
    if route:
        db.execute("DELETE FROM stops WHERE route_id=?", (route_id,))
        db.execute("DELETE FROM routes WHERE id=?", (route_id,))
        db.commit()
    db.close()
    return redirect(url_for('driver_dashboard'))


@app.route('/driver/route/<int:route_id>/reset', methods=['POST'])
def route_reset(route_id):
    """Reset all stops back to pending without deleting the route.
    Keeps zone assignments, sticker numbers, addresses, and stop order.
    Use this to re-sort packages into bags and restart delivery from scratch."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    db = get_db()
    route = db.execute(
        "SELECT * FROM routes WHERE id=? AND driver_id=?",
        (route_id, session['driver_id'])
    ).fetchone()
    if not route:
        db.close()
        return jsonify({'ok': False, 'error': 'route not found'}), 404
    db.execute(
        """UPDATE stops
           SET status='pending',
               driver_lat=NULL, driver_lng=NULL,
               approach_sms_sent=0,
               sms_blast_sent=0
           WHERE route_id=?""",
        (route_id,)
    )
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM stops WHERE route_id=?", (route_id,)).fetchone()[0]
    db.close()
    log.info(f'[route_reset] route {route_id} reset — {count} stops back to pending')
    return jsonify({'ok': True, 'reset': count})

# ─── ADDRESS SUGGESTIONS (internal DB) ────────────────────────


@app.route('/driver/history', methods=['GET', 'POST'])
def delivery_history():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    results = []
    query = ''
    if request.method == 'POST':
        query = request.form.get('address', '').strip()
        if query:
            street = query.split(',')[0].strip()
            db = get_db()
            results = db.execute(
                """SELECT s.*, r.name as route_name, r.date as route_date,
                          res.drop_spot as res_drop_spot, res.door_notes as res_door_notes,
                          res.phone as res_phone
                   FROM stops s
                   LEFT JOIN routes r ON s.route_id = r.id
                   LEFT JOIN residents res ON s.address LIKE '%' || res.address || '%'
                                          OR res.address LIKE '%' || s.address || '%'
                   WHERE LOWER(s.address) LIKE LOWER(?) OR LOWER(s.customer_name) LIKE LOWER(?)
                   ORDER BY s.created_at DESC LIMIT 50""",
                (f'%{street}%', f'%{query}%')
            ).fetchall()
            db.close()
    return render_template('delivery_history.html', results=results, query=query)

@app.route('/api/address-suggest')
def address_suggest():
    q = request.args.get('q', '').strip()
    if len(q) < 3:
        return jsonify([])
    db = get_db()
    results = db.execute(
        """SELECT address, unit, customer_name FROM stops
           WHERE LOWER(address) LIKE LOWER(?) GROUP BY address, unit, customer_name ORDER BY MAX(id) DESC LIMIT 8""",
        (f'%{q}%',)
    ).fetchall()
    db.close()
    return jsonify([{'address': r['address'], 'unit': r['unit'] or '', 'name': r['customer_name'] or ''} for r in results])

@app.route('/api/name-suggest')
def name_suggest():
    """Search stops by customer name — returns address + unit for autofill."""
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])
    db = get_db()
    results = db.execute(
        """SELECT customer_name, address, unit, phone
           FROM stops
           WHERE LOWER(customer_name) LIKE LOWER(?)
           GROUP BY customer_name, address, unit, phone
           ORDER BY MAX(id) DESC LIMIT 8""",
        (f'%{q}%',)
    ).fetchall()
    db.close()
    return jsonify([{
        'name':    r['customer_name'],
        'address': r['address'] or '',
        'unit':    r['unit'] or '',
        'phone':   r['phone'] or ''
    } for r in results if r['customer_name']])

@app.route('/api/resident-suggest')
def resident_suggest():
    """Search residents by name — returns address + delivery prefs for manual entry autofill."""
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])
    db = get_db()
    results = db.execute(
        """SELECT address, unit, phone, drop_spot, door_notes
           FROM residents
           WHERE address LIKE ? OR unit LIKE ?
           ORDER BY id DESC LIMIT 10""",
        (f'%{q}%', f'%{q}%')
    ).fetchall()
    db.close()
    return jsonify([{
        'address':    r['address'],
        'unit':       r['unit'] or '',
        'phone':      r['phone'] or '',
        'drop_spot':  r['drop_spot'] or '',
        'door_notes': r['door_notes'] or ''
    } for r in results])

# ─── GPS API ───────────────────────────────────────────────────

@app.route('/api/location', methods=['POST'])
def update_location():
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    lat  = data.get('lat')
    lng  = data.get('lng')
    stop_id = data.get('stop_id')
    if not lat or not lng:
        return jsonify({'error': 'no coords'}), 400

    db = get_db()
    db.execute("UPDATE drivers SET current_lat=?, current_lng=?, last_seen=? WHERE id=?",
               (lat, lng, datetime.now().isoformat(), session['driver_id']))

    result = {'status': 'ok', 'sms_triggered': False, 'distance_miles': None, 'at_stop': False, 'distance_feet': None}

    if stop_id:
        db.execute("UPDATE stops SET driver_lat=?, driver_lng=? WHERE id=?", (lat, lng, stop_id))
        stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()

        if stop and stop['dest_lat']:
            distance = miles_away(lat, lng, stop['dest_lat'], stop['dest_lng'])
            result['distance_miles'] = round(distance, 2)
            result['distance_feet']  = int(distance * 5280)
            result['at_stop']        = distance <= GEOFENCE_RADIUS_MILES

            if not stop['approach_sms_sent'] and distance <= APPROACH_RADIUS_MILES and stop['phone']:
                track_url = f"{get_base_url()}/track/{stop['token']}"
                mins = max(1, int(distance * 3))
                name_part = stop['customer_name'].split()[0] if stop['customer_name'] else 'there'
                msg = (f"Hey {name_part}! Your SpeedX driver is {mins} min away"
                       f"{' — Unit ' + stop['unit'] if stop['unit'] else ''}.\n"
                       f"Track live: {track_url}")
                mms_img = f"{get_base_url()}/static/speedx_mms.jpg" if (TWILIO_SID and not TEXTBELT_KEY) else None
                ok, _ = send_sms(format_phone(stop['phone']), msg, media_url=mms_img)
                if ok:
                    db.execute("UPDATE stops SET approach_sms_sent=1 WHERE id=?", (stop_id,))
                    result['sms_triggered'] = True

    db.commit()
    db.close()
    return jsonify(result)



@app.route('/driver/route/<int:route_id>/optimize', methods=['POST'])
def optimize_route(route_id):
    """Reorder stops using nearest-neighbor from driver's current GPS, or OSRM trip."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    db = get_db()
    data = request.get_json() or {}
    driver_lat = data.get('lat')
    driver_lng = data.get('lng')

    stops = db.execute(
        """SELECT id, dest_lat, dest_lng, stop_number FROM stops
           WHERE route_id=? AND status='pending' AND dest_lat IS NOT NULL
           ORDER BY stop_number ASC""",
        (route_id,)
    ).fetchall()

    if not stops:
        db.close()
        return jsonify({'ok': False, 'error': 'No geocoded stops to optimize'})

    # Try OSRM trip optimization
    optimized_ids = []
    try:
        coords = ';'.join(f"{s['dest_lng']},{s['dest_lat']}" for s in stops)
        osrm_url = (f"http://router.project-osrm.org/trip/v1/driving/{coords}"
                    f"?roundtrip=false&source=first&destination=last&overview=false")
        resp = requests.get(osrm_url, timeout=8)
        if resp.status_code == 200:
            trip = resp.json()
            if trip.get('code') == 'Ok' and trip.get('waypoints'):
                order = sorted(trip['waypoints'], key=lambda w: w['waypoint_index'])
                optimized_ids = [stops[w['trips_index'] if 'trips_index' in w else stops.index(
                    min(stops, key=lambda s: abs(s['dest_lat'] - w['location'][1]) + abs(s['dest_lng'] - w['location'][0]))
                )]['id'] for w in order]
    except Exception as e:
        log.warning(f'OSRM optimize failed: {e}')

    # Fallback: nearest-neighbor from driver location
    if not optimized_ids:
        remaining = list(stops)
        cur_lat = driver_lat or (stops[0]['dest_lat'] if stops else 0)
        cur_lng = driver_lng or (stops[0]['dest_lng'] if stops else 0)
        while remaining:
            closest = min(remaining, key=lambda s: miles_away(cur_lat, cur_lng, s['dest_lat'], s['dest_lng']))
            optimized_ids.append(closest['id'])
            cur_lat, cur_lng = closest['dest_lat'], closest['dest_lng']
            remaining.remove(closest)

    # Renumber pending stops in optimized order (keep delivered/failed stops in place)
    delivered_count = db.execute(
        "SELECT COUNT(*) FROM stops WHERE route_id=? AND status!='pending'", (route_id,)
    ).fetchone()[0]

    for i, stop_id in enumerate(optimized_ids):
        new_num = delivered_count + i + 1
        db.execute("UPDATE stops SET stop_number=? WHERE id=?", (new_num, stop_id))
    db.commit()
    db.close()
    log.info(f'Route {route_id} optimized: {len(optimized_ids)} stops reordered')
    return jsonify({'ok': True, 'reordered': len(optimized_ids)})


@app.route('/driver/stop/<int:stop_id>/send-message', methods=['POST'])
def stop_send_message(stop_id):
    """Send a custom SMS to the customer from the stop active screen."""
    if 'driver_id' not in session:
        return jsonify({'ok': False}), 401
    data = request.get_json() or {}
    msg = data.get('message', '').strip()
    if not msg:
        return jsonify({'ok': False, 'error': 'No message'})
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    if not stop or not stop['phone']:
        db.close()
        return jsonify({'ok': False, 'error': 'No phone number for this stop'})
    ok, err = send_sms(format_phone(stop['phone']), msg)
    db.close()
    return jsonify({'ok': ok, 'error': err if not ok else None})


@app.route('/driver/test/proximity', methods=['POST'])
def test_proximity_alert():
    """Test the iMessage proximity alert — fires immediately to driver phone."""
    if 'driver_id' not in session:
        return jsonify({'ok': False}), 401
    db = get_db()
    data = request.get_json() or {}
    stop_id = data.get('stop_id')
    # Get driver phone
    driver = db.execute("SELECT * FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    if not driver or not driver['phone']:
        db.close()
        return jsonify({'ok': False, 'error': 'No phone number on your driver account'})
    # Build test message
    if stop_id:
        stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
        if stop:
            track_url = f"{get_base_url()}/track/{stop['token']}"
            customer_msg = f"Your driver is on the way! Track live \U0001F4CD {track_url}"
            test_msg = (f"\U0001F4E6 UNIT TEST — Stop #{stop['stop_number']}\n"
                       f"{stop['address'].split(',')[0] if stop['address'] else 'Test Stop'}"
                       f"{' Apt ' + stop['unit'] if stop['unit'] else ''}\n\n"
                       f"Copy for Speed X:\n{customer_msg}\n\n"
                       f"(This is a proximity alert test)")
        else:
            test_msg = "\U0001F4E6 UNIT TEST — proximity alert is working! \U00002705"
    else:
        test_msg = "\U0001F4E6 UNIT TEST — proximity alert is working! \U00002705\n\nWhen you are within 0.5 miles of a stop, this message will auto-fire with the Speed X copy text."
    # Try Twilio SMS first, fall back to iMessage
    ok, err = send_sms(format_phone(driver['phone']), test_msg)
    if not ok:
        ok = send_imessage_to_driver(driver['phone'], test_msg)
        err = None if ok else 'SMS unavailable — check Twilio/Textbelt balance'
    db.close()
    return jsonify({'ok': ok, 'sent_to': driver['phone'], 'message': test_msg if ok else (err or 'Send failed')})

# ─── QUICK LIVE SHARE (no stop/address required) ─────────────────────────────

@app.route('/driver/live/start', methods=['POST'])
def live_start():
    """Create a new quick-share live session for the logged-in driver."""
    driver_id = session.get('driver_id')
    if not driver_id:
        return jsonify({'ok': False, 'error': 'Not logged in'}), 401
    db = get_db()
    driver = db.execute("SELECT name FROM drivers WHERE id=?", (driver_id,)).fetchone()
    token = secrets.token_urlsafe(12)
    db.execute(
        "INSERT INTO live_sessions (token, driver_id, driver_name, status) VALUES (?,?,?,?)",
        (token, driver_id, driver['name'] if driver else 'Driver', 'active')
    )
    db.commit()
    db.close()
    base = get_base_url()
    return jsonify({'ok': True, 'token': token, 'url': f'{base}/live/{token}'})

@app.route('/api/live/<token>/location', methods=['POST'])
def live_update_location(token):
    """Driver pings their GPS to the live session."""
    data = request.get_json() or {}
    lat, lng = data.get('lat'), data.get('lng')
    if not lat or not lng:
        return jsonify({'ok': False}), 400
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    if not sess or sess['status'] != 'active':
        db.close()
        return jsonify({'ok': False, 'status': 'ended'}), 200
    db.execute(
        "UPDATE live_sessions SET driver_lat=?, driver_lng=?, last_seen=? WHERE token=?",
        (lat, lng, datetime.now().isoformat(), token)
    )
    db.commit()
    db.close()
    return jsonify({'ok': True, 'status': 'active'})

@app.route('/api/live/<token>')
def live_poll(token):
    """Customer polls for driver location. Logs view on first open."""
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    if not sess:
        db.close()
        return jsonify({'error': 'not found'}), 404
    # Log view — first time sets viewed_at, always increments view_count
    now = datetime.now().isoformat()
    if not sess['viewed_at']:
        db.execute("UPDATE live_sessions SET viewed_at=?, view_count=1 WHERE token=?", (now, token))
    else:
        db.execute("UPDATE live_sessions SET view_count=view_count+1 WHERE token=?", (token,))
    db.commit()
    db.close()
    return jsonify({
        'status':     sess['status'],
        'driver_lat': sess['driver_lat'],
        'driver_lng': sess['driver_lng'],
        'last_seen':  sess['last_seen'],
    })

@app.route('/api/live/<token>/status')
def live_session_status(token):
    """Driver checks if customer has opened the tracking link."""
    driver_id = session.get('driver_id')
    if not driver_id:
        return jsonify({'error': 'not logged in'}), 401
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    db.close()
    if not sess:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'status':     sess['status'],
        'viewed_at':  sess['viewed_at'],
        'view_count': sess['view_count'] or 0,
        'driver_lat': sess['driver_lat'],
        'driver_lng': sess['driver_lng'],
    })

@app.route('/driver/live/<token>/end', methods=['POST'])
def live_end(token):
    """Driver marks the live session as delivered/ended."""
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    if not sess:
        db.close()
        return jsonify({'ok': False, 'error': 'not found'}), 404
    db.execute("UPDATE live_sessions SET status=\'delivered\' WHERE token=?", (token,))
    db.commit()
    db.close()
    return jsonify({'ok': True})

@app.route('/driver/live/<token>/failed', methods=['POST'])
def live_fail(token):
    """Driver marks the live session as failed — customer gets never-miss link."""
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    if not sess:
        db.close()
        return jsonify({'ok': False, 'error': 'not found'}), 404
    db.execute("UPDATE live_sessions SET status='failed' WHERE token=?", (token,))
    db.commit()
    db.close()
    return jsonify({'ok': True})

@app.route('/live/<token>')
def live_track(token):
    """Customer-facing live tracking page."""
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    db.close()
    if not sess:
        return "Tracking session not found", 404
    return render_template('live_track.html', sess=sess)

@app.route('/live/<token>/signup', methods=['POST'])
def live_signup(token):
    """Customer signup from live tracking page (delivered or failed)."""
    db = get_db()
    sess = db.execute("SELECT * FROM live_sessions WHERE token=?", (token,)).fetchone()
    if not sess:
        db.close()
        return jsonify({'ok': False}), 404
    name      = request.form.get('name', '').strip()
    phone     = format_phone(request.form.get('phone', '').strip()) if request.form.get('phone', '').strip() else ''
    drop_spot = request.form.get('drop_spot', '').strip()
    if not name or not phone:
        db.close()
        return jsonify({'ok': False, 'error': 'Name and phone required'}), 400
    try:
        existing = db.execute("SELECT id FROM residents WHERE phone=?", (phone,)).fetchone()
        if not existing:
            db.execute(
                "INSERT INTO residents (address, unit, phone, customer_name, drop_spot, sms_consent, sms_consent_at) VALUES (?,?,?,?,?,1,?)",
                ('', '', phone, name, drop_spot, datetime.now().isoformat())
            )
            db.commit()
            log.info(f'Customer signup from live track ({sess["status"]}): {name} {phone} drop_spot={drop_spot}')
        elif drop_spot:
            db.execute("UPDATE residents SET drop_spot=?, customer_name=? WHERE phone=?", (drop_spot, name, phone))
            db.commit()
    except Exception as e:
        log.error(f'live_signup error: {e}')
        try: db._conn.rollback()
        except: pass
    db.close()
    return jsonify({'ok': True})

# ─── CUSTOMER SIGNUP FROM TRACKING PAGE ──────────────────────────────

@app.route('/track/<token>/signup', methods=['POST'])
def track_signup(token):
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE token=?", (token,)).fetchone()
    if not stop:
        db.close()
        return jsonify({'ok': False, 'error': 'Stop not found'}), 404
    name  = request.form.get('name', '').strip()
    phone = format_phone(request.form.get('phone', '').strip()) if request.form.get('phone', '').strip() else ''
    if not name or not phone:
        return jsonify({'ok': False, 'error': 'Name and phone required'}), 400
    address = stop['address']
    unit    = stop['unit'] or ''  # empty string satisfies NOT NULL constraint
    try:
        # Check if already registered at this address+phone
        existing = db.execute(
            "SELECT id FROM residents WHERE phone=? AND LOWER(address) LIKE LOWER(?)",
            (phone, f'%{address.split(",")[0].strip()}%')
        ).fetchone()
        if not existing:
            db.execute(
                "INSERT INTO residents (address, unit, phone, customer_name, sms_consent, sms_consent_at) VALUES (?,?,?,?,1,?)",
                (address, unit, phone, name, datetime.now().isoformat())
            )
            db.commit()
            log.info(f'Customer signup from track page: {name} {phone} @ {address}')
    except Exception as e:
        log.error(f'track_signup error: {e}')
        try: db._conn.rollback()
        except: pass
    db.close()
    return jsonify({'ok': True})

# ─── TRACKING PAGE (for customers) ─────────────────────────────

@app.route('/track/<token>')
def track(token):
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE token=?", (token,)).fetchone()
    db.close()
    if not stop: return "Delivery not found", 404
    return render_template('track.html', stop=stop)

@app.route('/api/track/<token>')
def track_api(token):
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE token=?", (token,)).fetchone()
    db.close()
    if not stop: return jsonify({'error': 'not found'}), 404
    distance = None
    if stop['driver_lat'] and stop['dest_lat']:
        distance = round(miles_away(stop['driver_lat'], stop['driver_lng'],
                                    stop['dest_lat'], stop['dest_lng']), 2)
    return jsonify({
        'driver_lat': stop['driver_lat'],
        'driver_lng': stop['driver_lng'],
        'dest_lat':   stop['dest_lat'],
        'dest_lng':   stop['dest_lng'],
        'status':     stop['status'],
        'address':    stop['address'],
        'unit':       stop['unit'],
        'distance_miles': distance
    })

# ─── RESIDENT ──────────────────────────────────────────────────

@app.route('/resident', methods=['GET', 'POST'])
def resident_portal():
    success = False
    if request.method == 'POST':
        db = get_db()
        db.execute(
            "INSERT INTO residents (address,unit,phone,backup_phone,drop_spot,door_notes,sms_consent,sms_consent_at) VALUES (?,?,?,?,?,?,?,?)",
            (request.form.get('address'), request.form.get('unit'),
             request.form.get('phone'),   request.form.get('backup_phone'),
             request.form.get('drop_spot'), request.form.get('door_notes'),
             1 if request.form.get('sms_consent') else 0,
             datetime.now().isoformat() if request.form.get('sms_consent') else None)
        )
        db.commit()
        db.close()
        success = True
    return render_template('resident_portal.html', success=success)

# ─── ADMIN ─────────────────────────────────────────────────────

ADMIN_PIN = os.environ.get('ADMIN_PIN', '')
if not ADMIN_PIN:
    raise RuntimeError('ADMIN_PIN env var is required')

# ─── BRUTE FORCE PROTECTION ────────────────────────────────────
import time as _time
_login_attempts = {}  # ip -> [timestamp, ...]
LOCKOUT_WINDOW = 300  # seconds
MAX_ATTEMPTS   = 5

def get_real_ip():
    """Get real client IP — works behind Cloudflare + Render proxy."""
    return (request.headers.get('CF-Connecting-IP') or
            request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or
            request.remote_addr or 'unknown')

def is_rate_limited(ip):
    now = _time.time()
    attempts = [t for t in _login_attempts.get(ip, []) if now - t < LOCKOUT_WINDOW]
    _login_attempts[ip] = attempts
    return len(attempts) >= MAX_ATTEMPTS

def record_attempt(ip):
    _login_attempts.setdefault(ip, []).append(_time.time())

def clear_attempts(ip):
    _login_attempts.pop(ip, None)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    ip = get_real_ip()
    if request.method == 'POST':
        if is_rate_limited(ip):
            return render_template('admin_login.html', error='Too many attempts. Try again in 5 minutes.')
        if request.form.get('pin', '').strip() == ADMIN_PIN:
            session['admin'] = True
            clear_attempts(ip)
            return redirect(url_for('admin'))
        record_attempt(ip)
        error = 'Wrong PIN'
    return render_template('admin_login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('index'))

@app.route('/admin/create-driver', methods=['POST'])
def admin_create_driver():
    """Create a driver directly from admin — no Stripe required (beta/test accounts)."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    name    = request.form.get('name', '').strip()
    phone   = format_phone(request.form.get('phone', '').strip())
    company = request.form.get('company', '').strip()
    is_beta = 1  # all admin-created drivers are beta/free
    if not name:
        return redirect(url_for('admin'))
    pin = str(secrets.randbelow(9000) + 1000)
    try:
        db = get_db()
        db.execute(
            "INSERT INTO drivers (name, phone, company, pin, is_beta) VALUES (?,?,?,?,?)",
            (name, phone, company, pin, is_beta)
        )
        db.commit()
        db.close()
    except Exception as e:
        log.error(f'admin_create_driver DB error: {e}')
        try: db._conn.rollback()
        except: pass
        flash(f'Error creating driver: {e}', 'beta_pin')
        return redirect(url_for('admin'))
    if phone:
        send_sms(phone, f"Your UNIT driver PIN is: {pin}\nLogin at: {get_base_url()}/driver/login")
    flash(f'Driver created — {name} | PIN: {pin} | Login: {get_base_url()}/driver/login', 'beta_pin')
    return redirect(url_for('admin'))

@app.route('/admin/driver/<int:driver_id>/assign-zips', methods=['POST'])
def admin_assign_zips(driver_id):
    """Admin: assign zip codes to a driver for today's route."""
    if not session.get('admin'):
        return jsonify({'ok': False}), 403
    data = request.get_json() or {}
    zips_raw = data.get('zips', '')
    # Normalize: comma-separated, strip spaces, numbers only
    zips = ','.join(z.strip() for z in str(zips_raw).replace(' ', '').split(',') if z.strip().isdigit() and len(z.strip()) == 5)
    db = get_db()
    db.execute("UPDATE drivers SET assigned_zips=? WHERE id=?", (zips or None, driver_id))
    db.commit()
    db.close()
    return jsonify({'ok': True, 'assigned_zips': zips})


@app.route('/admin/cleanup-drivers', methods=['POST'])
def admin_cleanup_drivers():
    """Remove duplicate driver rows — keep the most recent unique PIN."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    # Keep only the latest row per PIN
    db.execute("""
        DELETE FROM drivers WHERE id NOT IN (
            SELECT MAX(id) FROM drivers GROUP BY pin
        )
    """)
    db.commit()
    db.close()
    return redirect(url_for('admin'))

@app.route('/admin/delete-driver/<int:driver_id>', methods=['POST'])
def admin_delete_driver(driver_id):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    db.execute("DELETE FROM drivers WHERE id=?", (driver_id,))
    db.commit()
    db.close()
    return redirect(url_for('admin'))

@app.route('/admin/mark-onboarded/<int:driver_id>', methods=['POST'])
def admin_mark_onboarded(driver_id):
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    try:
        db.execute("INSERT INTO driver_onboarding (driver_id) VALUES (?)", (driver_id,))
        db.commit()
    except:
        try: db._conn.rollback()
        except: pass
    db.close()
    return redirect(url_for('admin'))

@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    drivers_list = db.execute("SELECT * FROM drivers ORDER BY created_at DESC").fetchall()
    routes     = db.execute("SELECT * FROM routes ORDER BY created_at DESC LIMIT 20").fetchall()
    buildings  = db.execute("SELECT * FROM buildings ORDER BY confirmed_count DESC").fetchall()
    deliveries = db.execute(
        """SELECT s.*, r.name as route_name, r.date as route_date, d.name as driver_name
           FROM stops s
           LEFT JOIN routes r ON s.route_id = r.id
           LEFT JOIN drivers d ON r.driver_id = d.id
           ORDER BY s.created_at DESC LIMIT 50"""
    ).fetchall()
    stats = {
        'total_deliveries': db.execute("SELECT COUNT(*) FROM stops").fetchone()[0],
        'delivered':        db.execute("SELECT COUNT(*) FROM stops WHERE status='delivered'").fetchone()[0],
        'failed':           db.execute("SELECT COUNT(*) FROM stops WHERE status='failed'").fetchone()[0],
        'sms_sent':         db.execute("SELECT COUNT(*) FROM stops WHERE approach_sms_sent=1").fetchone()[0],
        'buildings':        db.execute("SELECT COUNT(*) FROM buildings").fetchone()[0],
        'residents':        db.execute("SELECT COUNT(*) FROM residents").fetchone()[0],
        # Address intelligence stats
        'intel_addresses':  db.execute("SELECT COUNT(*) FROM address_intel").fetchone()[0] if _table_exists(db, 'address_intel') else 0,
        'intel_delivered':  db.execute("SELECT COUNT(*) FROM address_intel WHERE delivery_count > 0").fetchone()[0] if _table_exists(db, 'address_intel') else 0,
        'intel_top_zips':   db.execute("SELECT zip_code, COUNT(*) as cnt FROM address_intel WHERE zip_code IS NOT NULL GROUP BY zip_code ORDER BY cnt DESC LIMIT 5").fetchall() if _table_exists(db, 'address_intel') else [],
    }
    db.close()
    return render_template('admin.html', routes=routes, buildings=buildings, deliveries=deliveries, stats=stats, drivers_list=drivers_list)

# ─── MANAGER PORTAL (company ops — not app-owner admin) ───────
# Managers (e.g. Rolling Logistics) see only their company's drivers,
# routes, payroll, and daily check-ins.

PAYROLL_DEDUCTION_KINDS = {'claim', 'deduction'}

def _payroll_week_bounds(anchor=None):
    """Return (start_date, end_date) for the Saturday–Friday week containing anchor."""
    if not anchor:
        d = datetime.now().date()
    elif isinstance(anchor, str):
        try:
            d = datetime.strptime(anchor.strip(), '%Y-%m-%d').date()
        except Exception:
            d = datetime.now().date()
    else:
        d = anchor
    days_since_sat = (d.weekday() - 5) % 7
    start = d - timedelta(days=days_since_sat)
    return start, start + timedelta(days=6)

def _manager_session():
    """Return (manager_id, company_id, manager_name, company_name) or Nones."""
    return (
        session.get('manager_id'),
        session.get('company_id'),
        session.get('manager_name'),
        session.get('company_name'),
    )

def _require_manager():
    if not session.get('manager_id'):
        return redirect(url_for('manager_login'))
    return None

def _driver_in_company(db, driver_id, company_id):
    row = db.execute("SELECT id FROM drivers WHERE id = ? AND company_id = ?",
                     (driver_id, company_id)).fetchone()
    return bool(row)

def _company_driver_ids(db, company_id):
    rows = db.execute("SELECT id FROM drivers WHERE company_id = ?", (company_id,)).fetchall()
    return [r['id'] for r in rows]

def _build_payroll(db, start, end, company_id):
    """Aggregate payroll for one company's drivers in [start, end]."""
    start_s, end_s = start.isoformat(), end.isoformat()
    drivers = db.execute(
        "SELECT id, name, phone, company, COALESCE(default_rate, 0) AS default_rate "
        "FROM drivers WHERE company_id = ? ORDER BY name",
        (company_id,)
    ).fetchall()
    driver_ids = [d['id'] for d in drivers]
    if not driver_ids:
        empty = {'gross': 0.0, 'deductions': 0.0, 'additions': 0.0, 'net': 0.0, 'stops': 0, 'drivers_paid': 0}
        return [], empty, []

    ph = ','.join('?' * len(driver_ids))
    lines = db.execute(
        f"SELECT * FROM payroll_days WHERE work_date >= ? AND work_date <= ? "
        f"AND driver_id IN ({ph}) ORDER BY work_date, id",
        (start_s, end_s, *driver_ids)
    ).fetchall()
    adjustments = db.execute(
        f"SELECT * FROM payroll_adjustments WHERE work_date >= ? AND work_date <= ? "
        f"AND driver_id IN ({ph}) ORDER BY work_date, id",
        (start_s, end_s, *driver_ids)
    ).fetchall()

    by_driver = {}
    for d in drivers:
        by_driver[d['id']] = {
            'driver': d, 'lines': [], 'adjustments': [],
            'gross': 0.0, 'deductions': 0.0, 'additions': 0.0,
            'net': 0.0, 'total_stops': 0,
        }
    for r in lines:
        grp = by_driver.get(r['driver_id'])
        if not grp:
            continue
        amt = round((r['stops'] or 0) * (r['rate_per_stop'] or 0), 2)
        grp['lines'].append({'row': r, 'amount': amt})
        grp['gross'] += amt
        grp['total_stops'] += (r['stops'] or 0)
    for a in adjustments:
        grp = by_driver.get(a['driver_id'])
        if not grp:
            continue
        grp['adjustments'].append(a)
        if (a['kind'] or '').lower() in PAYROLL_DEDUCTION_KINDS:
            grp['deductions'] += (a['amount'] or 0)
        else:
            grp['additions'] += (a['amount'] or 0)

    summary = {'gross': 0.0, 'deductions': 0.0, 'additions': 0.0,
               'net': 0.0, 'stops': 0, 'drivers_paid': 0}
    groups = []
    for grp in by_driver.values():
        grp['gross'] = round(grp['gross'], 2)
        grp['deductions'] = round(grp['deductions'], 2)
        grp['additions'] = round(grp['additions'], 2)
        grp['net'] = round(grp['gross'] + grp['additions'] - grp['deductions'], 2)
        if grp['lines'] or grp['adjustments']:
            groups.append(grp)
            summary['gross'] += grp['gross']
            summary['deductions'] += grp['deductions']
            summary['additions'] += grp['additions']
            summary['net'] += grp['net']
            summary['stops'] += grp['total_stops']
            summary['drivers_paid'] += 1
    for k in ('gross', 'deductions', 'additions', 'net'):
        summary[k] = round(summary[k], 2)
    groups.sort(key=lambda g: (g['driver']['name'] or '').lower())
    return groups, summary, drivers

def _driver_reliability(db, driver_id, days=7):
    """Simple reliability: delivery completion % over recent routes."""
    since = (datetime.now().date() - timedelta(days=days)).isoformat()
    row = db.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN s.status = 'delivered' THEN 1 ELSE 0 END) AS done
           FROM stops s
           JOIN routes r ON s.route_id = r.id
           WHERE r.driver_id = ? AND r.date >= ?""",
        (driver_id, since)
    ).fetchone()
    total = (row['total'] or 0) if row else 0
    done = (row['done'] or 0) if row else 0
    if total == 0:
        return None
    return round(100 * done / total)

# ── Finish-by-9 estimation ──────────────────────────────────
# Goal: tell drivers/managers a realistic finish time BEFORE overload.
SERVICE_MIN_PER_STOP = 3.5     # park, walk, deliver, photo (avg)
DRIVE_MIN_PER_STOP_FALLBACK = 4.0  # used when no Mapbox drive estimate exists
DEADLINE_HOUR = 21             # 9:00 PM

def _fmt_time(dt):
    return dt.strftime('%I:%M %p').lstrip('0')

def _route_finish_estimate(db, route_id, total=None, pending=None, now=None):
    """Estimate finish time from remaining stops + prorated drive time.

    Returns dict: pending, minutes_remaining, finish_label, late, eta (short label).
    """
    now = now or datetime.now()
    if total is None or pending is None:
        row = db.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN status NOT IN ('delivered','failed') THEN 1 ELSE 0 END) AS pending
               FROM stops WHERE route_id=?""",
            (route_id,)
        ).fetchone()
        total = (row['total'] or 0) if row else 0
        pending = (row['pending'] or 0) if row else 0
    if total <= 0 or pending <= 0:
        return {'pending': 0, 'minutes_remaining': 0, 'finish_label': 'Done',
                'late': False, 'eta': 'Done'}
    rt = db.execute("SELECT est_duration_mins FROM routes WHERE id=?", (route_id,)).fetchone()
    if rt and rt['est_duration_mins'] and total > 0:
        drive_min = (rt['est_duration_mins'] / total) * pending
    else:
        drive_min = pending * DRIVE_MIN_PER_STOP_FALLBACK
    minutes = int(round(drive_min + pending * SERVICE_MIN_PER_STOP))
    finish = now + timedelta(minutes=minutes)
    late = finish.hour > DEADLINE_HOUR or (finish.hour == DEADLINE_HOUR and finish.minute > 0)
    label = _fmt_time(finish)
    return {
        'pending': pending,
        'minutes_remaining': minutes,
        'finish_label': label,
        'late': late,
        'eta': ('~' + label + (' ⚠️ past 9' if late else '')),
    }

@app.route('/manager/login', methods=['GET', 'POST'])
def manager_login():
    error = None
    ip = get_real_ip()
    if request.method == 'POST':
        if is_rate_limited(ip):
            return render_template('manager_login.html', error='Too many attempts. Try again in 5 minutes.')
        pin = request.form.get('pin', '').strip()
        db = get_db()
        mgr = db.execute(
            """SELECT m.*, c.name AS company_name
               FROM managers m JOIN companies c ON c.id = m.company_id
               WHERE m.pin = ?""",
            (pin,)
        ).fetchone()
        db.close()
        if mgr:
            session['manager_id'] = mgr['id']
            session['company_id'] = mgr['company_id']
            session['manager_name'] = mgr['name']
            session['company_name'] = mgr['company_name']
            clear_attempts(ip)
            return redirect(url_for('manager_dashboard'))
        record_attempt(ip)
        error = 'Wrong PIN'
    return render_template('manager_login.html', error=error)

@app.route('/manager/logout')
def manager_logout():
    for k in ('manager_id', 'company_id', 'manager_name', 'company_name'):
        session.pop(k, None)
    return redirect(url_for('index'))

@app.route('/manager')
def manager_dashboard():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, manager_name, company_name = _manager_session()
    today = datetime.now().date().isoformat()
    db = get_db()
    drivers = db.execute(
        """SELECT d.*, COALESCE(c.status, 'unknown') AS checkin_status,
                  c.assignment AS assignment
           FROM drivers d
           LEFT JOIN driver_checkins c ON c.driver_id = d.id AND c.check_date = ?
           WHERE d.company_id = ?
           ORDER BY d.name""",
        (today, company_id)
    ).fetchall()
    team = []
    for d in drivers:
        routes_today = db.execute(
            """SELECT r.*,
                      (SELECT COUNT(*) FROM stops WHERE route_id = r.id) AS total_stops,
                      (SELECT COUNT(*) FROM stops WHERE route_id = r.id AND status = 'delivered') AS done_stops,
                      (SELECT COUNT(*) FROM stops WHERE route_id = r.id AND status NOT IN ('delivered','failed')) AS pending_stops
               FROM routes r
               WHERE r.driver_id = ? AND r.date = ?
               ORDER BY r.created_at DESC""",
            (d['id'], today)
        ).fetchall()
        route_info = None
        if routes_today:
            r = routes_today[0]
            pending = r['pending_stops'] or 0
            total = r['total_stops'] or 0
            est = _route_finish_estimate(db, r['id'], total=total, pending=pending)
            route_info = {
                'name': r['name'] or 'Route',
                'total': total,
                'done': r['done_stops'] or 0,
                'pending': pending,
                'eta': est['eta'],
                'late': est['late'],
            }
        team.append({
            'driver': d,
            'checkin': d['checkin_status'],
            'assignment': d['assignment'] if 'assignment' in d.keys() else None,
            'reliability': _driver_reliability(db, d['id']),
            'route': route_info,
        })
    checked_in = sum(1 for t in team if t['checkin'] == 'in')
    not_in = sum(1 for t in team if t['checkin'] == 'out')
    unknown = sum(1 for t in team if t['checkin'] not in ('in', 'out'))
    db.close()
    return render_template(
        'manager_dashboard.html',
        team=team, today=today,
        manager_name=manager_name, company_name=company_name,
        checked_in=checked_in, not_in=not_in, unknown=unknown,
    )

@app.route('/manager/checkin', methods=['POST'])
def manager_checkin():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    driver_id = request.form.get('driver_id', '').strip()
    status = request.form.get('status', 'unknown').strip()
    if status not in ('in', 'out', 'unknown'):
        status = 'unknown'
    today = datetime.now().date().isoformat()
    if not driver_id.isdigit():
        return redirect(url_for('manager_dashboard'))
    db = get_db()
    if not _driver_in_company(db, int(driver_id), company_id):
        db.close()
        return redirect(url_for('manager_dashboard'))
    now = datetime.now().isoformat()
    existing = db.execute(
        "SELECT id FROM driver_checkins WHERE driver_id = ? AND check_date = ?",
        (int(driver_id), today)
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE driver_checkins SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, existing['id'])
        )
    else:
        db.execute(
            "INSERT INTO driver_checkins (driver_id, check_date, status, updated_at) VALUES (?,?,?,?)",
            (int(driver_id), today, status, now)
        )
    db.commit()
    db.close()
    return redirect(url_for('manager_dashboard'))

@app.route('/manager/assign', methods=['POST'])
def manager_assign():
    """Set a driver's route/area assignment for today."""
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    driver_id = request.form.get('driver_id', '').strip()
    assignment = request.form.get('assignment', '').strip()
    if not driver_id.isdigit():
        return redirect(url_for('manager_dashboard'))
    today = datetime.now().date().isoformat()
    now = datetime.now().isoformat()
    db = get_db()
    if not _driver_in_company(db, int(driver_id), company_id):
        db.close()
        return redirect(url_for('manager_dashboard'))
    existing = db.execute(
        "SELECT id FROM driver_checkins WHERE driver_id=? AND check_date=?",
        (int(driver_id), today)
    ).fetchone()
    if existing:
        db.execute("UPDATE driver_checkins SET assignment=?, updated_at=? WHERE id=?",
                   (assignment or None, now, existing['id']))
    else:
        db.execute(
            "INSERT INTO driver_checkins (driver_id, check_date, status, assignment, updated_at) VALUES (?,?,?,?,?)",
            (int(driver_id), today, 'unknown', assignment or None, now)
        )
    db.commit()
    db.close()
    return redirect(url_for('manager_dashboard'))

@app.route('/manager/rescue', methods=['POST'])
def manager_rescue():
    """Reassign a driver's remaining (pending) stops to another driver today."""
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    from_id = request.form.get('from_driver_id', '').strip()
    to_id = request.form.get('to_driver_id', '').strip()
    if not (from_id.isdigit() and to_id.isdigit()) or from_id == to_id:
        flash('Pick a different driver to rescue to.', 'rescue')
        return redirect(url_for('manager_dashboard'))
    from_id, to_id = int(from_id), int(to_id)
    today = datetime.now().strftime('%Y-%m-%d')
    db = get_db()
    if not (_driver_in_company(db, from_id, company_id) and _driver_in_company(db, to_id, company_id)):
        db.close()
        return redirect(url_for('manager_dashboard'))
    src_route = db.execute(
        "SELECT * FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
        (from_id, today)
    ).fetchone()
    if not src_route:
        db.close()
        flash('That driver has no route loaded today.', 'rescue')
        return redirect(url_for('manager_dashboard'))
    pending = db.execute(
        "SELECT id FROM stops WHERE route_id=? AND status NOT IN ('delivered','failed')",
        (src_route['id'],)
    ).fetchall()
    if not pending:
        db.close()
        flash('No unfinished stops to rescue.', 'rescue')
        return redirect(url_for('manager_dashboard'))
    # Find or create target driver's route for today
    to_driver = db.execute("SELECT name FROM drivers WHERE id=?", (to_id,)).fetchone()
    dst_route = db.execute(
        "SELECT * FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
        (to_id, today)
    ).fetchone()
    if not dst_route:
        db.execute(
            "INSERT INTO routes (driver_id, driver_name, name, date) VALUES (?,?,?,?)",
            (to_id, to_driver['name'] if to_driver else '', f'Rescue {today}', today)
        )
        db.commit()
        dst_route = db.execute(
            "SELECT * FROM routes WHERE driver_id=? AND date=? ORDER BY id DESC LIMIT 1",
            (to_id, today)
        ).fetchone()
    maxrow = db.execute("SELECT COALESCE(MAX(stop_number),0) AS mx FROM stops WHERE route_id=?", (dst_route['id'],)).fetchone()
    next_num = (maxrow['mx'] or 0) + 1
    moved = 0
    for s in pending:
        db.execute("UPDATE stops SET route_id=?, stop_number=? WHERE id=?", (dst_route['id'], next_num, s['id']))
        next_num += 1
        moved += 1
    db.commit()
    db.close()
    flash(f'Rescued {moved} stop(s) to {to_driver["name"] if to_driver else "driver"}.', 'rescue')
    return redirect(url_for('manager_dashboard'))

@app.route('/manager/team')
def manager_team():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, company_name = _manager_session()
    db = get_db()
    drivers = db.execute(
        """SELECT d.*,
                  COALESCE(d.default_rate, d.pay_rate, 0) AS rate
           FROM drivers d WHERE d.company_id = ? ORDER BY d.name""",
        (company_id,)
    ).fetchall()
    db.close()
    return render_template('manager_team.html', drivers=drivers, company_name=company_name)

@app.route('/manager/team/add', methods=['POST'])
def manager_team_add():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    name = request.form.get('name', '').strip()
    phone = format_phone(request.form.get('phone', '').strip())
    area = request.form.get('area', '').strip()
    rate = request.form.get('rate', '').strip()
    if not name:
        flash('Driver name is required.', 'team')
        return redirect(url_for('manager_team'))
    try: rate_v = float(rate) if rate else 1.50
    except Exception: rate_v = 1.50
    pin = str(secrets.randbelow(9000) + 1000)
    db = get_db()
    db.execute(
        "INSERT INTO drivers (name, phone, company, pin, is_beta, company_id, pay_rate, default_rate, assigned_zips) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (name, phone, 'Rolling Logistics', pin, 1, company_id, rate_v, rate_v, area or None)
    )
    db.commit()
    db.close()
    if phone:
        send_sms(phone, f"You've been added to UNIT for Rolling Logistics. Your driver PIN is: {pin}\nLog in: {get_base_url()}/driver/login")
    flash(f'Added {name} — PIN: {pin}{" (texted)" if phone else ""}', 'team')
    return redirect(url_for('manager_team'))

@app.route('/manager/team/<int:driver_id>/update', methods=['POST'])
def manager_team_update(driver_id):
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    db = get_db()
    if not _driver_in_company(db, driver_id, company_id):
        db.close()
        return redirect(url_for('manager_team'))
    phone = format_phone(request.form.get('phone', '').strip())
    area = request.form.get('area', '').strip()
    rate = request.form.get('rate', '').strip()
    try: rate_v = float(rate) if rate else 0
    except Exception: rate_v = 0
    db.execute(
        "UPDATE drivers SET phone=?, assigned_zips=?, pay_rate=?, default_rate=? WHERE id=?",
        (phone or None, area or None, rate_v, rate_v, driver_id)
    )
    db.commit()
    db.close()
    flash('Driver updated.', 'team')
    return redirect(url_for('manager_team'))

@app.route('/manager/team/<int:driver_id>/remove', methods=['POST'])
def manager_team_remove(driver_id):
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    db = get_db()
    if _driver_in_company(db, driver_id, company_id):
        db.execute("UPDATE drivers SET company_id=NULL WHERE id=?", (driver_id,))
        db.commit()
    db.close()
    flash('Driver removed from team.', 'team')
    return redirect(url_for('manager_team'))

@app.route('/manager/messages')
def manager_messages():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, company_name = _manager_session()
    db = get_db()
    drivers = db.execute(
        "SELECT id, name, phone FROM drivers WHERE company_id = ? ORDER BY name",
        (company_id,)
    ).fetchall()
    with_phone = [d for d in drivers if d['phone']]
    history = db.execute(
        "SELECT * FROM manager_messages WHERE company_id = ? ORDER BY id DESC LIMIT 20",
        (company_id,)
    ).fetchall()
    db.close()
    return render_template('manager_messages.html',
                           company_name=company_name, total=len(drivers),
                           reachable=len(with_phone), history=history)

@app.route('/manager/messages/send', methods=['POST'])
def manager_messages_send():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, company_name = _manager_session()
    body = request.form.get('body', '').strip()
    if not body:
        flash('Message is empty.', 'msg')
        return redirect(url_for('manager_messages'))
    db = get_db()
    drivers = db.execute(
        "SELECT name, phone FROM drivers WHERE company_id = ? AND phone IS NOT NULL AND phone != ''",
        (company_id,)
    ).fetchall()
    prefix = f'[{company_name}] '
    sent = fail = 0
    for d in drivers:
        ok, _info = send_sms(d['phone'], prefix + body)
        if ok:
            sent += 1
        else:
            fail += 1
    db.execute(
        "INSERT INTO manager_messages (company_id, body, sent_count, fail_count) VALUES (?,?,?,?)",
        (company_id, body, sent, fail)
    )
    db.commit()
    db.close()
    if sent:
        flash(f'Sent to {sent} driver(s){f", {fail} failed" if fail else ""}.', 'msg')
    else:
        flash('No messages sent — check that drivers have phone numbers.', 'msg')
    return redirect(url_for('manager_messages'))

@app.route('/manager/payroll')
def manager_payroll():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, company_name = _manager_session()
    start, end = _payroll_week_bounds(request.args.get('week'))
    db = get_db()
    groups, summary, all_drivers = _build_payroll(db, start, end, company_id)
    db.close()
    return render_template(
        'payroll.html',
        groups=groups, summary=summary, all_drivers=all_drivers,
        week_start=start, week_end=end,
        week_start_s=start.isoformat(), week_end_s=end.isoformat(),
        week_days=[start + timedelta(days=i) for i in range(7)],
        prev_week=(start - timedelta(days=7)).isoformat(),
        next_week=(start + timedelta(days=7)).isoformat(),
        today_s=datetime.now().date().isoformat(),
        company_name=company_name,
    )

@app.route('/manager/payroll/pull', methods=['POST'])
def manager_payroll_pull():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    start, end = _payroll_week_bounds(request.form.get('week'))
    start_s, end_s = start.isoformat(), end.isoformat()
    driver_ids = _company_driver_ids(db := get_db(), company_id)
    if not driver_ids:
        db.close()
        flash('No drivers on your team yet.', 'payroll')
        return redirect(url_for('manager_payroll', week=start_s))
    ph = ','.join('?' * len(driver_ids))
    rows = db.execute(
        f"""SELECT r.driver_id AS driver_id, substr(s.delivered_at, 1, 10) AS d, COUNT(*) AS cnt
            FROM stops s
            JOIN routes r ON s.route_id = r.id
            WHERE s.status = 'delivered' AND s.delivered_at IS NOT NULL
              AND substr(s.delivered_at, 1, 10) >= ? AND substr(s.delivered_at, 1, 10) <= ?
              AND r.driver_id IN ({ph})
            GROUP BY r.driver_id, substr(s.delivered_at, 1, 10)""",
        (start_s, end_s, *driver_ids)
    ).fetchall()
    created = updated = 0
    for row in rows:
        did, wd, cnt = row['driver_id'], row['d'], row['cnt']
        if not did or not wd:
            continue
        existing = db.execute(
            "SELECT id FROM payroll_days WHERE driver_id = ? AND work_date = ? AND source = 'auto'",
            (did, wd)
        ).fetchone()
        drow = db.execute("SELECT COALESCE(default_rate, 0) AS dr FROM drivers WHERE id = ?", (did,)).fetchone()
        rate = (drow['dr'] if drow else 0) or 0
        if existing:
            db.execute("UPDATE payroll_days SET stops = ?, updated_at = ? WHERE id = ?",
                       (cnt, datetime.now().isoformat(), existing['id']))
            updated += 1
        else:
            db.execute(
                "INSERT INTO payroll_days (driver_id, work_date, stops, rate_per_stop, area, source) "
                "VALUES (?,?,?,?,?,?)",
                (did, wd, cnt, rate, 'UNIT deliveries', 'auto')
            )
            created += 1
    db.commit()
    db.close()
    flash(f'Pulled UNIT deliveries — {created} day(s) added, {updated} updated.', 'payroll')
    return redirect(url_for('manager_payroll', week=start_s))

@app.route('/manager/payroll/save', methods=['POST'])
def manager_payroll_save():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    start, _ = _payroll_week_bounds(request.form.get('week'))
    db = get_db()
    for lid in [x for x in request.form.get('line_ids', '').split(',') if x.strip().isdigit()]:
        row = db.execute(
            "SELECT pd.driver_id FROM payroll_days pd JOIN drivers d ON d.id = pd.driver_id "
            "WHERE pd.id = ? AND d.company_id = ?",
            (int(lid), company_id)
        ).fetchone()
        if not row:
            continue
        stops = request.form.get(f'stops_{lid}', '').strip()
        rate = request.form.get(f'rate_{lid}', '').strip()
        area = request.form.get(f'area_{lid}', '').strip()
        try: stops_v = int(float(stops)) if stops != '' else 0
        except Exception: stops_v = 0
        try: rate_v = float(rate) if rate != '' else 0
        except Exception: rate_v = 0
        db.execute(
            "UPDATE payroll_days SET stops = ?, rate_per_stop = ?, area = ?, updated_at = ? WHERE id = ?",
            (stops_v, rate_v, area or None, datetime.now().isoformat(), int(lid))
        )
    db.commit()
    db.close()
    flash('Payroll saved.', 'payroll')
    return redirect(url_for('manager_payroll', week=start.isoformat()))

@app.route('/manager/payroll/line/add', methods=['POST'])
def manager_payroll_line_add():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    start, _ = _payroll_week_bounds(request.form.get('week'))
    driver_id = request.form.get('driver_id', '').strip()
    work_date = request.form.get('work_date', '').strip()
    stops = request.form.get('stops', '').strip()
    rate = request.form.get('rate', '').strip()
    area = request.form.get('area', '').strip()
    if driver_id.isdigit() and work_date:
        db = get_db()
        if not _driver_in_company(db, int(driver_id), company_id):
            db.close()
            return redirect(url_for('manager_payroll', week=start.isoformat()))
        try: stops_v = int(float(stops)) if stops else 0
        except Exception: stops_v = 0
        try: rate_v = float(rate) if rate else 0
        except Exception: rate_v = 0
        db.execute(
            "INSERT INTO payroll_days (driver_id, work_date, stops, rate_per_stop, area, source) "
            "VALUES (?,?,?,?,?,?)",
            (int(driver_id), work_date, stops_v, rate_v, area or None, 'manual')
        )
        if rate_v:
            db.execute("UPDATE drivers SET default_rate = ? WHERE id = ?", (rate_v, int(driver_id)))
        db.commit()
        db.close()
    return redirect(url_for('manager_payroll', week=start.isoformat()))

@app.route('/manager/payroll/line/<int:line_id>/delete', methods=['POST'])
def manager_payroll_line_delete(line_id):
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    start, _ = _payroll_week_bounds(request.form.get('week'))
    db = get_db()
    db.execute(
        "DELETE FROM payroll_days WHERE id = ? AND driver_id IN "
        "(SELECT id FROM drivers WHERE company_id = ?)",
        (line_id, company_id)
    )
    db.commit()
    db.close()
    return redirect(url_for('manager_payroll', week=start.isoformat()))

@app.route('/manager/payroll/adjustment/add', methods=['POST'])
def manager_payroll_adjustment_add():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    start, _ = _payroll_week_bounds(request.form.get('week'))
    driver_id = request.form.get('driver_id', '').strip()
    work_date = request.form.get('work_date', '').strip() or start.isoformat()
    kind = request.form.get('kind', 'claim').strip() or 'claim'
    amount = request.form.get('amount', '').strip()
    note = request.form.get('note', '').strip()
    if driver_id.isdigit():
        db = get_db()
        if not _driver_in_company(db, int(driver_id), company_id):
            db.close()
            return redirect(url_for('manager_payroll', week=start.isoformat()))
        try: amount_v = abs(float(amount)) if amount else 0
        except Exception: amount_v = 0
        if amount_v:
            db.execute(
                "INSERT INTO payroll_adjustments (driver_id, work_date, kind, amount, note) "
                "VALUES (?,?,?,?,?)",
                (int(driver_id), work_date, kind, amount_v, note or None)
            )
            db.commit()
        db.close()
    return redirect(url_for('manager_payroll', week=start.isoformat()))

@app.route('/manager/payroll/adjustment/<int:adj_id>/delete', methods=['POST'])
def manager_payroll_adjustment_delete(adj_id):
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    start, _ = _payroll_week_bounds(request.form.get('week'))
    db = get_db()
    db.execute(
        "DELETE FROM payroll_adjustments WHERE id = ? AND driver_id IN "
        "(SELECT id FROM drivers WHERE company_id = ?)",
        (adj_id, company_id)
    )
    db.commit()
    db.close()
    return redirect(url_for('manager_payroll', week=start.isoformat()))

@app.route('/manager/payroll/export')
def manager_payroll_export():
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, company_name = _manager_session()
    start, end = _payroll_week_bounds(request.args.get('week'))
    db = get_db()
    groups, summary, _ = _build_payroll(db, start, end, company_id)
    db.close()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([f'{company_name} Payroll', f'{start.isoformat()} to {end.isoformat()}'])
    w.writerow([])
    w.writerow(['Driver', 'Phone', 'Stops', 'Gross', 'Additions', 'Deductions', 'Net Pay'])
    for g in groups:
        w.writerow([g['driver']['name'], g['driver']['phone'] or '', g['total_stops'],
                    f"{g['gross']:.2f}", f"{g['additions']:.2f}",
                    f"{g['deductions']:.2f}", f"{g['net']:.2f}"])
    w.writerow([])
    w.writerow(['TOTAL', '', summary['stops'], f"{summary['gross']:.2f}",
                f"{summary['additions']:.2f}", f"{summary['deductions']:.2f}", f"{summary['net']:.2f}"])
    from flask import Response
    slug = (company_name or 'payroll').lower().replace(' ', '_')
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={slug}_{start.isoformat()}.csv'})

@app.route('/manager/payroll/statement/<int:driver_id>')
def manager_payroll_statement(driver_id):
    guard = _require_manager()
    if guard:
        return guard
    _, company_id, _, _ = _manager_session()
    if not _driver_in_company(db := get_db(), driver_id, company_id):
        db.close()
        return render_template('payroll_statement.html', grp=None,
                               week_start=datetime.now().date(), week_end=datetime.now().date())
    start, end = _payroll_week_bounds(request.args.get('week'))
    groups, _, _ = _build_payroll(db, start, end, company_id)
    db.close()
    grp = next((g for g in groups if g['driver']['id'] == driver_id), None)
    return render_template('payroll_statement.html', grp=grp,
                           week_start=start, week_end=end, week_start_s=start.isoformat())

# Legacy admin payroll URLs → manager portal
@app.route('/admin/payroll')
@app.route('/admin/payroll/<path:subpath>')
def admin_payroll_redirect(subpath=None):
    flash('Payroll moved to the Manager portal.', 'payroll')
    return redirect(url_for('manager_login'))

# ─── TEST SMS ─────────────────────────────────────────────────

@app.route('/admin/regeocode', methods=['POST'])
def admin_regeocode():
    """Clear bad (null) geocode results so stops re-geocode on next load."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    _geocache.clear()
    db = get_db()
    db.execute("UPDATE stops SET dest_lat=NULL, dest_lng=NULL WHERE dest_lat IS NULL OR (dest_lat > -0.001 AND dest_lat < 0.001)")
    db.commit()
    db.close()
    flash('Geocache cleared — stops will re-geocode on next load.', 'beta_pin')
    return redirect(url_for('admin'))

@app.route('/admin/building', methods=['POST'])
def admin_add_building():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    address            = request.form.get('address', '').strip()
    access_code        = request.form.get('access_code', '').strip()
    buzzer_notes       = request.form.get('buzzer_notes', '').strip()
    interior_directions = request.form.get('interior_directions', '').strip()
    access_type        = request.form.get('access_type', 'code').strip()
    if address:
        try:
            db.execute(
                """INSERT INTO buildings (address, access_code, buzzer_notes, interior_directions, access_type)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(address) DO UPDATE SET
                     access_code=excluded.access_code,
                     buzzer_notes=excluded.buzzer_notes,
                     interior_directions=excluded.interior_directions,
                     access_type=excluded.access_type""",
                (address, access_code, buzzer_notes, interior_directions, access_type)
            )
            db.commit()
        except Exception as e:
            log.error(f'Building save error: {e}')
    db.close()
    return redirect(url_for('admin'))

@app.route('/admin/building/<int:building_id>/delete', methods=['POST'])
def admin_delete_building(building_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    db.execute("DELETE FROM buildings WHERE id=?", (building_id,))
    db.commit()
    db.close()
    return redirect(url_for('admin'))

@app.route('/admin/test-sms', methods=['POST'])
def admin_test_sms():
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    phone = format_phone(data.get('phone', '').strip())
    if not phone:
        return jsonify({'success': False, 'error': 'No phone number provided'})
    msg = '🚚 UNIT Test — SMS delivery confirmed. Your system is working correctly.'
    ok, detail = send_sms(phone, msg)
    provider = 'textbelt' if TEXTBELT_KEY else ('twilio' if TWILIO_SID else 'mock')
    return jsonify({'success': ok, 'provider': provider, 'detail': str(detail)})

# ─── ACCOUNT ─────────────────────────────────────────────────────

@app.route('/account')
def account():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    driver = db.execute("SELECT * FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    db.close()
    pay_rate = float(driver['pay_rate']) if driver and driver['pay_rate'] else 1.50
    return render_template('account.html', driver=session['driver_name'], phone=driver['phone'] or '', pay_rate=pay_rate)

@app.route('/account/edit', methods=['POST'])
def account_edit():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    name     = request.form.get('name', '').strip()
    phone    = request.form.get('phone', '').strip()
    pay_rate_str = request.form.get('pay_rate', '').strip()
    db = get_db()
    if name and phone:
        # Validate pay rate
        try:
            new_pay_rate = float(pay_rate_str)
            if new_pay_rate <= 0 or new_pay_rate > 99:
                new_pay_rate = None
        except (ValueError, TypeError):
            new_pay_rate = None

        if new_pay_rate:
            db.execute("UPDATE drivers SET name=?, phone=?, pay_rate=? WHERE id=?",
                       (name, format_phone(phone), new_pay_rate, session['driver_id']))
        else:
            db.execute("UPDATE drivers SET name=?, phone=? WHERE id=?",
                       (name, format_phone(phone), session['driver_id']))
        db.commit()
        session['driver_name'] = name
    db.close()
    return redirect(url_for('account'))

@app.route('/account/manage')
def account_manage():
    """Redirect to Stripe Customer Portal for billing/cancel."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    try:
        db = get_db()
        driver = db.execute("SELECT * FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
        db.close()
        # Find Stripe customer by email/phone
        customers = stripe.Customer.search(query=f"phone:'{driver['phone']}'")
        if customers and customers.data:
            customer_id = customers.data[0].id
        else:
            return redirect(url_for('account'))
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=get_base_url() + '/account'
        )
        return redirect(portal.url)
    except Exception as e:
        log.error(f'Portal error: {e}')
        return redirect(url_for('account'))

# ─── SIGNUP + STRIPE ─────────────────────────────────────────────────────

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        phone   = format_phone(request.form.get('phone', '').strip())
        company = request.form.get('company', '').strip()
        email   = request.form.get('email', '').strip()
        if not name or not phone or not email:
            error = 'Name, phone, and email are required.'
        else:
            return render_template('signup_checkout.html',
                name=name, phone=phone, company=company, email=email,
                publishable_key=STRIPE_PUB_KEY)
    return render_template('signup.html', error=error)

@app.route('/signup/complete', methods=['POST'])
def signup_complete():
    data = request.get_json()
    name    = data.get('name', '').strip()
    phone   = format_phone(data.get('phone', '').strip())
    company = data.get('company', '').strip()
    email   = data.get('email', '').strip()
    pm_id   = data.get('payment_method_id', '')

    try:
        # Create Stripe customer
        customer = stripe.Customer.create(
            name=name, email=email, phone=phone,
            payment_method=pm_id,
            invoice_settings={'default_payment_method': pm_id}
        )
        # Create subscription with 14-day trial
        stripe.Subscription.create(
            customer=customer.id,
            items=[{'price': STRIPE_PRICE_ID}],
            trial_period_days=14,
            expand=['latest_invoice.payment_intent']
        )
        # Generate PIN and create driver account
        pin = str(secrets.randbelow(9000) + 1000)  # 4-digit PIN
        db = get_db()
        db.execute(
            "INSERT INTO drivers (name, phone, company, pin) VALUES (?,?,?,?)",
            (name, phone, company, pin)
        )
        db.commit()
        db.close()
        # Send PIN via SMS
        send_sms(phone, f"Your UNIT driver PIN is: {pin}\nLogin at: {get_base_url()}/driver/login\nTrial ends in 14 days. $20/month after.")
        return jsonify({'success': True, 'pin': pin})
    except stripe.error.StripeError as e:
        log.error(f'Stripe error: {e}')
        return jsonify({'success': False, 'error': str(e.user_message)})
    except Exception as e:
        log.error(f'Signup error: {e}')
        return jsonify({'success': False, 'error': 'Something went wrong. Please try again.'})

@app.route('/signup/success')
def signup_success():
    pin = request.args.get('pin', '----')
    return render_template('signup_success.html', pin=pin)

# ─── LEGAL ─────────────────────────────────────────────────────

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

# ─── HEALTH ────────────────────────────────────────────────────

@app.route('/health')
def health():
    try:
        db = get_db()
        db.execute('SELECT 1').fetchone()
        db.close()
        import subprocess
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(__file__) or '.', stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = 'unknown'
        return jsonify({'status': 'ok', 'time': datetime.now().isoformat(), 'version': git_hash,
                        'model': _vision_provider_label(),
                        'vision_key': 'set' if _vision_available() else 'MISSING',
                        'vision_provider': 'gemini' if GEMINI_API_KEY else ('anthropic' if ANTHROPIC_KEY else 'none'),
                        'mapbox_token': 'set' if MAPBOX_TOKEN else 'MISSING',
                        'in_app_nav': 'live' if MAPBOX_TOKEN else 'fallback-only (set MAPBOX_TOKEN)'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

# ─── ROUTE LOG / PAYROLL ───────────────────────────────────────────────────

@app.route('/driver/route-log')
def route_log():
    """Daily route log — payroll tracker. No screenshots needed."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))

    db        = get_db()
    driver_id = session['driver_id']
    driver    = db.execute("SELECT * FROM drivers WHERE id=?", (driver_id,)).fetchone()
    pay_rate  = float(driver['pay_rate']) if driver and driver['pay_rate'] else 1.50

    # Daily breakdown — last 30 days.
    # GROUP BY date only; aggregate name with MAX so this is valid on PostgreSQL
    # (Postgres rejects non-aggregated SELECT columns missing from GROUP BY).
    days = db.execute("""
        SELECT
            r.date,
            COUNT(s.id)                                              AS total,
            SUM(CASE WHEN s.status='delivered' THEN 1 ELSE 0 END)   AS delivered,
            SUM(CASE WHEN s.status='failed'    THEN 1 ELSE 0 END)   AS failed,
            SUM(CASE WHEN s.status='pending'   THEN 1 ELSE 0 END)   AS pending,
            MAX(r.name)                                             AS route_name
        FROM routes r
        LEFT JOIN stops s ON s.route_id = r.id
        WHERE r.driver_id = ?
        GROUP BY r.date
        ORDER BY r.date DESC
        LIMIT 30
    """, (driver_id,)).fetchall()

    # Manual log entries (for days without scan data)
    try:
        manual = db.execute("""
            SELECT * FROM route_manual_log
            WHERE driver_id=?
            ORDER BY date DESC LIMIT 30
        """, (driver_id,)).fetchall()
    except Exception:
        manual = []
        try: db._conn.rollback()
        except: pass

    # This week totals (Mon–today)
    from datetime import date as _date, timedelta
    today      = _date.today()
    week_start = (today - timedelta(days=today.weekday())).strftime('%Y-%m-%d')
    week_row   = db.execute("""
        SELECT
            SUM(CASE WHEN s.status='delivered' THEN 1 ELSE 0 END) AS week_delivered,
            COUNT(s.id) AS week_total
        FROM routes r
        LEFT JOIN stops s ON s.route_id = r.id
        WHERE r.driver_id=? AND r.date >= ?
    """, (driver_id, week_start)).fetchone()

    db.close()

    week_delivered = week_row['week_delivered'] or 0
    week_total     = week_row['week_total']     or 0
    week_earnings  = round(week_delivered * pay_rate, 2)

    # Build combined day list with earnings
    log_days = []
    for d in days:
        delivered = d['delivered'] or 0
        log_days.append({
            'date':       d['date'],
            'total':      d['total'] or 0,
            'delivered':  delivered,
            'failed':     d['failed'] or 0,
            'pending':    d['pending'] or 0,
            'earnings':   round(delivered * pay_rate, 2),
            'route_name': d['route_name'] or '',
            'source':     'scan'
        })

    for m in manual:
        log_days.append({
            'date':      m['date'],
            'total':     m['packages'] or 0,
            'delivered': m['packages'] or 0,
            'failed':    0,
            'pending':   0,
            'earnings':  round((m['packages'] or 0) * pay_rate, 2),
            'route_name': m['notes'] or 'Manual entry',
            'source':    'manual'
        })

    # Sort combined by date desc
    log_days.sort(key=lambda x: x['date'], reverse=True)

    return render_template('route_log.html',
                           log_days=log_days,
                           pay_rate=pay_rate,
                           week_delivered=week_delivered,
                           week_total=week_total,
                           week_earnings=week_earnings,
                           week_start=week_start,
                           today=today.strftime('%Y-%m-%d'))


@app.route('/driver/route-log/manual', methods=['POST'])
def route_log_manual():
    """Add a manual day entry when scan data wasn't captured."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))

    date_val = request.form.get('date', '').strip()
    packages = request.form.get('packages', '0').strip()
    notes    = request.form.get('notes', '').strip()

    if not date_val or not packages.isdigit():
        return redirect(url_for('route_log'))

    db = get_db()
    db.execute(
        "INSERT INTO route_manual_log (driver_id, date, packages, notes) VALUES (?,?,?,?)",
        (session['driver_id'], date_val, int(packages), notes)
    )
    db.commit()
    db.close()
    return redirect(url_for('route_log'))


@app.route('/driver/route-log/manual/<int:entry_id>/delete', methods=['POST'])
def route_log_manual_delete(entry_id):
    """Delete a manual entry."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    db.execute("DELETE FROM route_manual_log WHERE id=? AND driver_id=?",
               (entry_id, session['driver_id']))
    db.commit()
    db.close()
    return redirect(url_for('route_log'))


# ─── QR BUILDING ACCESS ────────────────────────────────────────
# Driver scans a QR posted at a building entrance → enters PIN → GPS is
# checked against the building location (100m radius) → access codes and
# per-unit delivery notes are revealed. Adapted to this stack: reuses the
# existing `drivers.pin` auth (no separate hashed driver_pins table) and
# geopy's geodesic for the distance gate (Haversine equivalent).

BUILDING_GEOFENCE_METERS = 100

def _gen_building_code():
    """Generate a unique, URL-safe building identifier like bld_a1b2c3d4."""
    return 'bld_' + secrets.token_hex(4)


def _verify_driver_pin(db, pin):
    """Return the driver row for a valid PIN, else None. Mirrors driver_login."""
    if not pin or not pin.strip():
        return None
    return db.execute("SELECT * FROM drivers WHERE pin=?", (pin.strip(),)).fetchone()


def _meters_between(lat1, lng1, lat2, lng2):
    """Great-circle distance in meters (geopy geodesic — Haversine equivalent)."""
    try:
        return geodesic((lat1, lng1), (lat2, lng2)).meters
    except Exception:
        return float('inf')


def _building_public(b):
    """Minimal building info safe to show BEFORE verification (no codes)."""
    return {
        'building_code': _ss_val(b, 'building_code'),
        'name': _ss_val(b, 'name') or _ss_val(b, 'address') or 'Building',
        'address': _ss_val(b, 'address') or '',
        'has_geo': bool(_ss_val(b, 'lat') and _ss_val(b, 'lng')),
    }


@app.route('/driver/building-scan')
def building_scan():
    """QR scanner screen — camera decodes the building QR and redirects."""
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    return render_template('building_scan.html', driver=session['driver_name'])


@app.route('/building/<building_code>')
def building_gate(building_code):
    """
    Landing page from a scanned QR. Requires a logged-in driver, then shows
    the PIN + GPS verification gate. Access data is NOT sent until verified.
    """
    if 'driver_id' not in session:
        # Remember where we were headed, send through login, come back here.
        session['post_login_redirect'] = url_for('building_gate', building_code=building_code)
        return redirect(url_for('driver_login'))

    db = get_db()
    b = db.execute("SELECT * FROM buildings WHERE building_code=?", (building_code,)).fetchone()
    db.close()
    if not b:
        return render_template('building_access.html', not_found=True,
                               building_code=building_code), 404
    return render_template('building_access.html', not_found=False,
                           building=_building_public(b),
                           building_code=building_code,
                           geofence_m=BUILDING_GEOFENCE_METERS)


@app.route('/api/building/verify', methods=['POST'])
def building_verify():
    """
    Secure verification gate.
    Inputs (JSON): building_code, pin, lat, lng.
    Steps: valid PIN  →  within geofence  →  return access payload.
    """
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not_logged_in'}), 401

    ip = get_real_ip()
    if is_rate_limited(ip):
        return jsonify({'ok': False, 'error': 'rate_limited',
                        'message': 'Too many attempts — wait a few minutes.'}), 429

    data = request.get_json(silent=True) or {}
    building_code = (data.get('building_code') or '').strip()
    entered_pin   = (data.get('pin') or '').strip()
    try:
        driver_lat = float(data.get('lat'))
        driver_lng = float(data.get('lng'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'no_location',
                        'message': 'Location required — enable GPS and try again.'}), 400

    db = get_db()
    b = db.execute("SELECT * FROM buildings WHERE building_code=?", (building_code,)).fetchone()
    if not b:
        db.close()
        return jsonify({'ok': False, 'error': 'not_found',
                        'message': 'Building not found.'}), 404

    # 1) PIN check (reuses existing driver auth)
    driver = _verify_driver_pin(db, entered_pin)
    if not driver:
        record_attempt(ip)
        db.close()
        return jsonify({'ok': False, 'error': 'bad_pin',
                        'message': 'Invalid PIN.'}), 403

    # 2) Geofence check (100m via Haversine/geodesic)
    blat, blng = _ss_val(b, 'lat'), _ss_val(b, 'lng')
    if blat is None or blng is None:
        db.close()
        return jsonify({'ok': False, 'error': 'no_building_geo',
                        'message': 'Building has no saved coordinates — contact admin.'}), 409

    distance_m = _meters_between(driver_lat, driver_lng, blat, blng)
    if distance_m > BUILDING_GEOFENCE_METERS:
        db.close()
        return jsonify({'ok': False, 'error': 'too_far',
                        'distance_m': round(distance_m),
                        'message': f'You are {round(distance_m)}m away — must be within '
                                   f'{BUILDING_GEOFENCE_METERS}m of the building.'}), 403

    # 3) Both passed — return access payload + per-unit instructions
    units = db.execute(
        """SELECT unit_number, customer_notes FROM delivery_instructions
           WHERE building_id=? ORDER BY unit_number ASC""",
        (b['id'],)
    ).fetchall()
    db.close()

    return jsonify({
        'ok': True,
        'building': {
            'name': _ss_val(b, 'name') or _ss_val(b, 'address') or 'Building',
            'address': _ss_val(b, 'address') or '',
            'general_access_code': _ss_val(b, 'general_access_code') or _ss_val(b, 'access_code') or '',
            'package_room_notes': _ss_val(b, 'package_room_notes') or '',
            'lockbox_notes': _ss_val(b, 'lockbox_notes') or '',
            'buzzer_notes': _ss_val(b, 'buzzer_notes') or '',
            'interior_directions': _ss_val(b, 'interior_directions') or '',
        },
        'distance_m': round(distance_m),
        'delivery_instructions': [
            {'unit_number': u['unit_number'], 'customer_notes': u['customer_notes'] or ''}
            for u in units
        ],
    })


# ─── ADMIN: BUILDING ACCESS MANAGEMENT ─────────────────────────

@app.route('/admin/buildings')
def admin_buildings():
    """Manage building access data + per-unit notes + QR codes."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    buildings = db.execute("SELECT * FROM buildings ORDER BY name, address").fetchall()
    rows = []
    for b in buildings:
        units = db.execute(
            "SELECT * FROM delivery_instructions WHERE building_id=? ORDER BY unit_number ASC",
            (b['id'],)
        ).fetchall()
        rows.append({'b': b, 'units': units})
    db.close()
    return render_template('admin_buildings.html', rows=rows,
                           base_url=request.host_url.rstrip('/'))


@app.route('/admin/buildings/save', methods=['POST'])
def admin_building_save():
    """Create or update a building's access data (admin only)."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    f = request.form
    building_id = (f.get('building_id') or '').strip()
    address     = (f.get('address') or '').strip()
    if not address:
        flash('Address is required.', 'beta_pin')
        return redirect(url_for('admin_buildings'))

    name                = (f.get('name') or '').strip()
    general_access_code = (f.get('general_access_code') or '').strip()
    package_room_notes  = (f.get('package_room_notes') or '').strip()
    lockbox_notes       = (f.get('lockbox_notes') or '').strip()
    interior_directions = (f.get('interior_directions') or '').strip()
    lat_raw, lng_raw    = (f.get('lat') or '').strip(), (f.get('lng') or '').strip()

    lat = lng = None
    try:
        if lat_raw and lng_raw:
            lat, lng = float(lat_raw), float(lng_raw)
    except ValueError:
        lat = lng = None
    # Auto-geocode from address if coordinates not provided
    if lat is None or lng is None:
        try:
            coords = geocode_address(address)
            if coords:
                lat, lng = coords
        except Exception as e:
            log.warning(f'building geocode failed: {e}')

    db = get_db()
    if building_id:
        db.execute(
            """UPDATE buildings SET name=?, address=?, general_access_code=?,
               package_room_notes=?, lockbox_notes=?, interior_directions=?,
               lat=COALESCE(?, lat), lng=COALESCE(?, lng)
               WHERE id=?""",
            (name, address, general_access_code, package_room_notes,
             lockbox_notes, interior_directions, lat, lng, building_id)
        )
        # Ensure it has a building_code for QR
        row = db.execute("SELECT building_code FROM buildings WHERE id=?", (building_id,)).fetchone()
        if row and not _ss_val(row, 'building_code'):
            db.execute("UPDATE buildings SET building_code=? WHERE id=?",
                       (_gen_building_code(), building_id))
        db.commit()
    else:
        code = _gen_building_code()
        try:
            db.execute(
                """INSERT INTO buildings (address, name, general_access_code,
                   package_room_notes, lockbox_notes, interior_directions,
                   lat, lng, building_code)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(address) DO UPDATE SET
                     name=excluded.name,
                     general_access_code=excluded.general_access_code,
                     package_room_notes=excluded.package_room_notes,
                     lockbox_notes=excluded.lockbox_notes,
                     interior_directions=excluded.interior_directions,
                     lat=COALESCE(excluded.lat, buildings.lat),
                     lng=COALESCE(excluded.lng, buildings.lng)""",
                (address, name, general_access_code, package_room_notes,
                 lockbox_notes, interior_directions, lat, lng, code)
            )
            db.commit()
            # Backfill code if the row already existed without one
            row = db.execute("SELECT building_code FROM buildings WHERE address=?", (address,)).fetchone()
            if row and not _ss_val(row, 'building_code'):
                db.execute("UPDATE buildings SET building_code=? WHERE address=?", (code, address))
                db.commit()
        except Exception as e:
            log.error(f'building save error: {e}')
            try: db._conn.rollback()
            except: pass
    db.close()
    flash('Building saved.', 'beta_pin')
    return redirect(url_for('admin_buildings'))


@app.route('/admin/buildings/<int:building_id>/unit', methods=['POST'])
def admin_building_add_unit(building_id):
    """Add / update a per-unit delivery instruction."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    unit_number = (request.form.get('unit_number') or '').strip()
    customer_notes = (request.form.get('customer_notes') or '').strip()
    if unit_number:
        db = get_db()
        existing = db.execute(
            "SELECT id FROM delivery_instructions WHERE building_id=? AND unit_number=?",
            (building_id, unit_number)
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE delivery_instructions SET customer_notes=?, updated_at=? WHERE id=?",
                (customer_notes, datetime.now().isoformat(), existing['id'])
            )
        else:
            db.execute(
                """INSERT INTO delivery_instructions (building_id, unit_number, customer_notes, updated_at)
                   VALUES (?,?,?,?)""",
                (building_id, unit_number, customer_notes, datetime.now().isoformat())
            )
        db.commit()
        db.close()
    return redirect(url_for('admin_buildings'))


@app.route('/admin/buildings/unit/<int:unit_id>/delete', methods=['POST'])
def admin_building_delete_unit(unit_id):
    """Remove a per-unit delivery instruction."""
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    db.execute("DELETE FROM delivery_instructions WHERE id=?", (unit_id,))
    db.commit()
    db.close()
    return redirect(url_for('admin_buildings'))


# Run init_db in a background thread so gunicorn binds to the port immediately.
# On free-tier Render, cold PostgreSQL wakes slowly and blocks gunicorn startup
# causing Render's port scan to time out and roll back the deploy.
import threading
def _startup_init():
    with app.app_context():
        try:
            init_db()
        except Exception as e:
            log.error(f'init_db failed: {e}')

threading.Thread(target=_startup_init, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV', 'development') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
