from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import sqlite3, os, json, requests, logging, traceback, csv, io, secrets, re, base64
import pdfplumber
from PIL import Image
import anthropic

ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

def extract_stops_from_image(img_bytes):
    """Use Claude Vision to extract stops from a Speed X screenshot."""
    if not ANTHROPIC_KEY:
        return []
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        b64    = base64.standard_b64encode(img_bytes).decode('utf-8')
        resp   = client.messages.create(
            model='claude-haiku-4-5',
            max_tokens=2048,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}
                    },
                    {
                        'type': 'text',
                        'text': '''This is a Speed X delivery app screenshot. Extract ALL delivery stops visible.
Return ONLY a JSON array, no other text. Format:
[{"stop_num": "88", "address": "18078 Joseph Campau, Detroit, MI 48201", "name": "John Smith", "tracking": "SPXDTW001234"}]
Rules:
- address must be full street address with city, state, zip
- Remove ",USA" from end of addresses
- stop_num is the number after "Stop:" label
- tracking starts with SPXDTW
- name is the customer name
- Include every stop visible on screen'''
                    }
                ]
            }]
        )
        text = resp.content[0].text.strip()
        # Extract JSON array from response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            stops = json.loads(match.group())
            return [{
                'address':  s.get('address','').strip(),
                'name':     s.get('name','').strip(),
                'tracking': s.get('tracking','').strip(),
                'stop_num': str(s.get('stop_num','')).strip()
            } for s in stops if s.get('address')]
    except Exception as e:
        log.error(f'Claude Vision error: {e}')
    return []
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
                # Skip SQLite-only pragmas
                if stmt.upper().startswith('PRAGMA'): continue
                try:
                    self._cur.execute(self._fix(stmt))
                except Exception: pass
        else:
            self._conn.executescript(script)
        return self

    def _to_dict(self, row):
        """Convert pg8000 tuple row to dict using cursor description."""
        if row is None: return None
        cols = [d[0] for d in self._cur.description]
        return dict(zip(cols, row))

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
TWILIO_SID   = os.environ.get('TWILIO_SID', '')
TWILIO_TOKEN = os.environ.get('TWILIO_TOKEN', '')
TWILIO_PHONE = os.environ.get('TWILIO_PHONE', '')
STRIPE_SECRET      = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUB_KEY     = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
STRIPE_PRICE_ID    = os.environ.get('STRIPE_PRICE_ID', 'price_1TZeWUEQpiT0nKEdHs158Phk')
stripe.api_key     = STRIPE_SECRET
APPROACH_RADIUS_MILES = 0.5
GEOFENCE_RADIUS_MILES  = 0.028
_geocache = {}

# ─── DB ────────────────────────────────────────────────────────

def get_db():
    if USE_PG:
        import urllib.parse
        url = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        p = urllib.parse.urlparse(url)
        conn = pg8000.connect(
            host=p.hostname,
            port=p.port or 5432,
            database=p.path.lstrip('/'),
            user=p.username,
            password=p.password,
            ssl_context=True
        )
        conn.autocommit = False
        return DBWrapper(conn, pg=True)
    else:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect(DB, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
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
    ''')
    db.commit()
    try:
        db.execute("INSERT INTO drivers (name, phone, company, pin) VALUES (?,?,?,?)",
                   ('Director X', '3135550000', 'SpeedX', '1234'))
        db.commit()
    except: pass
    # Safe migrations — add columns if they don't exist yet
    for migration in [
        "ALTER TABLE stops ADD COLUMN drop_spot TEXT",
        "ALTER TABLE residents ADD COLUMN sms_consent INTEGER DEFAULT 0",
        "ALTER TABLE residents ADD COLUMN sms_consent_at TEXT",
        "CREATE TABLE IF NOT EXISTS pin_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT UNIQUE NOT NULL, lat REAL NOT NULL, lng REAL NOT NULL, corrected_by TEXT, corrected_at TEXT DEFAULT CURRENT_TIMESTAMP)",
    ]:
        try:
            db.execute(migration)
            db.commit()
        except: pass
    db.close()

# ─── HELPERS ───────────────────────────────────────────────────

def geocode_address(address):
    if address in _geocache: return _geocache[address]
    try:
        geo = Nominatim(user_agent='unit-delivery-app', timeout=8)
        loc = geo.geocode(address)
        if loc:
            _geocache[address] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except Exception as e:
        log.warning(f'Geocode failed for {address}: {e}')
    _geocache[address] = (None, None)
    return None, None

TEXTBELT_KEY = os.environ.get('TEXTBELT_KEY', '')

def send_sms(to_phone, message):
    # Use Textbelt if key provided (no A2P registration needed)
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

    # Fallback to Twilio
    if not TWILIO_SID or not TWILIO_TOKEN:
        log.info(f'[SMS MOCK] To: {to_phone} | {message[:80]}')
        return True, 'mock'
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=message, from_=TWILIO_PHONE, to=to_phone)
        log.info(f'SMS sent to {to_phone}: {msg.sid}')
        return True, msg.sid
    except Exception as e:
        log.error(f'SMS failed to {to_phone}: {e}')
        return False, str(e)

def miles_away(lat1, lng1, lat2, lng2):
    try:
        if None in (lat1, lng1, lat2, lng2): return 999
        return geodesic((lat1, lng1), (lat2, lng2)).miles
    except: return 999

def get_base_url():
    return request.host_url.rstrip('/')

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
    if request.method == 'POST':
        pin = request.form.get('pin', '').strip()
        db = get_db()
        driver = db.execute("SELECT * FROM drivers WHERE pin=?", (pin,)).fetchone()
        db.close()
        if driver:
            session['driver_id'] = driver['id']
            session['driver_name'] = driver['name']
            return redirect(url_for('driver_dashboard'))
        error = 'Invalid PIN'
    return render_template('driver_login.html', error=error)

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
    db.close()
    return render_template('driver_dashboard.html', route=route, stops=stops, driver=session['driver_name'])

# ─── ROUTE IMPORT ──────────────────────────────────────────────

@app.route('/driver/route/new', methods=['GET', 'POST'])
def route_new():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))

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

        for route_file in route_files:
            if not route_file or not route_file.filename: continue
            fname = route_file.filename.lower()

            # ── IMAGE / SCREENSHOT — Claude Vision ──
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_bytes = route_file.read()
                    # Convert to JPEG for smaller payload to Claude
                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=85)
                    img_bytes = buf.getvalue()
                    stops_from_img = extract_stops_from_image(img_bytes)
                    for s in stops_from_img:
                        key = s.get('tracking') or s.get('address', '')
                        if key and key not in collected:
                            collected[key] = s
                except Exception as img_err:
                    log.error(f'Image processing error on {fname}: {img_err}')
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
            resident  = db.execute("SELECT * FROM residents WHERE address LIKE ?", (f'%{street}%',)).fetchone()
            phone      = resident['phone']      if resident else ''
            drop_spot  = resident['drop_spot']  if resident else ''
            door_notes = resident['door_notes'] if resident else ''
            unit       = s.get('unit', '') or (resident['unit'] if resident and resident.get('unit') else '')
            # Building access
            building   = db.execute("SELECT * FROM buildings WHERE address LIKE ?", (f'%{street}%',)).fetchone()
            access_note = ''
            if building:
                parts = []
                if building['access_code']:        parts.append(f"Code: {building['access_code']}")
                if building['buzzer_notes']:        parts.append(f"Buzzer: {building['buzzer_notes']}")
                if building['interior_directions']: parts.append(building['interior_directions'])
                access_note = ' | '.join(parts)
            notes     = ' | '.join(filter(None, [door_notes, access_note]))
            correction = db.execute("SELECT lat, lng FROM pin_corrections WHERE address=?", (full_addr,)).fetchone()
            saved_lat  = correction['lat'] if correction else None
            saved_lng  = correction['lng'] if correction else None
            stop_num   = s.get('stop_num') or idx + 1
            token      = secrets.token_urlsafe(12)
            db.execute(
                "INSERT INTO stops (route_id, stop_number, address, unit, customer_name, tracking, phone, drop_spot, notes, token, dest_lat, dest_lng) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (route_id, stop_num, full_addr, unit, s.get('name',''), s.get('tracking',''), phone, drop_spot, notes, token, saved_lat, saved_lng)
            )
            stops_added += 1

        db.commit()
        db.close()

        if stops_added == 0:
            # Nothing parsed — go to manual stop entry
            return redirect(url_for('route_manual'))

        return redirect(url_for('route_detail', route_id=route_id))

    return render_template('route_new.html')

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
            correction = db.execute(
                "SELECT lat, lng FROM pin_corrections WHERE address=?",
                (clean_addr,)
            ).fetchone()
            saved_lat = correction['lat'] if correction else None
            saved_lng = correction['lng'] if correction else None
            db.execute(
                "INSERT INTO stops (route_id, stop_number, address, unit, customer_name, phone, token, dest_lat, dest_lng) VALUES (?,?,?,?,?,?,?,?,?)",
                (route_id, i+1, clean_addr,
                 units[i].strip() if i < len(units) else '',
                 names[i].strip() if i < len(names) else '',
                 format_phone(phones[i].strip()) if i < len(phones) and phones[i].strip() else '',
                 token, saved_lat, saved_lng)
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
    db.close()
    total    = len(stops)
    with_phone = sum(1 for s in stops if s['phone'])
    return render_template('route_detail.html', route=route, stops=stops, total=total, with_phone=with_phone)

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
        address  = request.form.get('address', '').strip()
        unit     = request.form.get('unit', '').strip()
        name     = request.form.get('name', '').strip()
        phone    = format_phone(request.form.get('phone', '').strip()) if request.form.get('phone', '').strip() else ''
        notes    = request.form.get('notes', '').strip()

        # Re-geocode if address changed
        lat, lng = stop['dest_lat'], stop['dest_lng']
        if address != stop['address']:
            lat, lng = geocode_address(address)

        db.execute(
            "UPDATE stops SET address=?, unit=?, customer_name=?, phone=?, notes=?, dest_lat=?, dest_lng=?, status='pending', approach_sms_sent=0 WHERE id=?",
            (address, unit, name, phone, notes, lat, lng, stop_id)
        )
        db.commit()
        route_id = stop['route_id']
        db.close()
        return redirect(url_for('route_detail', route_id=route_id))

    route_id = stop['route_id']
    db.close()
    return render_template('stop_edit.html', stop=stop, route_id=route_id)

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
        ok, _ = send_sms(format_phone(stop['phone']), msg)
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
    db.close()
    if not stop: return redirect(url_for('driver_dashboard'))
    return render_template('stop_active.html', stop=stop)

@app.route('/driver/stop/<int:stop_id>/pin', methods=['POST'])
def stop_pin(stop_id):
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    data = request.get_json()
    lat, lng = data.get('lat'), data.get('lng')
    db = get_db()
    # Save to this stop
    stop = db.execute("SELECT address FROM stops WHERE id=?", (stop_id,)).fetchone()
    db.execute("UPDATE stops SET dest_lat=?, dest_lng=?, approach_sms_sent=0 WHERE id=?", (lat, lng, stop_id))
    # Save permanently to pin_corrections — survives future routes
    if stop:
        db.execute('''
            INSERT INTO pin_corrections (address, lat, lng, corrected_by, corrected_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                lat=excluded.lat,
                lng=excluded.lng,
                corrected_by=excluded.corrected_by,
                corrected_at=excluded.corrected_at
        ''', (stop['address'], lat, lng, session.get('driver_name', 'driver'), datetime.now().isoformat()))
    db.commit()
    db.close()
    return jsonify({'ok': True})

@app.route('/driver/stop/<int:stop_id>/delivered', methods=['POST'])
def stop_delivered(stop_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    db.execute("UPDATE stops SET status='delivered' WHERE id=?", (stop_id,))
    db.commit()
    # Redirect back to route
    route_id = stop['route_id'] if stop else None
    db.close()
    return redirect(url_for('route_detail', route_id=route_id) if route_id else url_for('driver_dashboard'))

@app.route('/driver/stop/<int:stop_id>/failed', methods=['POST'])
def stop_failed(stop_id):
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    db.execute("UPDATE stops SET status='failed' WHERE id=?", (stop_id,))
    db.commit()
    route_id = stop['route_id'] if stop else None
    db.close()
    return redirect(url_for('route_detail', route_id=route_id) if route_id else url_for('driver_dashboard'))

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
                """SELECT s.*, r.name as route_name, r.date as route_date
                   FROM stops s
                   LEFT JOIN routes r ON s.route_id = r.id
                   WHERE s.address LIKE ?
                   ORDER BY s.created_at DESC LIMIT 50""",
                (f'%{street}%',)
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
        """SELECT DISTINCT address, unit, customer_name FROM stops
           WHERE address LIKE ? ORDER BY rowid DESC LIMIT 8""",
        (f'%{q}%',)
    ).fetchall()
    db.close()
    return jsonify([{'address': r['address'], 'unit': r['unit'] or '', 'name': r['customer_name'] or ''} for r in results])

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
                ok, _ = send_sms(format_phone(stop['phone']), msg)
                if ok:
                    db.execute("UPDATE stops SET approach_sms_sent=1 WHERE id=?", (stop_id,))
                    result['sms_triggered'] = True

    db.commit()
    db.close()
    return jsonify(result)

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

ADMIN_PIN = os.environ.get('ADMIN_PIN', '8798')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        if request.form.get('pin', '').strip() == ADMIN_PIN:
            session['admin'] = True
            return redirect(url_for('admin'))
        error = 'Wrong PIN'
    return render_template('admin_login.html', error=error)

@app.route('/admin/bypass/unit8798')
def admin_bypass():
    session['admin'] = True
    return redirect(url_for('admin'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    db = get_db()
    routes    = db.execute("SELECT * FROM routes ORDER BY created_at DESC LIMIT 20").fetchall()
    buildings = db.execute("SELECT * FROM buildings ORDER BY confirmed_count DESC").fetchall()
    stats = {
        'total_deliveries': db.execute("SELECT COUNT(*) FROM stops").fetchone()[0],
        'delivered':        db.execute("SELECT COUNT(*) FROM stops WHERE status='delivered'").fetchone()[0],
        'failed':           db.execute("SELECT COUNT(*) FROM stops WHERE status='failed'").fetchone()[0],
        'sms_sent':         db.execute("SELECT COUNT(*) FROM stops WHERE approach_sms_sent=1").fetchone()[0],
        'buildings':        db.execute("SELECT COUNT(*) FROM buildings").fetchone()[0],
        'residents':        db.execute("SELECT COUNT(*) FROM residents").fetchone()[0],
    }
    db.close()
    return render_template('admin.html', routes=routes, buildings=buildings, stats=stats)

# ─── TEST SMS ─────────────────────────────────────────────────

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
    return render_template('account.html', driver=session['driver_name'])

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
        return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

with app.app_context():
    init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV', 'development') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
