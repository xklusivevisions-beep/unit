from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, get_flashed_messages
from datetime import datetime, timedelta
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import sqlite3, os, json, requests, logging, traceback, csv, io, secrets, re, base64
import pdfplumber
from PIL import Image
import anthropic

ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

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
    """Use Claude Vision to extract stops from a Speed X screenshot."""
    if not ANTHROPIC_KEY:
        return []
    try:
        img_bytes = compress_for_api(img_bytes)
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        b64    = base64.standard_b64encode(img_bytes).decode('utf-8')
        resp   = client.messages.create(
            model='claude-haiku-4-5-20251001',
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
Return ONLY a JSON array, no other text. No markdown, no code blocks, just the raw array.

SPEED X ADDRESS FORMAT — addresses are split across TWO lines:
  Line 1: street number + street name (may end with "Apt", "St", "Blvd," or just the street)
  Line 2: unit/apt number, City, STATE, ZIP, USA — all comma-separated, no spaces
Example:
  Line 1: "3439 Woodward Ave Apt"
  Line 2: "550,Detroit,MI,48201,USA"
  → Reconstruct as: "3439 Woodward Ave Apt 550, Detroit, MI 48201"

Another example:
  Line 1: "690 Brainard ST"
  Line 2: "405,DETROIT,MI,48201-2283,..."
  → Reconstruct as: "690 Brainard ST Apt 405, Detroit, MI 48201"

Another example:
  Line 1: "676 Martin Luther King Jr Blvd,"
  Line 2: "Apt 2c, MI,Detroit,MI,48201,USA"
  → Reconstruct as: "676 Martin Luther King Jr Blvd Apt 2C, Detroit, MI 48201"

Output format:
[{"stop_num": "51", "address": "690 Brainard ST Apt 405, Detroit, MI 48201", "name": "Ianita Manning", "tracking": "SPXDTW119702831650", "unit": "405", "phone": "3135550123"}]

Rules:
- Reconstruct the full address by combining both lines as shown above
- Remove ",USA" and trailing ",..." from addresses
- Normalize city to title case (Detroit not DETROIT)
- stop_num is the number after "Stop:" label (bottom-right of each card)
- tracking is the full SPXDTW code (blue text) — copy it exactly, it can be 18-24 chars
- name is the customer name (blue text next to phone icon) — expand truncated names if you can read enough, otherwise use what is visible
- unit is the apartment/unit number extracted from the address
- phone is the customer phone number if visible anywhere on the card (digits only, no dashes or spaces). If not visible, use empty string ""
- Include EVERY stop card visible on screen
- If address is truncated with "..." reconstruct as much as possible from visible text'''
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
                'stop_num': str(s.get('stop_num','')).strip(),
                'unit':     s.get('unit','').strip(),
                'phone':    re.sub(r'\D', '', s.get('phone',''))
            } for s in stops if s.get('address')]
    except Exception as e:
        log.error(f'Claude Vision error: {e}')
    return []


def extract_package_label(img_bytes):
    """Use Claude Vision to extract delivery info from a shipping label photo."""
    if not ANTHROPIC_KEY:
        return None
    try:
        img_bytes = compress_for_api(img_bytes)
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=512,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b64}
                    },
                    {
                        'type': 'text',
                        'text': '''This is a shipping label. Extract the delivery information.
Return ONLY a JSON object, no markdown, no code blocks.

Extract:
- tracking: the main tracking/barcode number (usually at bottom, longest number)
- name: recipient name ("Ship To" field)
- address: full delivery address as one string (street, city, state, zip)
- zip: just the 5-digit zip code

Example output:
{"tracking": "YWORD010176279569", "name": "Skye Scaglione", "address": "5750 Woodward Avenue 6, Detroit, MI 48202", "zip": "48202"}

If a field is not visible, use an empty string.
Return ONLY the JSON object.'''
                    }
                ]
            }]
        )
        text = resp.content[0].text.strip()
        # Strip markdown if model wrapped it
        if text.startswith('```'):
            text = re.sub(r'^```[a-z]*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        return json.loads(text)
    except Exception as e:
        log.error(f'extract_package_label error: {e}')
        return None


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
        {'id': 'trunk-back',   'label': 'Trunk — Back Wall',    'icon': '🔵', 'desc': 'Against the back wall of trunk'},
        {'id': 'trunk-mid',    'label': 'Trunk — Middle',       'icon': '🔵', 'desc': 'Middle of trunk'},
        {'id': 'trunk-front',  'label': 'Trunk — Front',       'icon': '🟡', 'desc': 'Front of trunk, near seat'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',   'icon': '🟠', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',    'icon': '🟠', 'desc': 'Left rear passenger seat'},
        {'id': 'backseat-mid', 'label': 'Back Seat — Middle',  'icon': '🟠', 'desc': 'Middle rear seat'},
    ],
    'sedan': [
        {'id': 'trunk-back',   'label': 'Trunk — Back Wall',    'icon': '🔵', 'desc': 'Against the back wall of trunk'},
        {'id': 'trunk-mid',    'label': 'Trunk — Middle',       'icon': '🔵', 'desc': 'Middle of trunk'},
        {'id': 'trunk-front',  'label': 'Trunk — Front',       'icon': '🟡', 'desc': 'Front of trunk, near seat'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',   'icon': '🟠', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',    'icon': '🟠', 'desc': 'Left rear passenger seat'},
        {'id': 'backseat-mid', 'label': 'Back Seat — Middle',  'icon': '🟠', 'desc': 'Middle rear seat'},
        {'id': 'front-pass',   'label': 'Front Passenger',      'icon': '⚪', 'desc': 'Front passenger seat floor/seat'},
    ],
    'suv': [
        {'id': 'cargo-back',   'label': 'Cargo — Back Wall',   'icon': '🔵', 'desc': 'Against rear cargo wall (load first)'},
        {'id': 'cargo-mid',    'label': 'Cargo — Middle',      'icon': '🔵', 'desc': 'Center of cargo area'},
        {'id': 'cargo-door',   'label': 'Cargo — Near Door',   'icon': '🟡', 'desc': 'Just inside rear door'},
        {'id': 'row3-r',       'label': '3rd Row — Right',     'icon': '🟠', 'desc': '3rd row folded flat, right side'},
        {'id': 'row3-l',       'label': '3rd Row — Left',      'icon': '🟠', 'desc': '3rd row folded flat, left side'},
        {'id': 'backseat-r',   'label': '2nd Row — Right',     'icon': '⚪', 'desc': 'Right rear passenger seat'},
        {'id': 'backseat-l',   'label': '2nd Row — Left',      'icon': '⚪', 'desc': 'Left rear passenger seat'},
        {'id': 'front-pass',   'label': 'Front Passenger',      'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'minivan': [
        {'id': 'cargo-back',   'label': 'Cargo — Back',        'icon': '🔵', 'desc': 'Rear cargo area behind seats'},
        {'id': 'row3-r',       'label': '3rd Row — Right',     'icon': '🔵', 'desc': 'Right side 3rd row (fold flat)'},
        {'id': 'row3-l',       'label': '3rd Row — Left',      'icon': '🔵', 'desc': 'Left side 3rd row (fold flat)'},
        {'id': 'row2-r',       'label': '2nd Row — Right',     'icon': '🟠', 'desc': 'Right captain\'s chair area'},
        {'id': 'row2-l',       'label': '2nd Row — Left',      'icon': '🟠', 'desc': 'Left captain\'s chair area'},
        {'id': 'row2-mid',     'label': '2nd Row — Middle',    'icon': '🟠', 'desc': 'Center aisle / middle row'},
        {'id': 'front-pass',   'label': 'Front Passenger',      'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'pickup': [
        {'id': 'bed-cab',      'label': 'Bed — Cab Wall',      'icon': '🔵', 'desc': 'Against cab — load first, stays dry if covered'},
        {'id': 'bed-mid',      'label': 'Bed — Middle',        'icon': '🔵', 'desc': 'Middle of truck bed'},
        {'id': 'bed-gate',     'label': 'Bed — Near Tailgate', 'icon': '🟡', 'desc': 'Near tailgate — grab first'},
        {'id': 'backseat-r',   'label': 'Back Seat — Right',   'icon': '🟠', 'desc': 'Crew cab right rear (if applicable)'},
        {'id': 'backseat-l',   'label': 'Back Seat — Left',    'icon': '🟠', 'desc': 'Crew cab left rear (if applicable)'},
        {'id': 'front-pass',   'label': 'Front Passenger',      'icon': '⚪', 'desc': 'Front passenger seat/floor'},
    ],
    'cargo_van': [
        {'id': 'A1', 'label': 'Zone A1 — Front Left',   'icon': '🔵', 'desc': 'Bulkhead left, load first'},
        {'id': 'A2', 'label': 'Zone A2 — Front Right',  'icon': '🔵', 'desc': 'Bulkhead right, load first'},
        {'id': 'B1', 'label': 'Zone B1 — Mid Left',     'icon': '🔵', 'desc': 'Mid-van left side'},
        {'id': 'B2', 'label': 'Zone B2 — Mid Right',    'icon': '🔵', 'desc': 'Mid-van right side'},
        {'id': 'C1', 'label': 'Zone C1 — Rear Left',    'icon': '🟡', 'desc': 'Near rear doors, left'},
        {'id': 'C2', 'label': 'Zone C2 — Rear Right',   'icon': '🟡', 'desc': 'Near rear doors, right'},
        {'id': 'C3', 'label': 'Zone C3 — Door Stack',   'icon': '🟡', 'desc': 'Stacked at rear door opening'},
    ],
    'box_truck': [
        {'id': 'A1', 'label': 'Row A — Left Front',  'icon': '🔵', 'desc': 'Front of box, driver side'},
        {'id': 'A2', 'label': 'Row A — Right Front', 'icon': '🔵', 'desc': 'Front of box, passenger side'},
        {'id': 'B1', 'label': 'Row B — Left Mid',    'icon': '🔵', 'desc': 'Middle of box, left'},
        {'id': 'B2', 'label': 'Row B — Right Mid',   'icon': '🔵', 'desc': 'Middle of box, right'},
        {'id': 'C1', 'label': 'Row C — Left Rear',   'icon': '🟡', 'desc': 'Near door, left side'},
        {'id': 'C2', 'label': 'Row C — Right Rear',  'icon': '🟡', 'desc': 'Near door, right side'},
        {'id': 'D1', 'label': 'Door Stack — Left',   'icon': '🟡', 'desc': 'Right at the door, grab first'},
        {'id': 'D2', 'label': 'Door Stack — Right',  'icon': '🟡', 'desc': 'Right at the door, grab first'},
    ],
}

VEHICLE_LABELS = {
    'small_car':  '🚗 Small Car',
    'sedan':      '🚗 Sedan',
    'suv':        '🚙 SUV / Crossover',
    'minivan':    '🚐 Minivan',
    'pickup':     '🚚 Pickup Truck',
    'cargo_van':  '🚐 Cargo Van',
    'box_truck':  '🚚 Box Truck',
}

def assign_vehicle_zones(sorted_pkgs, vehicle_type):
    """Assign a specific vehicle zone to each package based on delivery order.
    Delivery order 1 = deliver first = load LAST = zone closest to door.
    Delivery order N = deliver last = load FIRST = deepest zone.
    """
    zones = VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv'])
    total = len(sorted_pkgs)
    if total == 0:
        return sorted_pkgs
    zone_count = len(zones)
    for pkg in sorted_pkgs:
        delivery_order = pkg.get('delivery_order', 1)  # 1 = deliver first = load last = door
        # delivery_order=1    → door zone   (zone_count-1, last index)
        # delivery_order=total → deepest zone (index 0)
        if total > 1:
            zone_idx = int(round((delivery_order - 1) / (total - 1) * (zone_count - 1)))
        else:
            zone_idx = 0
        zone_idx = zone_count - 1 - zone_idx   # invert: high delivery_order = deep zone
        zone_idx = max(0, min(zone_idx, zone_count - 1))
        zone = zones[zone_idx]
        pkg['zone_id']    = zone['id']
        pkg['zone_label'] = zone['label']
        pkg['zone_icon']  = zone['icon']
        pkg['zone_desc']  = zone['desc']
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
        "ALTER TABLE drivers ADD COLUMN vehicle_type TEXT DEFAULT 'suv'",
        "ALTER TABLE drivers ADD COLUMN vehicle_capacity INTEGER DEFAULT 100",
        "ALTER TABLE drivers ADD COLUMN assigned_zips TEXT",
        "ALTER TABLE stops ADD COLUMN delivered_at TEXT",
        "ALTER TABLE routes ADD COLUMN est_distance_miles REAL",
        "ALTER TABLE routes ADD COLUMN est_duration_mins REAL",
        "ALTER TABLE routes ADD COLUMN route_started_at TEXT",
        "ALTER TABLE routes ADD COLUMN first_delivery_at TEXT",
        "CREATE TABLE IF NOT EXISTS pin_corrections (id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT UNIQUE NOT NULL, lat REAL NOT NULL, lng REAL NOT NULL, corrected_by TEXT, corrected_at TEXT DEFAULT CURRENT_TIMESTAMP)",
        "CREATE TABLE IF NOT EXISTS login_attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, ip TEXT NOT NULL, attempted_at TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts (ip, attempted_at)",
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
    ]:
        try:
            db.execute(migration)
            db.commit()
        except:
            try: db._conn.rollback()
            except: pass
    db.close()

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
    if address in _geocache: return _geocache[address]
    # Normalize spelled-out numbers (Eight Mile -> 8 Mile)
    normalized = _normalize_street_numbers(address)
    # Strip apt/unit suffixes before geocoding
    clean = re.sub(r'\s+(Apt|Unit|Suite|Ste|#)\s*[\w-]+', '', normalized, flags=re.IGNORECASE).strip()

    # 1. Try US Census Bureau (most accurate for US addresses, free, no key)
    lat, lng = _census_geocode(clean)
    if lat and lng:
        log.info(f'Census geocode hit: {address} -> {lat:.5f}, {lng:.5f}')
        _geocache[address] = (lat, lng)
        return lat, lng

    # 2. Fall back to Nominatim
    try:
        geo = Nominatim(user_agent='unit-delivery-app', timeout=8)
        loc = geo.geocode(clean) or geo.geocode(normalized) or geo.geocode(address)
        if loc:
            log.info(f'Nominatim fallback hit: {address} -> {loc.latitude:.5f}, {loc.longitude:.5f}')
            _geocache[address] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except Exception as e:
        log.warning(f'Nominatim geocode failed for {address}: {e}')

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
    return render_template('driver_dashboard.html', route=route, stops=stops, driver=session['driver_name'], gmaps_key=GOOGLE_MAPS_KEY, mapbox_token=MAPBOX_TOKEN)


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


@app.route('/driver/scan', methods=['GET'])
def scan_packages():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    ss_id = _get_or_create_scan_session(db, session['driver_id'])
    items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=? ORDER BY id ASC",
        (ss_id,)
    ).fetchall()
    db.close()
    return render_template('scan.html', items=items, session_id=ss_id, driver=session['driver_name'])


@app.route('/driver/scan/process', methods=['POST'])
def scan_process():
    """Receive label photo, run Claude Vision, return parsed JSON."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    file = request.files.get('photo')
    if not file:
        return jsonify({'ok': False, 'error': 'No photo received'})
    img_bytes = file.read()
    result = extract_package_label(img_bytes)
    if not result:
        return jsonify({'ok': False, 'error': 'Could not read label — try a clearer photo'})
    return jsonify({'ok': True, 'data': result})


@app.route('/driver/scan/add', methods=['POST'])
def scan_add():
    """Add a confirmed package to the scan session — geocodes immediately for live sort."""
    if 'driver_id' not in session:
        return jsonify({'ok': False, 'error': 'not logged in'}), 401
    data = request.get_json() or {}
    tracking = data.get('tracking', '').strip()
    name = data.get('name', '').strip()
    address = data.get('address', '').strip()
    zip_code = data.get('zip', '').strip()
    if not address:
        return jsonify({'ok': False, 'error': 'Address is required'})

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

    # Check zip against driver's assigned zips
    zip_warning = None
    driver_row = db.execute("SELECT assigned_zips FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    if driver_row and driver_row['assigned_zips'] and zip_code:
        assigned = [z.strip() for z in driver_row['assigned_zips'].split(',') if z.strip()]
        if assigned and zip_code not in assigned:
            zip_warning = f'ZIP {zip_code} is outside your assigned zone ({driver_row["assigned_zips"]})'

    db.execute(
        """INSERT INTO scan_items (session_id, tracking, customer_name, address, zip_code, raw_json, dest_lat, dest_lng)
           VALUES (?,?,?,?,?,?,?,?)""",
        (ss_id, tracking, name, address, zip_code, json.dumps(data), lat, lng)
    )
    db.commit()
    new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    count = db.execute("SELECT COUNT(*) FROM scan_items WHERE session_id=?", (ss_id,)).fetchone()[0]
    db.close()
    return jsonify({'ok': True, 'count': count, 'geocoded': lat is not None,
                   'new_item_id': new_id, 'zip_warning': zip_warning})


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
    # Get driver vehicle type
    driver = db.execute("SELECT vehicle_type FROM drivers WHERE id=?", (session['driver_id'],)).fetchone()
    vehicle_type = (driver['vehicle_type'] if driver and driver['vehicle_type'] else 'suv')
    db.close()

    if not items:
        return jsonify({'ok': True, 'items': [], 'sorted': False, 'vehicle_type': vehicle_type,
                        'vehicle_label': VEHICLE_LABELS.get(vehicle_type, 'Vehicle'),
                        'vehicle_zones': VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv'])})

    packages = []
    for item in items:
        packages.append({
            'id':       item['id'],
            'tracking': item['tracking'],
            'name':     item['customer_name'],
            'address':  item['address'],
            'lat':      item['dest_lat'],
            'lng':      item['dest_lng'],
        })

    # Split into geocoded and non-geocoded
    geocoded   = [p for p in packages if p['lat'] and p['lng']]
    ungeoced   = [p for p in packages if not (p['lat'] and p['lng'])]

    sorted_pkgs = []
    route_miles       = None
    route_drive_mins  = None
    naive_miles       = None
    savings_miles     = None

    if len(geocoded) >= 2:
        # ─ Naive distance (sequential scan order, no optimization) ─
        naive_dist_m = sum(
            geodesic((geocoded[i]['lat'], geocoded[i]['lng']),
                     (geocoded[i+1]['lat'], geocoded[i+1]['lng'])).meters
            for i in range(len(geocoded) - 1)
        )
        naive_miles = round(naive_dist_m * 0.000621371, 2)

        try:
            # OSRM trip optimization — request duration+distance in response
            # Split into chunks of 100 if >100 stops (OSRM public limit)
            chunk_size = 100
            if len(geocoded) <= chunk_size:
                chunks = [geocoded]
            else:
                chunks = [geocoded[i:i+chunk_size] for i in range(0, len(geocoded), chunk_size)]

            all_sorted = []
            total_dist_m = 0
            total_dur_s  = 0

            for chunk in chunks:
                coords = ';'.join(f"{p['lng']},{p['lat']}" for p in chunk)
                osrm_url = (f"http://router.project-osrm.org/trip/v1/driving/{coords}"
                            f"?roundtrip=false&source=first&destination=last&overview=false")
                resp = requests.get(osrm_url, timeout=8)
                if resp.status_code == 200:
                    trip_data = resp.json()
                    if trip_data.get('code') == 'Ok' and trip_data.get('waypoints'):
                        waypoints = sorted(trip_data['waypoints'], key=lambda w: w['waypoint_index'])
                        for wp in waypoints:
                            closest = min(
                                chunk,
                                key=lambda p: abs(p['lat'] - wp['location'][1]) + abs(p['lng'] - wp['location'][0])
                            )
                            all_sorted.append(closest)
                        # Accumulate distance + duration
                        if trip_data.get('trips'):
                            total_dist_m += trip_data['trips'][0].get('distance', 0)
                            total_dur_s  += trip_data['trips'][0].get('duration', 0)

            if all_sorted:
                # Deduplicate
                seen = set()
                for p in all_sorted:
                    if p['id'] not in seen:
                        seen.add(p['id'])
                        sorted_pkgs.append(p)
                route_miles      = round(total_dist_m * 0.000621371, 2)
                route_drive_mins = round(total_dur_s / 60, 1)
                savings_miles    = round(naive_miles - route_miles, 2) if naive_miles and route_miles else None
        except Exception as e:
            log.warning(f'OSRM live-sort failed: {e}')

    if not sorted_pkgs:
        # Nearest-neighbor fallback (pure Python)
        remaining = list(geocoded)
        if remaining:
            current = remaining.pop(0)
            sorted_pkgs = [current]
            while remaining:
                nearest = min(remaining, key=lambda p: geodesic(
                    (current['lat'], current['lng']), (p['lat'], p['lng'])
                ).miles)
                remaining.remove(nearest)
                sorted_pkgs.append(nearest)
                current = nearest

    # Append ungeocoded at the end
    seen_ids = {p['id'] for p in sorted_pkgs}
    for p in ungeoced:
        if p['id'] not in seen_ids:
            sorted_pkgs.append(p)
    # Also append any geocoded packages that got dropped
    for p in geocoded:
        if p['id'] not in seen_ids:
            sorted_pkgs.append(p)

    # Add load position (reverse of delivery order = how to load the truck)
    total = len(sorted_pkgs)
    for i, p in enumerate(sorted_pkgs):
        p['delivery_order'] = i + 1        # 1 = deliver first
        p['load_position']  = total - i    # highest = load first (goes in back of truck)

    # Assign vehicle zones based on driver's vehicle type
    sorted_pkgs = assign_vehicle_zones(sorted_pkgs, vehicle_type)

    # ETA projection: drive time + 3 min per stop
    est_total_mins = None
    if route_drive_mins is not None:
        est_total_mins = round(route_drive_mins + (total * 3), 1)

    return jsonify({
        'ok':              True,
        'items':           sorted_pkgs,
        'sorted':          len(geocoded) >= 2,
        'total':           total,
        'vehicle_type':    vehicle_type,
        'vehicle_label':   VEHICLE_LABELS.get(vehicle_type, 'Vehicle'),
        'vehicle_zones':   VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv']),
        # Route stats
        'route_miles':     route_miles,
        'route_drive_mins': route_drive_mins,
        'naive_miles':     naive_miles,
        'savings_miles':   savings_miles,
        'est_total_mins':  est_total_mins,
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
    vehicle_type = driver['vehicle_type'] if driver and driver['vehicle_type'] else 'suv'
    return jsonify({
        'ok': True,
        'current': vehicle_type,
        'label':   VEHICLE_LABELS.get(vehicle_type),
        'options': [{'value': k, 'label': v} for k, v in VEHICLE_LABELS.items()],
        'zones':   VEHICLE_ZONES.get(vehicle_type, VEHICLE_ZONES['suv']),
    })


@app.route('/driver/scan/remove/<int:item_id>', methods=['POST'])
def scan_remove(item_id):
    if 'driver_id' not in session:
        return jsonify({'ok': False}), 401
    db = get_db()
    db.execute("DELETE FROM scan_items WHERE id=?", (item_id,))
    db.commit()
    db.close()
    return jsonify({'ok': True})


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
        db.execute("UPDATE scan_sessions SET status='cleared' WHERE id=?", (ss['id'],))
        db.commit()
    db.close()
    return jsonify({'ok': True})


@app.route('/driver/scan/build-route', methods=['POST'])
def scan_build_route():
    """Geocode all scanned packages and create an optimized route."""
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
        return jsonify({'ok': False, 'error': 'No scan session found'})
    ss_id = ss['id']
    items = db.execute(
        "SELECT * FROM scan_items WHERE session_id=?", (ss_id,)
    ).fetchall()
    if not items:
        db.close()
        return jsonify({'ok': False, 'error': 'No packages scanned yet'})

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

    # Geocode and insert stops
    geocoded = 0
    failed_addresses = []
    for i, item in enumerate(items):
        lat, lng = None, None
        if item['address']:
            coords = geocode_address(item['address'])
            if coords:
                lat, lng = coords
                geocoded += 1
                # Update scan_items with geocoded coords
                db.execute(
                    "UPDATE scan_items SET dest_lat=?, dest_lng=? WHERE id=?",
                    (lat, lng, item['id'])
                )
            else:
                failed_addresses.append(item['address'])
        import secrets as _sec
        token = _sec.token_urlsafe(12)
        db.execute(
            """INSERT INTO stops
               (route_id, stop_number, address, customer_name, tracking, dest_lat, dest_lng, status, token)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (route_id, i + 1, item['address'], item['customer_name'],
             item['tracking'], lat, lng, 'pending', token)
        )
    db.commit()

    # Optimize using OSRM (nearest-neighbor fallback built into optimize_route)
    geocoded_stops = db.execute(
        "SELECT id, dest_lat, dest_lng FROM stops WHERE route_id=? AND dest_lat IS NOT NULL ORDER BY stop_number",
        (route_id,)
    ).fetchall()
    if len(geocoded_stops) >= 2:
        try:
            coords = ';'.join(f"{s['dest_lng']},{s['dest_lat']}" for s in geocoded_stops)
            osrm_url = (f"http://router.project-osrm.org/trip/v1/driving/{coords}"
                        f"?roundtrip=false&source=first&destination=last&overview=false")
            resp = requests.get(osrm_url, timeout=8)
            if resp.status_code == 200:
                trip = resp.json()
                if trip.get('code') == 'Ok' and trip.get('waypoints'):
                    waypoints = sorted(trip['waypoints'], key=lambda w: w['waypoint_index'])
                    stop_list = list(geocoded_stops)
                    for new_num, wp in enumerate(waypoints, 1):
                        # Match waypoint to closest stop
                        closest = min(
                            stop_list,
                            key=lambda s: abs(s['dest_lat'] - wp['location'][1]) + abs(s['dest_lng'] - wp['location'][0])
                        )
                        db.execute("UPDATE stops SET stop_number=? WHERE id=?", (new_num, closest['id']))
                    db.commit()
                    # Store route stats for ETA calculation
                    if trip.get('trips'):
                        dist_miles = round(trip['trips'][0].get('distance', 0) * 0.000621371, 2)
                        dur_mins   = round(trip['trips'][0].get('duration', 0) / 60, 1)
                        db.execute(
                            "UPDATE routes SET est_distance_miles=?, est_duration_mins=? WHERE id=?",
                            (dist_miles, dur_mins, route_id)
                        )
                        db.commit()
                    log.info(f'Scan route {route_id} optimized via OSRM')
        except Exception as e:
            log.warning(f'OSRM optimize on scan build failed: {e}')

    # Mark scan session as built
    db.execute("UPDATE scan_sessions SET status='built' WHERE id=?", (ss_id,))
    db.commit()
    db.close()
    return jsonify({
        'ok': True,
        'route_id': route_id,
        'total': len(items),
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
            'anthropic_key_set': bool(ANTHROPIC_KEY)
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
    db.close()
    total    = len(stops)
    with_phone = sum(1 for s in stops if s['phone'])
    return render_template('route_detail.html', route=route, stops=stops, total=total, with_phone=with_phone, mapbox_token=MAPBOX_TOKEN)

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
    if not stop:
        db.close()
        return redirect(url_for('driver_dashboard'))
    # Lazy geocode — if no pin yet, geocode now so map loads correctly
    if not stop['dest_lat']:
        lat, lng = geocode_address(stop['address'])
        if lat and lng:
            db.execute("UPDATE stops SET dest_lat=?, dest_lng=? WHERE id=?", (lat, lng, stop_id))
            db.commit()
            stop = db.execute("SELECT * FROM stops WHERE id=?", (stop_id,)).fetchone()
    db.close()
    return render_template('stop_active.html', stop=stop, gmaps_key=GOOGLE_MAPS_KEY, mapbox_token=MAPBOX_TOKEN)

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
    now_iso = datetime.now().isoformat()
    db.execute("UPDATE stops SET status='delivered', delivered_at=? WHERE id=?", (now_iso, stop_id))
    db.commit()
    route_id = stop['route_id'] if stop else None

    if route_id:
        route = db.execute("SELECT * FROM routes WHERE id=?", (route_id,)).fetchone()
        # Record first delivery time on the route
        if route and not route['first_delivery_at']:
            db.execute("UPDATE routes SET first_delivery_at=?, route_started_at=? WHERE id=?",
                       (now_iso, now_iso, route_id))
            db.commit()

    # Auto-advance: find next pending stop in route
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
                ok, _ = send_sms(format_phone(stop['phone']), msg)
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
    }
    db.close()
    return render_template('admin.html', routes=routes, buildings=buildings, deliveries=deliveries, stats=stats, drivers_list=drivers_list)

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
    return render_template('account.html', driver=session['driver_name'], phone=driver['phone'] or '')

@app.route('/account/edit', methods=['POST'])
def account_edit():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    name  = request.form.get('name', '').strip()
    phone = request.form.get('phone', '').strip()
    db = get_db()
    if name and phone:
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
        return jsonify({'status': 'ok', 'time': datetime.now().isoformat(), 'version': git_hash, 'model': 'claude-haiku-4-5-20251001'})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

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
