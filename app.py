from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import sqlite3, os, json, requests, logging, traceback
from twilio.rest import Client

# ─── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'unit-secret-2025')

# ─── GLOBAL ERROR HANDLERS ─────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('error.html', code=404, msg='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    log.error(f'500 error: {traceback.format_exc()}')
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Server error'}), 500
    return render_template('error.html', code=500, msg='Something went wrong — we\'re on it'), 500

@app.errorhandler(Exception)
def unhandled(e):
    log.error(f'Unhandled exception: {traceback.format_exc()}')
    if request.path.startswith('/api/'):
        return jsonify({'error': str(e)}), 500
    return render_template('error.html', code=500, msg='Unexpected error — please try again'), 500

DB = 'data/unit.db'

TWILIO_SID   = os.environ.get('TWILIO_SID', '')
TWILIO_TOKEN = os.environ.get('TWILIO_TOKEN', '')
TWILIO_PHONE = os.environ.get('TWILIO_PHONE', '')

# How close (miles) before auto-SMS fires
APPROACH_RADIUS_MILES = 0.5

# ─── DB ────────────────────────────────────────────────────────

def get_db():
    return safe_db()

def init_db():
    db = get_db()
    db.executescript('''
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

        CREATE TABLE IF NOT EXISTS residents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            unit TEXT NOT NULL,
            phone TEXT NOT NULL,
            backup_phone TEXT,
            drop_spot TEXT,
            door_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER,
            driver_name TEXT,
            address TEXT,
            unit TEXT,
            tracking TEXT,
            dest_lat REAL,
            dest_lng REAL,
            driver_lat REAL,
            driver_lng REAL,
            status TEXT DEFAULT 'pending',
            sms_sent INTEGER DEFAULT 0,
            approach_sms_sent INTEGER DEFAULT 0,
            resident_confirmed INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            delivered_at TEXT,
            notes TEXT
        );

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
    ''')
    db.commit()

    # Seed driver
    try:
        db.execute("INSERT INTO drivers (name, phone, company, pin) VALUES (?,?,?,?)",
                   ('Director X', '3135550000', 'SpeedX', '1234'))
        db.commit()
    except:
        pass

    # Seed buildings
    buildings = [
        ('4500 Cass Ave, Detroit, MI 48201',   '4500#', 'Press 4500 then #', 'Elevator to floor 9, turn right', 'code',      42.3534, -83.0654),
        ('4701 Chrysler Dr, Detroit, MI 48201', None,   'Buzzer broken — text resident', '3rd floor east wing', 'text_only', 42.3612, -83.0481),
        ('430 E Warren Ave, Detroit, MI 48201', '7721', 'Press 7721', 'Elevator left of lobby, 2nd floor', 'code',           42.3505, -83.0603),
        ('4647 Chrysler Dr, Detroit, MI 48201', None,   'Key fob — text resident', 'Unit 102 ground floor left', 'text_only',42.3599, -83.0483),
        ('3150 Woodward Ave, Detroit, MI 48201','3150', 'Front keypad 3150#', 'Main elevator, floors 3-5 right hall', 'code', 42.3458, -83.0516),
    ]
    for b in buildings:
        try:
            db.execute("INSERT INTO buildings (address,access_code,buzzer_notes,interior_directions,access_type,lat,lng) VALUES (?,?,?,?,?,?,?)", b)
            db.commit()
        except:
            pass
    db.close()

# ─── HELPERS ───────────────────────────────────────────────────

# Simple geocode cache — avoids hammering Nominatim
_geocache = {}

def geocode_address(address):
    if address in _geocache:
        return _geocache[address]
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

def send_sms(to_phone, message):
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
        if None in (lat1, lng1, lat2, lng2):
            return 999
        return geodesic((lat1, lng1), (lat2, lng2)).miles
    except Exception as e:
        log.warning(f'Distance calc failed: {e}')
        return 999

def safe_db():
    """Get DB connection with WAL mode for concurrent access."""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn

def get_base_url():
    return request.host_url.rstrip('/')

# ─── DRIVER ────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/driver')
def driver_dashboard():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    db = get_db()
    recent = db.execute(
        "SELECT * FROM deliveries WHERE driver_id=? ORDER BY timestamp DESC LIMIT 15",
        (session['driver_id'],)
    ).fetchall()
    db.close()
    return render_template('driver_dashboard.html', deliveries=recent, driver=session['driver_name'])

@app.route('/driver/lookup', methods=['GET', 'POST'])
def driver_lookup():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))

    building = None
    resident = None
    error = None
    address = ''
    unit = ''
    delivery_id = None
    sms_sent = False

    if request.method == 'POST':
        address  = request.form.get('address', '').strip()
        unit     = request.form.get('unit', '').strip()
        tracking = request.form.get('tracking', '').strip()
        db = get_db()

        street = address.split(',')[0].strip()

        building = db.execute("SELECT * FROM buildings WHERE address LIKE ?", (f'%{street}%',)).fetchone()

        if unit:
            resident = db.execute(
                "SELECT * FROM residents WHERE address LIKE ? AND unit=?",
                (f'%{street}%', unit)
            ).fetchone()

        # Geocode destination
        dest_lat, dest_lng = None, None
        if building and building['lat']:
            dest_lat, dest_lng = building['lat'], building['lng']
        else:
            dest_lat, dest_lng = geocode_address(address)
            if building and dest_lat:
                db.execute("UPDATE buildings SET lat=?, lng=? WHERE id=?",
                           (dest_lat, dest_lng, building['id']))
                db.commit()

        # Create delivery record
        db.execute(
            "INSERT INTO deliveries (driver_id,driver_name,address,unit,tracking,dest_lat,dest_lng,status) VALUES (?,?,?,?,?,?,?,?)",
            (session['driver_id'], session['driver_name'], address, unit, tracking, dest_lat, dest_lng, 'en_route')
        )
        db.commit()
        delivery_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.close()

        if not building and not resident:
            error = 'Building not in database yet. Add it below.'

    return render_template('driver_lookup.html',
                           building=building, resident=resident,
                           address=address, unit=unit,
                           sms_sent=sms_sent, error=error,
                           delivery_id=delivery_id)

# ─── LOCATION API (called from driver browser JS) ─────────────

@app.route('/api/location', methods=['POST'])
def update_location():
    """Driver browser posts GPS coords every 10 seconds."""
    if 'driver_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')
    delivery_id = data.get('delivery_id')

    if not lat or not lng:
        return jsonify({'error': 'no coords'}), 400

    db = get_db()

    # Update driver location
    db.execute("UPDATE drivers SET current_lat=?, current_lng=?, last_seen=? WHERE id=?",
               (lat, lng, datetime.now().isoformat(), session['driver_id']))

    result = {'status': 'ok', 'sms_triggered': False}

    if delivery_id:
        # Update delivery driver location
        db.execute("UPDATE deliveries SET driver_lat=?, driver_lng=? WHERE id=?",
                   (lat, lng, delivery_id))

        delivery = db.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()

        if delivery and delivery['dest_lat'] and not delivery['approach_sms_sent']:
            distance = miles_away(lat, lng, delivery['dest_lat'], delivery['dest_lng'])

            if distance <= APPROACH_RADIUS_MILES:
                # Find resident
                street = delivery['address'].split(',')[0].strip()
                resident = db.execute(
                    "SELECT * FROM residents WHERE address LIKE ? AND unit=?",
                    (f'%{street}%', delivery['unit'] or '')
                ).fetchone()

                if resident and resident['phone']:
                    mins = max(1, int(distance * 3))  # rough ETA
                    track_url = f"{get_base_url()}/track/{delivery_id}"
                    msg = (f"Hi! Your UNIT driver is {mins} min away with your package"
                           f"{' for unit ' + delivery['unit'] if delivery['unit'] else ''}.\n"
                           f"Track live: {track_url}\n"
                           f"Drop spot: {resident['drop_spot'] or 'front door'} if no answer.")
                    ok, _ = send_sms(resident['phone'], msg)
                    if ok:
                        db.execute("UPDATE deliveries SET approach_sms_sent=1, sms_sent=1 WHERE id=?",
                                   (delivery_id,))
                        result['sms_triggered'] = True
                        result['distance_miles'] = round(distance, 2)

                # Even if no resident — mark so we don't keep checking
                db.execute("UPDATE deliveries SET approach_sms_sent=1 WHERE id=?", (delivery_id,))

        result['distance_miles'] = round(
            miles_away(lat, lng, delivery['dest_lat'], delivery['dest_lng']), 2
        ) if delivery and delivery['dest_lat'] else None

    db.commit()
    db.close()
    return jsonify(result)

# ─── LIVE TRACKING PAGE (for residents) ───────────────────────

@app.route('/track/<int:delivery_id>')
def track(delivery_id):
    db = get_db()
    delivery = db.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    db.close()
    if not delivery:
        return "Delivery not found", 404
    return render_template('track.html', delivery=delivery)

@app.route('/api/track/<int:delivery_id>')
def track_api(delivery_id):
    """Polling endpoint — resident page calls this every 5s."""
    db = get_db()
    delivery = db.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    db.close()
    if not delivery:
        return jsonify({'error': 'not found'}), 404

    distance = None
    if delivery['driver_lat'] and delivery['dest_lat']:
        distance = round(miles_away(
            delivery['driver_lat'], delivery['driver_lng'],
            delivery['dest_lat'],   delivery['dest_lng']
        ), 2)

    return jsonify({
        'driver_lat':  delivery['driver_lat'],
        'driver_lng':  delivery['driver_lng'],
        'dest_lat':    delivery['dest_lat'],
        'dest_lng':    delivery['dest_lng'],
        'status':      delivery['status'],
        'address':     delivery['address'],
        'unit':        delivery['unit'],
        'distance_miles': distance
    })

# ─── DELIVERY ACTIONS ──────────────────────────────────────────

@app.route('/driver/confirm/<int:delivery_id>')
def confirm_delivery(delivery_id):
    db = get_db()
    db.execute("UPDATE deliveries SET status='delivered', delivered_at=? WHERE id=?",
               (datetime.now().isoformat(), delivery_id))
    db.commit()
    db.close()
    return redirect(url_for('driver_dashboard'))

@app.route('/driver/failed/<int:delivery_id>')
def failed_delivery(delivery_id):
    db = get_db()
    db.execute("UPDATE deliveries SET status='failed' WHERE id=?", (delivery_id,))
    db.commit()
    db.close()
    return redirect(url_for('driver_dashboard'))

@app.route('/driver/add_building', methods=['POST'])
def add_building():
    if 'driver_id' not in session:
        return redirect(url_for('driver_login'))
    address      = request.form.get('address')
    access_code  = request.form.get('access_code')
    buzzer_notes = request.form.get('buzzer_notes')
    interior     = request.form.get('interior_directions')
    access_type  = request.form.get('access_type', 'code')
    lat, lng     = geocode_address(address)
    db = get_db()
    try:
        db.execute(
            "INSERT INTO buildings (address,access_code,buzzer_notes,interior_directions,access_type,lat,lng) VALUES (?,?,?,?,?,?,?)",
            (address, access_code, buzzer_notes, interior, access_type, lat, lng)
        )
    except:
        db.execute(
            "UPDATE buildings SET access_code=?,buzzer_notes=?,interior_directions=?,access_type=?,confirmed_count=confirmed_count+1 WHERE address LIKE ?",
            (access_code, buzzer_notes, interior, access_type, f'%{address.split(",")[0].strip()}%')
        )
    db.commit()
    db.close()
    return redirect(url_for('driver_lookup'))

# ─── RESIDENT ──────────────────────────────────────────────────

@app.route('/resident', methods=['GET', 'POST'])
def resident_portal():
    success = False
    if request.method == 'POST':
        db = get_db()
        db.execute(
            "INSERT INTO residents (address,unit,phone,backup_phone,drop_spot,door_notes) VALUES (?,?,?,?,?,?)",
            (request.form.get('address'), request.form.get('unit'),
             request.form.get('phone'),   request.form.get('backup_phone'),
             request.form.get('drop_spot'), request.form.get('door_notes'))
        )
        db.commit()
        db.close()
        success = True
    return render_template('resident_portal.html', success=success)

# ─── ADMIN ─────────────────────────────────────────────────────

@app.route('/admin')
def admin():
    db = get_db()
    buildings  = db.execute("SELECT * FROM buildings ORDER BY confirmed_count DESC").fetchall()
    deliveries = db.execute("SELECT * FROM deliveries ORDER BY timestamp DESC LIMIT 50").fetchall()
    residents  = db.execute("SELECT * FROM residents ORDER BY created_at DESC").fetchall()
    stats = {
        'total':     db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0],
        'delivered': db.execute("SELECT COUNT(*) FROM deliveries WHERE status='delivered'").fetchone()[0],
        'failed':    db.execute("SELECT COUNT(*) FROM deliveries WHERE status='failed'").fetchone()[0],
        'sms':       db.execute("SELECT COUNT(*) FROM deliveries WHERE sms_sent=1").fetchone()[0],
        'buildings': db.execute("SELECT COUNT(*) FROM buildings").fetchone()[0],
        'residents': db.execute("SELECT COUNT(*) FROM residents").fetchone()[0],
    }
    db.close()
    return render_template('admin.html', buildings=buildings, deliveries=deliveries,
                           residents=residents, stats=stats)

@app.route('/health')
def health():
    """Railway/uptime monitoring health check."""
    try:
        db = get_db()
        db.execute('SELECT 1').fetchone()
        db.close()
        return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV', 'development') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
