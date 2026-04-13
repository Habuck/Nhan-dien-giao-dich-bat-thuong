import os, json, datetime, functools, sqlite3, random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import bcrypt, jwt, joblib, numpy as np
from sklearn.neighbors import NearestNeighbors

D = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(D, "fraud_detection.db")
ST = os.path.join(D, "static")
SK = os.environ.get("SECRET_KEY", "fraud-detection-secret-key")

app = Flask(__name__, static_folder=ST, static_url_path="/static")
CORS(app)
ml = joblib.load(os.path.join(D, "fraud_model.pkl"))

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init():
    d = db()
    d.executescript("""
      CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,role TEXT DEFAULT 'analyst',created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
      CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,features TEXT NOT NULL,prediction INTEGER NOT NULL,probability REAL NOT NULL,is_fraud INTEGER NOT NULL,amount REAL DEFAULT 0,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY(user_id)REFERENCES users(id));
      CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT,transaction_id INTEGER,alert_type TEXT NOT NULL,message TEXT,is_resolved INTEGER DEFAULT 0,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,FOREIGN KEY(transaction_id)REFERENCES transactions(id));
    """)
    if not d.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
        d.execute("INSERT INTO users(username,password_hash,role)VALUES(?,?,?)",("admin",bcrypt.hashpw(b"1234",bcrypt.gensalt()).decode(),"admin"))
        print("[+] admin/1234 created")
    d.commit(); d.close()

hp = lambda pw: bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
cp = lambda pw, h: bcrypt.checkpw(pw.encode(), h.encode())
mt = lambda uid, u, r: jwt.encode({"user_id":uid,"username":u,"role":r,"exp":datetime.datetime.utcnow()+datetime.timedelta(hours=24)}, SK, algorithm="HS256")

def auth(f):
    @functools.wraps(f)
    def w(*a, **k):
        t = request.headers.get("Authorization","").replace("Bearer ","")
        if not t: return jsonify({"error":"Token missing"}),401
        try: request.user = jwt.decode(t, SK, algorithms=["HS256"])
        except jwt.ExpiredSignatureError: return jsonify({"error":"Token expired"}),401
        except jwt.InvalidTokenError: return jsonify({"error":"Invalid token"}),401
        return f(*a, **k)
    return w

def _knn_anomaly_check(uid, amt):
    d = db()
    rows = d.execute("SELECT amount FROM transactions WHERE user_id=? ORDER BY created_at DESC LIMIT 100",(uid,)).fetchall()
    d.close()
    if len(rows) < 10: return False, 0.0
    amounts = np.array([r["amount"] for r in rows]).reshape(-1, 1)
    knn = NearestNeighbors(n_neighbors=min(5, len(amounts)))
    knn.fit(amounts)
    dist, _ = knn.kneighbors([[amt]])
    avg_dist = np.mean(dist)
    threshold = np.percentile(amounts, 95) * 2
    is_anomaly = amt > threshold or avg_dist > np.std(amounts) * 3
    anomaly_score = min(avg_dist / (np.std(amounts) + 1e-6), 1.0)
    return is_anomaly, float(anomaly_score)

def _predict(feats, uid, amt=None):
    arr = np.array(feats, dtype=float).reshape(1,-1)
    pred, prob = int(ml.predict(arr)[0]), float(ml.predict_proba(arr)[0][1])
    fraud = bool(pred == 1)
    if amt is None: amt = round(random.uniform(50000, 500000000))
    is_anomaly, anomaly_score = _knn_anomaly_check(uid, amt)
    d = db()
    cur = d.execute("INSERT INTO transactions(user_id,features,prediction,probability,is_fraud,amount)VALUES(?,?,?,?,?,?)",
                    (uid, json.dumps(feats), pred, prob, int(fraud), amt))
    tid = cur.lastrowid
    if fraud:
        d.execute("INSERT INTO alerts(transaction_id,alert_type,message)VALUES(?,?,?)",
                  (tid,"FRAUD_DETECTED",f"Fraud {prob:.2f} - {amt:,.0f} VND"))
    if is_anomaly:
        d.execute("INSERT INTO alerts(transaction_id,alert_type,message)VALUES(?,?,?)",
                  (tid,"AMOUNT_ANOMALY",f"Số tiền bất thường {amt:,.0f} VND (KNN score: {anomaly_score:.2f})"))
    d.commit(); d.close()
    return {"transaction_id":tid,"prediction":pred,"probability":prob,"is_fraud":fraud,"amount":amt,"is_amount_anomaly":bool(is_anomaly),"anomaly_score":anomaly_score}

@app.route("/api/auth/register", methods=["POST"])
def register():
    j = request.get_json(); u,p,r = j.get("username","").strip(), j.get("password",""), j.get("role","analyst")
    if not u or not p: return jsonify({"error":"Username and password required"}),400
    if len(p)<4: return jsonify({"error":"Password min 4 chars"}),400
    d = db()
    if d.execute("SELECT 1 FROM users WHERE username=?",(u,)).fetchone():
        d.close(); return jsonify({"error":"Username exists"}),409
    cur = d.execute("INSERT INTO users(username,password_hash,role)VALUES(?,?,?)",(u,hp(p),r))
    d.commit(); uid=cur.lastrowid; d.close()
    return jsonify({"token":mt(uid,u,r),"user":{"id":uid,"username":u,"role":r}}),201

@app.route("/api/auth/login", methods=["POST"])
def login():
    j = request.get_json(); u,p = j.get("username","").strip(), j.get("password","")
    if not u or not p: return jsonify({"error":"Username and password required"}),400
    d = db(); row = d.execute("SELECT * FROM users WHERE username=?",(u,)).fetchone(); d.close()
    if not row or not cp(p, row["password_hash"]): return jsonify({"error":"Invalid credentials"}),401
    return jsonify({"token":mt(row["id"],row["username"],row["role"]),"user":{"id":row["id"],"username":row["username"],"role":row["role"]}})

@app.route("/api/predict", methods=["POST"])
@auth
def predict():
    feats = request.get_json().get("features",[])
    if len(feats)!=30: return jsonify({"error":"Need 30 features"}),400
    try: return jsonify(_predict(feats, request.user["user_id"], request.get_json().get("amount")))
    except: return jsonify({"error":"Invalid features"}),400

@app.route("/api/predict/batch", methods=["POST"])
@auth
def predict_batch():
    txns = request.get_json().get("transactions",[])
    if not txns: return jsonify({"error":"No transactions"}),400
    res = []
    for t in txns:
        f = t.get("features",[])
        if len(f)!=30: res.append({"error":"Invalid feature count"}); continue
        res.append(_predict(f, request.user["user_id"], t.get("amount")))
    return jsonify({"results":res})

@app.route("/api/transactions", methods=["GET"])
@auth
def transactions():
    pg, pp = request.args.get("page",1,type=int), request.args.get("per_page",20,type=int)
    fo = request.args.get("fraud_only","false").lower()=="true"
    d = db(); uid = request.user["user_id"]
    w = "WHERE user_id=?" + (" AND is_fraud=1" if fo else "")
    total = d.execute(f"SELECT COUNT(*)as c FROM transactions {w}",(uid,)).fetchone()["c"]
    rows = d.execute(f"SELECT * FROM transactions {w} ORDER BY created_at DESC LIMIT ? OFFSET ?",(uid,pp,(pg-1)*pp)).fetchall()
    d.close()
    return jsonify({"transactions":[{"id":r["id"],"probability":r["probability"],"is_fraud":bool(r["is_fraud"]),
        "amount":r["amount"],"created_at":r["created_at"]} for r in rows],
        "total":total,"page":pg,"per_page":pp,"total_pages":(total+pp-1)//pp})

@app.route("/api/stats", methods=["GET"])
@auth
def stats():
    d = db()
    t = d.execute("SELECT COUNT(*)as c FROM transactions").fetchone()["c"]
    f = d.execute("SELECT COUNT(*)as c FROM transactions WHERE is_fraud=1").fetchone()["c"]
    r = d.execute("SELECT COUNT(*)as c FROM transactions WHERE is_fraud=1 AND created_at>=datetime('now','-24 hours')").fetchone()["c"]
    a = d.execute("SELECT COUNT(*)as c FROM alerts WHERE alert_type='AMOUNT_ANOMALY'").fetchone()["c"]
    d.close()
    return jsonify({"total_transactions":t,"fraud_count":f,"safe_count":t-f,"fraud_rate":round(f/t*100,2)if t else 0.0,"recent_fraud_24h":r,"amount_anomaly_count":a})

@app.route("/api/alerts", methods=["GET"])
@auth
def alerts():
    d = db()
    rows = d.execute("SELECT a.*,t.probability FROM alerts a JOIN transactions t ON a.transaction_id=t.id WHERE t.user_id=? ORDER BY a.created_at DESC LIMIT 50",(request.user["user_id"],)).fetchall()
    d.close()
    return jsonify({"alerts":[{"id":r["id"],"transaction_id":r["transaction_id"],"alert_type":r["alert_type"],
        "message":r["message"],"is_resolved":bool(r["is_resolved"]),"created_at":r["created_at"]} for r in rows]})

@app.route("/api/alerts/<int:aid>/resolve", methods=["PUT"])
@auth
def resolve(aid):
    d = db(); d.execute("UPDATE alerts SET is_resolved=1 WHERE id=?",(aid,)); d.commit(); d.close()
    return jsonify({"message":"Resolved"})

@app.route("/api/health")
def health(): return jsonify({"status":"ok","model":ml is not None})

@app.route("/")
def root(): return send_from_directory(ST,"index.html")

@app.route("/<path:f>")
def static_file(f): return send_from_directory(ST,f)

if __name__ == "__main__":
    init()
    print("[*] http://localhost:5002 | admin/1234")
    app.run(host="0.0.0.0", port=5002, debug=True)