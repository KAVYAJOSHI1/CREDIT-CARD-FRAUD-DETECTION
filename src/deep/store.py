"""SQLite transaction log: every scored transaction is persisted with its model outputs."""
import json, sqlite3, time

DB_PATH = "db/fraudshield.sqlite"
SCHEMA = """CREATE TABLE IF NOT EXISTS scored_transactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL, source TEXT,
  amount REAL, tx_time REAL, features TEXT,
  probability REAL, resmlp REAL, ft_transformer REAL, ae_error REAL, uncertainty REAL,
  level TEXT, status TEXT, true_label INTEGER, top_features TEXT)"""


def connect():
    import os; os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; c.execute(SCHEMA); return c


def insert(c, source, raw_row, res, level, true_label=None, created_at=None):
    c.execute("""INSERT INTO scored_transactions(created_at,source,amount,tx_time,features,probability,resmlp,
                 ft_transformer,ae_error,uncertainty,level,status,true_label,top_features) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (created_at or time.time(), source, float(raw_row[-1]), float(raw_row[0]), json.dumps([float(v) for v in raw_row]),
               res["probability"], res["resmlp"], res["ft_transformer"], res["ae_error"], res["uncertainty"], level,
               "blocked" if res["is_fraud"] else "approved", None if true_label is None else int(true_label),
               json.dumps(res.get("top_features", []))))
    c.commit()


def rows(c, where="", args=(), limit=50):
    q = f"SELECT * FROM scored_transactions {where} ORDER BY created_at DESC LIMIT ?"
    return [dict(r) for r in c.execute(q, (*args, limit))]
