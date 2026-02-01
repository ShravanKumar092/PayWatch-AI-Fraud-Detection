# ==================================================
# MODULE PATH FIX (must be first)
# ==================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

# Import feature_engineering from src directory
from src.feature_engineering import feature_engineering

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import random
import asyncio
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional
import csv
# from api.services.fraud_engine import fraud_decision
# from api.services.explain import explain

from api.auth.routes import router as auth_router
from api.auth.security_v2 import verify_token

# feature engineering import with fallback


# 🔄 IMPROVED: Make Redis optional with fakeredis fallback
# Ensure names exist for static analysis
redis: Any = None
fakeredis: Any = None
REDIS_AVAILABLE_IMPORT = False
FAKEREDIS_AVAILABLE = False
try:
    import redis as _redis  # type: ignore
    redis = _redis
    REDIS_AVAILABLE_IMPORT = True
except Exception:
    REDIS_AVAILABLE_IMPORT = False
    print(">>> Redis module not installed. Running without Redis caching.")

# Try to use fakeredis on Windows as fallback
try:
    import fakeredis as _fakeredis  # type: ignore
    fakeredis = _fakeredis
    FAKEREDIS_AVAILABLE = True
except Exception:
    FAKEREDIS_AVAILABLE = False



print(">>> API FILE LOADED FROM:", __file__)

# ==================================================
# FASTAPI APP
# ==================================================
app = FastAPI(
    title="PayWatch AI – Fraud Detection API",
    version="2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables on startup
@app.on_event("startup")
async def init_database():
    global transaction_counter, high_risk_counter
    
    # Reset global counters on startup for fresh evaluation count
    transaction_counter = 0
    high_risk_counter = 0
    
    try:
        from api.auth.db import Base, engine
        from api.auth.models import User
        Base.metadata.create_all(bind=engine)
        print(">>> Database tables initialized")
    except Exception as e:
        print(f">>> Database initialization warning: {e}")
    
    # Reset Redis counters on startup for fresh evaluation count
    if REDIS_AVAILABLE and redis_client is not None:
        try:
            rc = redis_client
            if hasattr(rc, "set"):
                rc.set("total_transactions", 0)
                rc.set("high_risk_count", 0)
                print(">>> Redis counters initialized to 0 for evaluation tracking")
        except Exception as e:
            print(f">>> Redis counter initialization warning: {e}")
    
    print(f">>> Global counters reset: transaction_counter={transaction_counter}, high_risk_counter={high_risk_counter}")


# ==================================================
# 🔄 REAL-TIME UPGRADE: REDIS CONNECTION POOL (OPTIONAL)
# ==================================================
REDIS_AVAILABLE = False
redis_client: Optional[Any] = None

if REDIS_AVAILABLE_IMPORT:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    try:
        # Connection pool for better performance
        redis_pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            max_connections=50,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        # Create client and ping only if available
        rc = redis.Redis(connection_pool=redis_pool)
        if hasattr(rc, "ping"):
            try:
                rc.ping()
                redis_client = rc
                REDIS_AVAILABLE = True
                print(">>> Redis connected successfully with connection pooling")
            except Exception:
                REDIS_AVAILABLE = False
        else:
            REDIS_AVAILABLE = False
    except Exception as e:
        REDIS_AVAILABLE = False
        print(f">>> Redis not available: {e}. Attempting to use fakeredis...")
        
        # Fallback to fakeredis
        if FAKEREDIS_AVAILABLE:
            try:
                rc = fakeredis.FakeStrictRedis(decode_responses=True)
                if hasattr(rc, "ping"):
                    try:
                        rc.ping()
                        redis_client = rc
                        REDIS_AVAILABLE = True
                        print(">>> Fakeredis initialized successfully (in-memory Redis for Windows)")
                    except Exception as e2:
                        print(f">>> Fakeredis ping failed: {e2}. Running without Redis caching.")
                else:
                    print(">>> Fakeredis available but no ping method; proceeding without Redis caching")
            except Exception as e2:
                print(f">>> Fakeredis also failed: {e2}. Running without Redis caching.")
        else:
            print(">>> Fakeredis module not installed. Running without Redis caching.")
else:
    # Try fakeredis if redis module not available
    if FAKEREDIS_AVAILABLE:
        try:
            rc = fakeredis.FakeStrictRedis(decode_responses=True)
            if hasattr(rc, "ping"):
                try:
                    rc.ping()
                    redis_client = rc
                    REDIS_AVAILABLE = True
                    print(">>> Fakeredis initialized successfully (in-memory Redis for Windows)")
                except Exception as e:
                    print(f">>> Fakeredis ping failed: {e}")
            else:
                print(">>> Fakeredis available but no ping method; proceeding without Redis caching")
        except Exception as e:
            print(f">>> Fakeredis initialization failed: {e}")
    else:
        print(">>> Redis module not installed. Install with: pip install redis")

# ==================================================
# 🔄 FALLBACK COUNTERS FOR TRANSACTION TRACKING
# ==================================================
# Global counters as fallback when Redis is unavailable
transaction_counter = 0
high_risk_counter = 0

# ==================================================
# 🔄 REAL-TIME UPGRADE: WEBSOCKET CONNECTION MANAGER
# ==================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.transaction_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f">>> Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f">>> Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f">>> Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f">>> Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# ==================================================
# LOAD MODEL & METADATA
# ==================================================
# Get the project root directory (parent of api directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.dirname(os.path.abspath(__file__))

# Try multiple possible locations for model files (include common variants)
possible_model_paths = [
    os.path.join(PROJECT_ROOT, "fraud_model.joblib"),  # Preferred name
    os.path.join(PROJECT_ROOT, "rf_fraud_model.joblib"),  # common fallback
    os.path.join(PROJECT_ROOT, "xgb_fraud_model.joblib"),
    os.path.join(PROJECT_ROOT, "random_forest_model.pkl"),
    os.path.join(API_DIR, "auth", "fraud_model.joblib"),  # api/auth/
    os.path.join(API_DIR, "fraud_model.joblib"),  # api/
    os.path.join(API_DIR, "rf_fraud_model.joblib"),
    "fraud_model.joblib",  # Current working directory
    "rf_fraud_model.joblib",
    "random_forest_model.pkl",
]

possible_columns_paths = [
    os.path.join(PROJECT_ROOT, "model_columns.joblib"),  # Project root
    os.path.join(API_DIR, "auth", "model_columns.joblib"),  # api/auth/
    os.path.join(API_DIR, "model_columns.joblib"),  # api/
    "model_columns.joblib"  # Current working directory
]

# Find the model file
MODEL_PATH = None
for path in possible_model_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# Find the model columns file (optional)
MODEL_COLUMNS_PATH = None
for path in possible_columns_paths:
    if os.path.exists(path):
        MODEL_COLUMNS_PATH = path
        break

if not MODEL_PATH:
    raise FileNotFoundError(
        f"Model file not found. Checked these locations:\n" +
        "\n".join(f"  - {p}" for p in possible_model_paths)
    )

if MODEL_COLUMNS_PATH is None:
    print(f">>> Warning: model_columns.joblib not found in any of the expected locations. Attempting to infer model columns after loading the model.")
else:
    print(f">>> Loading model columns from: {MODEL_COLUMNS_PATH}")

print(f">>> Loading model from: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# If model columns file exists load it, otherwise try to infer
if MODEL_COLUMNS_PATH:
    try:
        loaded_cols = joblib.load(MODEL_COLUMNS_PATH)
        # Convert to pure Python list - never leave it as numpy array or pandas Index
        model_columns = [str(col) for col in loaded_cols]
        print(f">>> Loaded model columns (count: {len(model_columns)})")
    except Exception as e:
        print(f">>> Failed to load model_columns.joblib: {e}. Will attempt to infer columns from model.")
        MODEL_COLUMNS_PATH = None

if MODEL_COLUMNS_PATH is None:
    # Try to infer feature names from sklearn models (feature_names_in_) or pipeline
    inferred = None
    if hasattr(model, "feature_names_in_"):
        try:
            inferred = list(model.feature_names_in_)
            print(">>> Inferred model columns from model.feature_names_in_")
        except Exception:
            inferred = None
    elif hasattr(model, "named_steps"):
        # Try to inspect pipeline steps
        try:
            # Attempt to find last estimator with feature_names_in_
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    inferred = list(step.feature_names_in_)
                    print(">>> Inferred model columns from pipeline step feature_names_in_")
                    break
        except Exception:
            inferred = None

    if inferred is not None:
        model_columns = inferred
    else:
        # Last resort: leave model_columns empty and skip reindexing later
        model_columns = []
        print(">>> Could not infer model columns; continuing without a fixed column list (reindexing will be skipped)")
else:
    # Ensure model_columns is always a pure Python list, never a numpy array or pandas Index
    model_columns = [str(col) for col in model_columns]

print(f">>> Final model_columns type: {type(model_columns)}, count: {len(model_columns)}")
print(">>> Model and columns loaded (or inferred) successfully")

LOG_FILE = "logs/transactions_log.csv"

def write_log(input_data, prediction):
    """Append transaction + prediction to CSV log"""
    row = {
        "datetime": datetime.now().isoformat(),
        **input_data,
        **prediction
    }

    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ==================================================
# 🔐 UPGRADE 6 — SECURITY & RATE LIMIT CONFIG
# ==================================================
API_KEY = "paywatch-secure-key"
RATE_LIMIT = 1000  # requests per minute per IP (increased for simulation)
request_tracker = defaultdict(list)

# ==================================================
# 🔐 UPGRADE 6 — AUTH + RATE LIMIT MIDDLEWARE
# ==================================================


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Public endpoints that do not require auth
    free_paths = [
        "/",             # Root path
        "/health",       # Health check - must be accessible
        "/auth/login",
        "/auth/signup",
        "/predict",      # FIXED: Allow predictions without auth
        "/stream",       # SSE stream for simulation
        "/stats",        # Statistics endpoint - FIXED: Added
        "/ws",           # WebSocket base
        "/ws/stream",    # WebSocket stream
        "/docs",         # API documentation
        "/openapi.json", # OpenAPI schema
        "/favicon.ico",  # Favicon
        "/redoc"         # ReDoc documentation
    ]

    path = request.url.path
    
    # Skip auth for free paths
    # Check if path exactly matches or starts with a free path
    is_free_path = False
    for fp in free_paths:
        if fp == "/" and path == "/":
            is_free_path = True
            break
        elif fp != "/" and (path == fp or path.startswith(fp + "/")):
            is_free_path = True
            break
    
    if is_free_path:
        return await call_next(request)

    # Rate limiting per IP (applies to auth-protected requests only)
    ip = request.client.host if request.client else "unknown"
    now = datetime.utcnow().timestamp()
    
    # Remove old requests (keep only last 60 seconds)
    if ip not in request_tracker:
        request_tracker[ip] = []
    request_tracker[ip] = [t for t in request_tracker[ip] if now - t < 60]
    
    # If more than N hits in last 60 seconds = block
    if len(request_tracker[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down!")
    
    # Add timestamp
    request_tracker[ip].append(now)

    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization Token")

    token = token.replace("Bearer ", "")
    user_data = verify_token(token)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    request.state.user = user_data
    return await call_next(request)# ==================================================
# 🧾 UPGRADE 6 — AUDIT LOGGING (ASYNC)
# ==================================================
async def log_transaction(tx, response):
    log = {
        "timestamp": datetime.now().isoformat(),
        "type": tx.get("type"),
        "amount": tx.get("amount"),
        "risk_level": response.get("risk_level"),
        "fraud_probability": response.get("fraud_probability")
    }

    df = pd.DataFrame([log])
    df.to_csv("audit_log.csv", mode="a", header=False, index=False)
    
    # 🔄 REAL-TIME UPGRADE: Store in Redis for real-time analytics + Global fallback counters
    global transaction_counter, high_risk_counter
    
    # Always increment global counters (works with or without Redis)
    transaction_counter += 1
    print(f">>> Transaction logged. Counter now: {transaction_counter}")
    
    if response.get("risk_level") == "HIGH":
        high_risk_counter += 1
        print(f">>> HIGH RISK detected. High risk counter now: {high_risk_counter}")
    
    if REDIS_AVAILABLE and redis_client is not None:
        try:
            rc = redis_client
            if hasattr(rc, "lpush"):
                rc.lpush("fraud_alerts", json.dumps(log))
            if hasattr(rc, "ltrim"):
                rc.ltrim("fraud_alerts", 0, 999)  # Keep last 1000 alerts
            if hasattr(rc, "incr"):
                rc.incr("total_transactions")
                if response.get("risk_level") == "HIGH":
                    rc.incr("high_risk_count")
        except Exception as e:
            print(f">>> Redis logging error: {e}")

# ==================================================
# 🔄 REAL-TIME UPGRADE: CACHE MANAGEMENT
# ==================================================
def get_cache_key(transaction: dict) -> str:
    """Generate cache key from transaction"""
    key_parts = [
        str(transaction.get("type", "")),
        str(round(transaction.get("amount", 0), 2)),
        str(round(transaction.get("oldbalanceOrg", 0), 2)),
        str(round(transaction.get("oldbalanceDest", 0), 2))
    ]
    return f"fraud_pred:{':'.join(key_parts)}"

async def get_cached_prediction(cache_key: str):
    """Get prediction from cache if available"""
    if not REDIS_AVAILABLE:
        return None
    try:
        if redis_client is None:
            return None
        rc = redis_client
        if not hasattr(rc, "get"):
            return None
        cached = rc.get(cache_key)
        if not cached:
            return None
        # Ensure cached is a str/bytes before json.loads to satisfy static typing
        if isinstance(cached, (str, bytes, bytearray)):
            return json.loads(cached)
        # Some clients return objects with a .text attribute
        if hasattr(cached, "text"):
            return json.loads(cached.text)
        # Fallback: attempt to return as-is
        return cached
    except Exception as e:
        print(f">>> Cache read error: {e}")
    return None

async def set_cached_prediction(cache_key: str, prediction: dict, ttl: int = 300):
    """Cache prediction for 5 minutes"""
    if not REDIS_AVAILABLE:
        return
    try:
        if redis_client is None:
            return
        if hasattr(redis_client, "setex"):
            redis_client.setex(cache_key, ttl, json.dumps(prediction))
    except Exception as e:
        print(f">>> Cache write error: {e}")

# ==================================================
# ❤️ UPGRADE 6 — HEALTH CHECK ENDPOINT
# ==================================================
@app.get("/health")
async def health_check():
    """Health check endpoint - must be simple and never fail"""
    try:
        return {
            "status": "UP",
            "service": "PayWatch AI Fraud API",
            "model_loaded": model is not None if 'model' in globals() else False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # Even if something fails, return a basic health response
        return {
            "status": "UP",
            "service": "PayWatch AI Fraud API",
            "model_loaded": False,
            "timestamp": datetime.now().isoformat(),
            "warning": str(e)
        }

# ==================================================
# 🔮 FRAUD PREDICTION ENDPOINT (ASYNC + CACHING)
# ==================================================
@app.post("/predict")
async def predict_fraud(transaction: dict):
    import traceback
    try:
        print("\n" + "="*60)
        print(">>> /predict called")
        print("="*60)
        
        # Validate input
        if not transaction:
            raise ValueError("Transaction data is empty")
        
        print(">>> RAW INPUT:", json.dumps(transaction, default=str))
        
        # Create a clean copy to avoid modifying the original
        tx_clean = dict(transaction)
        
        # Ensure required fields exist with proper defaults
        required_fields = {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 0.0,
            'oldbalanceOrg': 0.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        }
        
        for field, default_val in required_fields.items():
            if field not in tx_clean:
                tx_clean[field] = default_val
            # Convert to proper types
            if field == 'type':
                tx_clean[field] = str(tx_clean[field]).upper()
            elif field == 'step':
                try:
                    tx_clean[field] = int(tx_clean[field])
                except:
                    tx_clean[field] = 1
            else:
                try:
                    tx_clean[field] = float(tx_clean[field])
                except:
                    tx_clean[field] = default_val
        
        # Remove non-model fields
        tx_clean.pop('timestamp', None)
        tx_clean.pop('isFraud', None)
        
        print(">>> CLEANED INPUT:", tx_clean)
        
        # Create DataFrame
        df = pd.DataFrame([tx_clean])
        print(">>> DataFrame created - shape:", df.shape)
        print(">>> Columns:", df.columns.tolist())
        
        # Apply feature engineering
        try:
            print(">>> Applying feature_engineering...")
            df = feature_engineering(df)
            print(">>> Feature engineering done - shape:", df.shape)
            print(">>> Columns after FE:", df.columns.tolist())
        except Exception as fe:
            print(f">>> FEATURE ENGINEERING ERROR: {str(fe)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(fe)}")

        # Align with model columns
        try:
            if model_columns and len(model_columns) > 0:
                print(f">>> Reindexing to {len(model_columns)} model columns...")
                df = df.reindex(columns=model_columns, fill_value=0)
                print(">>> Reindex done - shape:", df.shape)
            else:
                print(">>> No model columns; skipping reindex")
        except Exception as ce:
            print(f">>> REINDEX ERROR: {str(ce)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Reindex failed: {str(ce)}")

        # Predict
        try:
            print(">>> Running prediction...")
            print(">>> DataFrame shape for model:", df.shape)
            print(">>> DataFrame dtypes:", df.dtypes.to_dict())
            
            proba = model.predict_proba(df)[0][1]
            print(f">>> Prediction successful: {proba}")
            
            risk = "HIGH" if proba > 0.8 else "MEDIUM" if proba > 0.4 else "LOW"

            # Create response object
            response = {
                "fraud_probability": float(proba),
                "risk_level": risk
            }
            
            # Log transaction BEFORE returning
            print(">>> Logging transaction...")
            await log_transaction(tx_clean, response)
            print(">>> Transaction logged successfully")

            print(">>> Returning result")
            return response
        except Exception as pe:
            print(f">>> PREDICTION ERROR: {str(pe)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(pe)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f">>> UNEXPECTED ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ==================================================
# 📡 REAL-TIME STREAM ENDPOINT (SSE - Server-Sent Events)
# ==================================================
@app.get("/stream")
async def stream_transaction():
    """Server-Sent Events endpoint for real-time transaction streaming"""
    async def generate():
        try:
            while True:
                try:
                    tx = {
                        "step": random.randint(1, 24),
                        "type": random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]),
                        "amount": round(random.uniform(10, 10000), 2),
                        "oldbalanceOrg": round(random.uniform(0, 20000), 2),
                        "newbalanceOrig": round(random.uniform(0, 20000), 2),
                        "oldbalanceDest": round(random.uniform(0, 20000), 2),
                        "newbalanceDest": round(random.uniform(0, 20000), 2),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(tx)}\n\n"
                    await asyncio.sleep(2)  # Generate transaction every 2 seconds
                except asyncio.CancelledError:
                    # Client disconnected, exit gracefully
                    print(">>> SSE stream cancelled (client disconnected)")
                    break
                except Exception as e:
                    # Log error but continue streaming
                    print(f">>> SSE stream error: {e}")
                    # Send error message to client
                    try:
                        error_msg = {"error": str(e), "timestamp": datetime.now().isoformat()}
                        yield f"data: {json.dumps(error_msg)}\n\n"
                    except:
                        pass
                    # Wait a bit before retrying
                    await asyncio.sleep(2)
        except Exception as e:
            # Final error handling - log and exit
            print(f">>> SSE stream fatal error: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            print(">>> SSE stream ended")
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx
        }
    )

# ==================================================
# 🔄 REAL-TIME UPGRADE: WEBSOCKET ENDPOINT
# ==================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for bidirectional real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal_message({"type": "pong"}, websocket)
                elif message.get("type") == "subscribe":
                    # Client wants to receive real-time updates
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "message": "You will receive real-time fraud alerts"
                    }, websocket)
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================================================
# 🔄 REAL-TIME UPGRADE: WEBSOCKET STREAM ENDPOINT
# ==================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint that streams transactions and predictions in real-time"""
    await manager.connect(websocket)
    try:
        while True:
            # Generate transaction
            tx = {
                "step": random.randint(1, 24),
                "type": random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]),
                "amount": round(random.uniform(10, 10000), 2),
                "oldbalanceOrg": round(random.uniform(0, 20000), 2),
                "newbalanceOrig": round(random.uniform(0, 20000), 2),
                "oldbalanceDest": round(random.uniform(0, 20000), 2),
                "newbalanceDest": round(random.uniform(0, 20000), 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # Get prediction
            try:
                df = pd.DataFrame([tx])
                df = feature_engineering(df)
                df = df.reindex(columns=model_columns, fill_value=0)
                proba = model.predict_proba(df)[0][1]
                risk = "HIGH" if proba > 0.8 else "MEDIUM" if proba > 0.4 else "LOW"
                
                prediction = {
                    "fraud_probability": round(float(proba), 4),
                    "risk_level": risk,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send transaction and prediction
                await manager.send_personal_message({
                    "type": "transaction",
                    "transaction": tx,
                    "prediction": prediction
                }, websocket)
            except Exception as e:
                await manager.send_personal_message({
                    "type": "error",
                    "message": str(e)
                }, websocket)
            
            await asyncio.sleep(2)  # Stream every 2 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================================================
# 🔄 REAL-TIME UPGRADE: STATISTICS ENDPOINT
# ==================================================
@app.get("/stats")
async def get_stats():
    """Always-safe statistics endpoint (even without Redis)."""
    global transaction_counter, high_risk_counter
    
    print(f">>> /stats called. Current counters: transaction_counter={transaction_counter}, high_risk_counter={high_risk_counter}")
    
    try:
        # Always return the global counters (works with or without Redis)
        if REDIS_AVAILABLE and redis_client is not None:
            try:
                # Try to use Redis counts if available
                rc = redis_client
                total_tx_raw = rc.get("total_transactions") if hasattr(rc, "get") else None
                high_risk_raw = rc.get("high_risk_count") if hasattr(rc, "get") else None

                # Normalize raw values to int safely
                try:
                    total_tx = int(total_tx_raw) if total_tx_raw is not None and str(total_tx_raw).isdigit() else transaction_counter
                except Exception:
                    total_tx = transaction_counter

                try:
                    high_risk = int(high_risk_raw) if high_risk_raw is not None and str(high_risk_raw).isdigit() else high_risk_counter
                except Exception:
                    high_risk = high_risk_counter
                
                # Use global counters if Redis values are None/0
                if total_tx == 0:
                    total_tx = transaction_counter
                if high_risk == 0:
                    high_risk = high_risk_counter
                
                print(f">>> /stats returning (Redis): total_transactions={total_tx}, high_risk_count={high_risk}")
                
                alerts = []
                raw_alerts = rc.lrange("fraud_alerts", 0, 9) if hasattr(rc, "lrange") else []
                # Ensure iterable
                if raw_alerts:
                    for a in raw_alerts:
                        try:
                            if isinstance(a, (str, bytes, bytearray)):
                                alerts.append(json.loads(a))
                            elif hasattr(a, "text"):
                                alerts.append(json.loads(a.text))
                            else:
                                alerts.append(a)
                        except Exception:
                            pass
                
                return {
                    "status": "ok",
                    "redis_enabled": True,
                    "total_transactions": total_tx,
                    "high_risk_count": high_risk,
                    "recent_alerts": alerts,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as redis_err:
                # Fall back to global counters if Redis fails
                print(f">>> /stats Redis error, falling back to global counters: {redis_err}")
                return {
                    "status": "ok",
                    "redis_enabled": False,
                    "total_transactions": transaction_counter,
                    "high_risk_count": high_risk_counter,
                    "recent_alerts": [],
                    "timestamp": datetime.now().isoformat(),
                }
        else:
            # No Redis available - use global counters
            print(f">>> /stats returning (No Redis): total_transactions={transaction_counter}, high_risk_count={high_risk_counter}")
            return {
                "status": "ok",
                "redis_enabled": False,
                "total_transactions": transaction_counter,
                "high_risk_count": high_risk_counter,
                "recent_alerts": [],
                "timestamp": datetime.now().isoformat(),
            }
    
    except Exception as e:
        # Never return 500 to Streamlit
        print(f">>> /stats error: {e}")
        return {
            "status": "error",
            "redis_enabled": False,
            "message": str(e),
            "total_transactions": transaction_counter,
            "high_risk_count": high_risk_counter,
            "recent_alerts": [],
            "timestamp": datetime.now().isoformat(),
        }

app.include_router(auth_router, prefix="/auth")
