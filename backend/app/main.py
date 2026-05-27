import os, json, time
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Go TWO levels up: backend/app/main.py -> backend/app -> backend -> AlgorithmCode (root)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

ROOT_ARTIFACTS = os.path.join(PROJECT_ROOT, "artifacts")
ROOT_DATA = os.path.join(PROJECT_ROOT, "data")

INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(ROOT_ARTIFACTS, "faiss.index"))
IDS_PATH = os.getenv("IDS_PATH", os.path.join(ROOT_ARTIFACTS, "item_ids.json"))
VECTORS_PATH = os.getenv("VECTORS_PATH", os.path.join(ROOT_ARTIFACTS, "item_vectors.npy"))
ITEMS_JSON = os.getenv("ITEMS_JSON", os.path.join(ROOT_DATA, "items.json"))

# Load FAISS + metadata on startup
index = faiss.read_index(INDEX_PATH)
with open(IDS_PATH, "r") as f:
    ITEM_IDS = json.load(f)
ITEM_VECS = np.load(VECTORS_PATH)  # (N, d)
with open(ITEMS_JSON, "r") as f:
    ITEMS_META = {it["item_id"]: it for it in json.load(f)}

EMBED_DIM = ITEM_VECS.shape[1]

# In-memory user vectors & seen set for demo
USER_VEC = {}   # user_id -> np.array (d,)
USER_SEEN = {}  # user_id -> set(item_id)

POPULARITY = {iid: 0.0 for iid in ITEM_IDS}
CREATED_AT = {iid: time.time() for iid in ITEM_IDS}


class FeedRequest(BaseModel):
    user_id: str
    limit: int = 15
    exclude_seen: bool = True


class Interaction(BaseModel):
    user_id: str
    item_id: str
    action: str   # "like" or "skip"


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://keyotic.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images
app.mount("/images", StaticFiles(directory=os.path.join(PROJECT_ROOT, "data", "images")), name="images")


@app.get("/healthz")
def healthz():
    return {"ok": True, "count": len(ITEM_IDS)}


def get_user_vec(uid: str):
    return USER_VEC.get(uid, None)


def set_user_vec(uid: str, v):
    USER_VEC[uid] = v


def mark_seen(uid: str, item_id: str):
    USER_SEEN.setdefault(uid, set()).add(item_id)


def has_seen(uid: str, item_id: str):
    return item_id in USER_SEEN.get(uid, set())


def update_user_vector(uid: str, item_vec: np.ndarray, like: bool, lam: float = 0.8):
    u = USER_VEC.get(uid)
    if u is None:
        u = item_vec.copy() if like else np.random.randn(EMBED_DIM).astype("float32")
        u /= np.linalg.norm(u) + 1e-8
    else:
        if like:
            u = lam * u + (1 - lam) * item_vec
        else:
            u = lam * u - (1 - lam) * 0.1 * item_vec
        u /= np.linalg.norm(u) + 1e-8
    USER_VEC[uid] = u
    return u


def score_items(user_vec: np.ndarray, idxes: np.ndarray, sims: np.ndarray, topk: int):
    now = time.time()
    alpha, beta, gamma = 0.85, 0.10, 0.05
    scored = []
    for row_idx, sim in zip(idxes, sims):
        item_id = ITEM_IDS[row_idx]
        pop = POPULARITY.get(item_id, 0.0)
        recency = np.exp(-max(0, now - CREATED_AT.get(item_id, now)) / (7*24*3600))
        s = alpha*float(sim) + beta*pop + gamma*recency
        scored.append((item_id, s, row_idx))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def mmr_rerank(user_vec: np.ndarray, candidates: list, topn: int, lam: float = 0.7):
    selected = []
    selected_vecs = []
    cand_vecs = ITEM_VECS[[row_idx for (_, _, row_idx) in candidates]]
    cand_items = [(item_id, score, row_idx) for (item_id, score, row_idx) in candidates]

    if not cand_items:
        return []
    first = cand_items[0]
    selected.append(first)
    selected_vecs.append(ITEM_VECS[first[2]])

    while len(selected) < min(topn, len(cand_items)):
        best_idx = None
        best_val = -1e9
        for (item_id, score, row_idx) in cand_items:
            if any(item_id == s[0] for s in selected):
                continue
            d_vec = ITEM_VECS[row_idx]
            sim_to_user = float(np.dot(user_vec, d_vec))
            sim_to_selected = 0.0
            if selected_vecs:
                sims = np.dot(np.vstack(selected_vecs), d_vec)
                sim_to_selected = float(np.max(sims))
            val = lam * sim_to_user - (1 - lam) * sim_to_selected
            if val > best_val:
                best_val = val
                best_idx = (item_id, score, row_idx)
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_vecs.append(ITEM_VECS[best_idx[2]])
    return selected


def convert_path_to_url(item):
    """Convert local file path to HTTP URL for frontend"""
    result = item.copy()
    if result['path'].startswith('data/'):
        filename = result['path'].split('/')[-1]
        # Use full backend URL so it works from any frontend domain
        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        result['path'] = f"{base_url}/images/{filename}"
    return result



@app.post("/feed")
def feed(req: FeedRequest):
    uid = req.user_id
    u = get_user_vec(uid)
    k_candidates = max(req.limit*5, 100)

    if u is None:
        items = []
        for item_id in ITEM_IDS:
            if not req.exclude_seen or not has_seen(uid, item_id):
                items.append(item_id)
            if len(items) >= req.limit:
                break
        result = [convert_path_to_url(ITEMS_META[i]) for i in items]
        return {"items": result}

    q = u.reshape(1, -1).astype("float32")
    
    # Try progressively wider searches
    attempts = [
        k_candidates,      # 100 items
        k_candidates * 3,  # 300 items
        k_candidates * 10, # 1000 items
        len(ITEM_IDS)      # All items
    ]
    
    cand = []
    
    for attempt_k in attempts:
        attempt_k = min(attempt_k, len(ITEM_IDS))
        sims, idxes = index.search(q, attempt_k)
        idxes, sims = idxes[0], sims[0]
        
        cand = [(ITEM_IDS[i], sims[j], i) for j, i in enumerate(idxes)
                if not req.exclude_seen or not has_seen(uid, ITEM_IDS[i])]
        
        if len(cand) >= req.limit:
            break
    
    # Last resort: reset seen items
    if len(cand) < req.limit:
        USER_SEEN[uid] = set()
        sims, idxes = index.search(q, k_candidates)
        idxes, sims = idxes[0], sims[0]
        cand = [(ITEM_IDS[i], sims[j], i) for j, i in enumerate(idxes)]

    scored = score_items(u, np.array([c[2] for c in cand]), np.array([c[1] for c in cand]), len(cand))
    reranked = mmr_rerank(u, scored, req.limit, lam=0.7)
    final_items = [convert_path_to_url(ITEMS_META[iid]) for (iid, _, _) in reranked]
    return {"items": final_items}


@app.post("/interactions")
def interactions(evt: Interaction):
    if evt.action not in ("like", "skip"):
        raise HTTPException(status_code=400, detail="Invalid action")
    try:
        row_idx = ITEM_IDS.index(evt.item_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown item_id")

    item_vec = ITEM_VECS[row_idx]
    like = (evt.action == "like")
    update_user_vector(evt.user_id, item_vec, like)
    mark_seen(evt.user_id, evt.item_id)
    POPULARITY[evt.item_id] = POPULARITY.get(evt.item_id, 0.0) + (1.0 if like else 0.0)
    return {"ok": True}
