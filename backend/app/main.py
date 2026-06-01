import os, json, time, random
import traceback
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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


# Log orphaned IDs at startup — do NOT filter ITEM_IDS/ITEM_VECS
# as FAISS row indices must stay perfectly aligned with ITEM_IDS
orphans = [iid for iid in ITEM_IDS if iid not in ITEMS_META]
if orphans:
    print(f"[startup] WARNING: {len(orphans)} orphaned IDs not in items.json: {orphans}")


EMBED_DIM = ITEM_VECS.shape[1]


# In-memory user vectors & seen set for demo
USER_VEC = {}   # user_id -> np.array (d,)
USER_SEEN = {}  # user_id -> set(item_id)

# Admin/debug tracking
USER_HISTORY = {}   # user_id -> list of {"item_id": str, "action": str, "ts": float}
LAST_ACTIVE_USER = None
LAST_ACTIVITY_AT = 0.0


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


class ResetRequest(BaseModel):
    user_id: str


app = FastAPI()


# Exception middleware — MUST be added BEFORE CORSMiddleware
# Ensures 500 errors still return CORS headers so the browser
# can read the actual error message instead of a generic "Network Error"
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"Access-Control-Allow-Origin": "*"},
        )


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


def mark_active(uid: str):
    global LAST_ACTIVE_USER, LAST_ACTIVITY_AT
    LAST_ACTIVE_USER = uid
    LAST_ACTIVITY_AT = time.time()


def log_interaction(uid: str, item_id: str, action: str):
    USER_HISTORY.setdefault(uid, []).append({
        "item_id": item_id,
        "action": action,
        "ts": time.time(),
    })


def generate_personalized_fortune(uid: str):
    history = USER_HISTORY.get(uid, [])
    likes = [h for h in history if h["action"] == "like"]
    skips = [h for h in history if h["action"] == "skip"]

    like_count = len(likes)
    skip_count = len(skips)
    total = like_count + skip_count

    if total == 1:
        pool = [
            "Your future is still unwritten, but possibility already surrounds you.",
            "The path ahead is quiet now, yet something meaningful is beginning.",
            "A hidden opportunity is waiting for you to make the first move.",
            "The first sign is subtle, but it will lead you somewhere worth finding.",
            "What feels distant now will soon step closer than expected.",
            "A quiet beginning is preparing a louder destiny.",
            "Something small is already shifting in your favor.",
            "The answer has not appeared yet, but the pattern has already begun.",
            "A new path opens the moment you decide to notice it."
        ]
        return random.choice(pool)

    like_ratio = like_count / total
    skip_ratio = skip_count / total

    if like_count >= 12 and like_ratio >= 0.7:
        pool = [
            "You move toward life with confidence. A bold opportunity will soon reward your instincts.",
            "You know what draws you in, and that certainty will open an unexpected door.",
            "Your future favors decisive energy. What you choose next may change more than you expect.",
            "You follow desire without apology, and fortune responds to that courage.",
            "The things you choose boldly now will echo back as luck later.",
            "A risk you are ready for will soon reveal itself.",
            "Your confidence is becoming a magnet for rare opportunities.",
            "You are not waiting for the future. You are pulling it toward you.",
            "A moment of fearless choice will set something powerful into motion."
        ]
    elif skip_count >= 12 and skip_ratio >= 0.7:
        pool = [
            "You are guided by discernment, not distraction. A clearer path is forming ahead of you.",
            "You know how to reject what is not meant for you. That wisdom will protect your future.",
            "Your restraint is a strength. By turning away from the wrong things, you are making space for the right one.",
            "Your future sharpens each time you refuse what does not fit.",
            "Clarity is becoming your greatest advantage.",
            "You are cutting through illusion, and that will soon reveal something real.",
            "Your ability to say no is quietly shaping a more honest destiny.",
            "You are not missing out. You are refining what truly belongs to you.",
            "By filtering the noise, you are making room for something rare."
        ]
    elif like_count > skip_count:
        pool = [
            "You follow curiosity with an open heart. Soon, something new will feel instantly familiar.",
            "You are drawn to possibility, and that openness will bring a fortunate surprise.",
            "Your future carries momentum. What excites you now is pointing toward what comes next.",
            "You are expanding faster than you realize, and the world is beginning to answer.",
            "An unexpected invitation will match the energy you have been moving toward.",
            "Your openness will lead you somewhere richer than your original plan.",
            "What delights you now is more than a preference. It is a direction.",
            "You are gathering signals from the future through what attracts you today.",
            "A joyful instinct will soon prove wiser than logic alone."
        ]
    elif skip_count > like_count:
        pool = [
            "You trust your inner filter, and it is sharpening your destiny.",
            "You are narrowing the noise around you. What remains will matter deeply.",
            "Your future grows clearer with every choice you refuse.",
            "You are learning through elimination, and that knowledge is powerful.",
            "What you reject now is protecting the shape of what comes next.",
            "The path ahead is becoming visible because you are no longer chasing everything.",
            "There is power in your refusal, and it will soon reveal purpose.",
            "By stepping away from the wrong things, you are drawing closer to the right one.",
            "Your patience with imperfection will soon be rewarded by something unmistakable."
        ]
    else:
        pool = [
            "You balance instinct and caution with rare precision. A meaningful choice is approaching.",
            "You move carefully, but not fearfully. That balance will serve you well.",
            "You are learning not only what you want, but why. That knowledge will shape your next chapter.",
            "You stand between desire and discernment, and that is where wisdom grows.",
            "A balanced heart often sees what others miss.",
            "You are not divided. You are measuring the world with care.",
            "Because you weigh both impulse and restraint, your next step will carry unusual strength.",
            "Your future is being built on thoughtful tension, and that makes it resilient.",
            "The harmony between your curiosity and caution will soon reveal a powerful answer."
        ]

    return random.choice(pool)


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
        # Guard: skip FAISS rows that fall outside ITEM_IDS bounds
        if row_idx < 0 or row_idx >= len(ITEM_IDS):
            continue
        item_id = ITEM_IDS[row_idx]
        if item_id not in ITEMS_META:
            continue
        pop = POPULARITY.get(item_id, 0.0)
        recency = np.exp(-max(0, now - CREATED_AT.get(item_id, now)) / (7 * 24 * 3600))
        s = alpha * float(sim) + beta * pop + gamma * recency
        scored.append((item_id, s, row_idx))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def mmr_rerank(user_vec: np.ndarray, candidates: list, topn: int, lam: float = 0.7):
    selected = []
    selected_vecs = []
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
    if result["path"].startswith("data/"):
        filename = result["path"].split("/")[-1]
        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        result["path"] = f"{base_url}/images/{filename}"
    return result


@app.post("/feed")
def feed(req: FeedRequest):
    try:
        mark_active(req.user_id)

        uid = req.user_id
        u = get_user_vec(uid)
        k_candidates = max(req.limit * 5, 100)

        if u is None:
            # Larger initial buffer so early skips don't exhaust the seen set
            items = []
            for item_id in ITEM_IDS:
                if item_id not in ITEMS_META:
                    continue
                if not req.exclude_seen or not has_seen(uid, item_id):
                    items.append(item_id)
                if len(items) >= req.limit * 3:
                    break
            result = [convert_path_to_url(ITEMS_META[i]) for i in items[:req.limit]]
            return {"items": result}

        q = u.reshape(1, -1).astype("float32")

        # Try progressively wider searches
        attempts = [
            k_candidates,       # 100 items
            k_candidates * 3,   # 300 items
            k_candidates * 10,  # 1000 items
            len(ITEM_IDS)       # All items
        ]

        cand = []

        for attempt_k in attempts:
            attempt_k = min(attempt_k, len(ITEM_IDS))
            sims, idxes = index.search(q, attempt_k)
            idxes, sims = idxes[0], sims[0]

            cand = [
                (ITEM_IDS[i], sims[j], i)
                for j, i in enumerate(idxes)
                if 0 <= i < len(ITEM_IDS)
                and ITEM_IDS[i] in ITEMS_META
                and (not req.exclude_seen or not has_seen(uid, ITEM_IDS[i]))
            ]

            if len(cand) >= req.limit:
                break

        # Smart reset: preserve liked items, clear skipped items from seen set
        if len(cand) < req.limit:
            liked_seen = {
                iid for iid in USER_SEEN.get(uid, set())
                if POPULARITY.get(iid, 0.0) > 0
            }
            USER_SEEN[uid] = liked_seen

            sims, idxes = index.search(q, k_candidates)
            idxes, sims = idxes[0], sims[0]
            cand = [
                (ITEM_IDS[i], sims[j], i)
                for j, i in enumerate(idxes)
                if 0 <= i < len(ITEM_IDS)
                and ITEM_IDS[i] in ITEMS_META
                and ITEM_IDS[i] not in liked_seen
            ]

        # Absolute fallback: scan full catalog to guarantee a full feed
        if len(cand) < req.limit:
            already = {c[0] for c in cand}
            for i, item_id in enumerate(ITEM_IDS):
                if (
                    item_id in ITEMS_META
                    and item_id not in already
                    and item_id not in USER_SEEN.get(uid, set())
                ):
                    sim = float(np.dot(u, ITEM_VECS[i]))
                    cand.append((item_id, sim, i))
                if len(cand) >= req.limit * 2:
                    break

        if not cand:
            return {"items": []}

        scored = score_items(
            u,
            np.array([c[2] for c in cand]),
            np.array([c[1] for c in cand]),
            len(cand)
        )
        reranked = mmr_rerank(u, scored, req.limit, lam=0.7)
        final_items = [
            convert_path_to_url(ITEMS_META[iid])
            for (iid, _, _) in reranked
            if iid in ITEMS_META
        ]
        return {"items": final_items}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interactions")
def interactions(evt: Interaction):
    mark_active(evt.user_id)

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
    log_interaction(evt.user_id, evt.item_id, evt.action)
    POPULARITY[evt.item_id] = POPULARITY.get(evt.item_id, 0.0) + (1.0 if like else 0.0)
    return {"ok": True}


@app.get("/fortune/{user_id}")
def get_fortune(user_id: str):
    history = USER_HISTORY.get(user_id, [])
    return {
        "user_id": user_id,
        "history_count": len(history),
        "fortune": generate_personalized_fortune(user_id),
    }


# Resets a user's vector, seen set, history, and active state back to a clean state
@app.post("/reset")
def reset(req: ResetRequest):
    global LAST_ACTIVE_USER, LAST_ACTIVITY_AT

    USER_VEC.pop(req.user_id, None)
    USER_SEEN.pop(req.user_id, None)
    USER_HISTORY.pop(req.user_id, None)

    if LAST_ACTIVE_USER == req.user_id:
        LAST_ACTIVE_USER = None
        LAST_ACTIVITY_AT = 0.0

    return {"ok": True}


# ── Admin / Debug endpoints ────────────────────────────────────────────────

@app.get("/admin/users")
def admin_users():
    return {
        "users": [
            {
                "user_id": uid,
                "seen_count": len(USER_SEEN.get(uid, set())),
                "has_vector": uid in USER_VEC,
                "history_count": len(USER_HISTORY.get(uid, [])),
                "is_active": uid == LAST_ACTIVE_USER,
            }
            for uid in sorted(set(
                list(USER_VEC.keys()) +
                list(USER_SEEN.keys()) +
                list(USER_HISTORY.keys())
            ))
        ]
    }


@app.get("/admin/user/{user_id}")
def admin_user_detail(user_id: str):
    history = USER_HISTORY.get(user_id, [])
    liked = [h["item_id"] for h in history if h["action"] == "like"]
    skipped = [h["item_id"] for h in history if h["action"] == "skip"]

    return {
        "user_id": user_id,
        "has_vector": user_id in USER_VEC,
        "total_seen": len(USER_SEEN.get(user_id, set())),
        "liked": liked,
        "skipped": skipped,
        "liked_count": len(liked),
        "skipped_count": len(skipped),
        "history": history,
        "is_active": user_id == LAST_ACTIVE_USER,
        "last_activity_at": LAST_ACTIVITY_AT if user_id == LAST_ACTIVE_USER else None,
    }


@app.get("/admin/current-user")
def admin_current_user():
    if not LAST_ACTIVE_USER:
        return {
            "active_user": None,
            "has_vector": False,
            "total_seen": 0,
            "liked": [],
            "skipped": [],
            "liked_count": 0,
            "skipped_count": 0,
            "history": [],
            "last_activity_at": None,
        }

    uid = LAST_ACTIVE_USER
    history = USER_HISTORY.get(uid, [])
    liked = [h["item_id"] for h in history if h["action"] == "like"]
    skipped = [h["item_id"] for h in history if h["action"] == "skip"]

    return {
        "active_user": uid,
        "has_vector": uid in USER_VEC,
        "total_seen": len(USER_SEEN.get(uid, set())),
        "liked": liked,
        "skipped": skipped,
        "liked_count": len(liked),
        "skipped_count": len(skipped),
        "history": history,
        "last_activity_at": LAST_ACTIVITY_AT,
    }


@app.get("/admin/popularity")
def admin_popularity():
    sorted_items = sorted(POPULARITY.items(), key=lambda x: x[1], reverse=True)
    return {
        "items": [
            {"item_id": iid, "likes": int(count)}
            for iid, count in sorted_items
            if count > 0
        ]
    }


@app.post("/admin/preview-feed/{user_id}")
def admin_preview_feed(user_id: str, limit: int = 15):
    return feed(FeedRequest(user_id=user_id, limit=limit, exclude_seen=True))