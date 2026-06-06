import os, json, time
import traceback
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
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
USER_HISTORY = {}       # user_id -> list of {"item_id": str, "action": str, "ts": float}
USER_CURRENT_FEED = {}  # user_id -> list of item dicts (the live batch from last /feed call)
USER_FEED_INDEX = {}    # user_id -> int (index of the item currently on screen)
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




class Advance(BaseModel):
    user_id: str
    item_id: str  # the item NOW on screen




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
    allow_origins=["*"],
    allow_credentials=False,
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




def get_fortune_state(uid: str):
    history = USER_HISTORY.get(uid, [])
    likes = [h for h in history if h["action"] == "like"]
    skips = [h for h in history if h["action"] == "skip"]


    like_count = len(likes)
    skip_count = len(skips)
    total = like_count + skip_count


    if total <= 1:
        return {
            "segment": "early-stage",
            "like_count": like_count,
            "skip_count": skip_count,
            "total": total,
            "like_ratio": 0.0 if total == 0 else like_count / total,
            "skip_ratio": 0.0 if total == 0 else skip_count / total,
            "pool": [
                "\"Your future is still unwritten, but possibility already surrounds you.\"",
                "\"The path ahead is silent now… but something meaningful has begun.\"",
                "\"A hidden opportunity waits for you to make the first move.\"",
                "\"The first sign is subtle… yet it will lead you somewhere rare.\"",
                "\"What feels distant now… will soon draw unexpectedly close.\"",
                "\"A quiet beginning is shaping a far greater destiny.\""
                "\"Something small is already shifting in your favor.\"",
                "\"The answer has not appeared… but the pattern has begun.\"",
                "\"A new path opens the moment you decide to notice it.\""
            ]
        }


    like_ratio = like_count / total
    skip_ratio = skip_count / total


    if like_count >= 12 and like_ratio >= 0.7:
        return {
            "segment": "bold-liker",
            "like_count": like_count,
            "skip_count": skip_count,
            "total": total,
            "like_ratio": like_ratio,
            "skip_ratio": skip_ratio,
            "pool": [
                "\"You move through life with confidence… and a bold opportunity will soon reward you.\"",
                "\"You know what draws you in, and that certainty will open an unexpected door.\"",
                "\"Your future favors decisive energy… what you choose next will shift more than you expect.\"",
                "\"You follow desire without hesitation… and fortune answers that courage.\"",
                "\"What you choose boldly now… will return to you as luck.\"",
                "\"A risk meant for you is already approaching.\"",
                "\"Your confidence is becoming a magnet for rare opportunities.\"",
                "\"You are not waiting for the future. You are pulling it toward you.\"",
                "\"A single fearless choice will set something powerful in motion.\"",
                "\"You trust your instincts… and they are already aligning things in your favor.\"",
                "\"What you claim without doubt… will begin to shape itself around you.\""
                "\"Opportunity recognizes your certainty… and moves closer because of it.\"",
                "\"You are closer than you think… to something worth the risk.\"",
                "\"Your momentum is building… and it will soon carry you further than expected.\"",
                "\"The next step you take boldly… will open more than one path.\""
            ]
        }
    elif skip_count >= 12 and skip_ratio >= 0.7:
        return {
            "segment": "high-discernment-skipper",
            "like_count": like_count,
            "skip_count": skip_count,
            "total": total,
            "like_ratio": like_ratio,
            "skip_ratio": skip_ratio,
            "pool": [
                "\"You are guided by discernment, not distraction. A clearer path is forming ahead of you.\"",
                "\"You reject what is not yours… and that wisdom protects your future.\"",
                "\"Your restraint is a strength… by turning away, you make space for what matters.\"",
                "\"Your future sharpens each time you refuse what does not fit.\"",
                "\"Clarity is becoming your greatest advantage.\"",
                "\"You are cutting through illusions… and something real will soon reveal itself.\"",
                "\"Your ability to say no is quietly shaping a more honest destiny.\"",
                "\"You are not missing out. You are refining what truly belongs to you.\"",
                "\"By filtering the noise, you are making room for something rare.\"",
                "\"What you release now… creates space for something more precise.\"",
                "\"Your standards are rising… and your path is adjusting to match.\""
                "\"You see what others overlook… and that awareness will guide you forward.\"",
                "\"By choosing less… you are preparing to receive more of what matters.\"",
                "\"Your patience is not empty… it is quietly aligning things in your favor.\"",
                "\"You are narrowing the path… until only the right one remains.\""
            ]
        }
    elif like_count > skip_count:
        return {
            "segment": "curious-openness",
            "like_count": like_count,
            "skip_count": skip_count,
            "total": total,
            "like_ratio": like_ratio,
            "skip_ratio": skip_ratio,
            "pool": [
                "\"You follow curiosity with an open heart. Soon, something new will feel instantly familiar.\"",
                "\"You are drawn to possibility, and that openness will bring a fortunate surprise.\"",
                "\"Your future carries momentum. What excites you now is pointing toward what comes next.\"",
                "\"You are expanding faster than you realize, and the world is beginning to respond.\"",
                "\"An unexpected invitation will match the energy you have been moving toward.\"",
                "\"Your openness will lead you somewhere richer than your original plan.\"",
                "\"What delights you now is more than a preference… it is a direction.\"",
                "\"You are gathering signals from the future through what draws you in today.\"",
                "\"A joyful instinct will soon prove wiser than logic alone.\"",
                "\"Curiosity is guiding you… and it is leading somewhere worth arriving.\"",
                "\"What you choose with lightness… will quietly shape something meaningful.\""
                "\"You are following a feeling… and it is closer to truth than it seems.\"",
                "\"Your attention is shifting… and new doors are adjusting to meet it.\"",
                "\"The things that spark interest now… will soon begin to connect.\"",
                "\"You are exploring the right edges… and something unexpected is waiting there.\""
            ]
        }
    elif skip_count > like_count:
        return {
            "segment": "refining-path",
            "like_count": like_count,
            "skip_count": skip_count,
            "total": total,
            "like_ratio": like_ratio,
            "skip_ratio": skip_ratio,
            "pool": [
                "\"You trust your inner filter, and it is sharpening your destiny.\"",
                "\"You are narrowing the noise around you… what remains will truly matter.\"",
                "\"Your future grows clearer with every choice you refuse.\"",
                "\"You are learning through elimination… and that knowledge is powerful.\"",
                "\"What you reject now is protecting the shape of what comes next.\"",
                "\"The path ahead is revealing itself… now that you no longer chase everything.\"",
                "\"There is power in your refusal, and it will soon reveal purpose.\"",
                "\"By stepping away from the wrong things… you move closer to the right one.\"",
                "\"Your patience with imperfection will soon be rewarded by something unmistakable.\"",
                "\"Each no you choose… brings the right yes into focus.\"",
                "\"You are refining your path… until only what matters remains.\"",
                "\"What you dismiss now… would have distracted you later.\"",
                "\"Your clarity is cutting deeper… and revealing what is true.\"",
                "\"You are leaving behind what no longer resonates… and the future is adjusting.\"",
                "\"Less is becoming more… and it is guiding you somewhere precise.\""
            ]
        }
    else:
        return {
            "segment": "balanced-thinker",
            "like_count": like_count,
            "skip_count": skip_count,
            "total": total,
            "like_ratio": like_ratio,
            "skip_ratio": skip_ratio,
            "pool": [
                "\"You balance instinct and caution with rare precision. A meaningful choice is approaching.\"",
                "\"You move carefully, but without fear… and that balance will serve you well\"",
                "\"You are learning not only what you want, but why. That knowledge will shape your next chapter.\"",
                "\"You stand between desire and discernment, and that is where wisdom grows.\"",
                "\"A balanced heart often sees what others overlook.\"",
                "\"You are not divided. You are measuring the world with care.\"",
                "\"Because you weigh both impulse and restraint... your next step will carry unusual strength.\"",
                "\"Your future is being built on thoughtful tension, and that makes it resilient.\"",
                "\"The harmony between your curiosity and caution will soon reveal a clear answer.\"",
                "\"You are holding both paths in view… and clarity is beginning to settle.\"",
                "\"Balance is guiding you… quietly aligning your next decision.\""
                "\"You are neither rushing nor resisting… and that is shaping something steady.\"",
                "\"Your awareness of both risks and rewards… will soon reveal the right moment.\"",
                "\"You are finding center… and from there, your direction will strengthen.\"",
                "\"What you consider now… will soon crystallize into certainty.\""
            ]
        }




def get_preview_fortune(uid: str):
    state = get_fortune_state(uid)
    pool = state["pool"]
    total = state["total"]


    if not pool:
        return None


    idx = min(total, len(pool) - 1)
    return pool[idx]




def generate_personalized_fortune(uid: str):
    return get_preview_fortune(uid)




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
    result = item.copy()
    if result["path"].startswith("data/"):
        filename = result["path"].split("/")[-1]
        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        result["path"] = f"{base_url}/images/{filename}"
    return result




def serialize_item(item_id: str, score: float = None, row_idx: int = None):
    if item_id not in ITEMS_META:
        return None
    result = convert_path_to_url(ITEMS_META[item_id])
    if score is not None:
        result["score"] = float(score)
    if row_idx is not None:
        result["row_idx"] = int(row_idx)
    return result




def build_feed_items(uid: str, limit: int = 15, exclude_seen: bool = True, include_debug: bool = False):
    u = get_user_vec(uid)
    k_candidates = max(limit * 5, 100)


    if u is None:
        items = []
        for item_id in ITEM_IDS:
            if item_id not in ITEMS_META:
                continue
            if not exclude_seen or not has_seen(uid, item_id):
                items.append(item_id)
            if len(items) >= limit * 3:
                break


        if include_debug:
            result = [serialize_item(i) for i in items[:limit]]
            return [x for x in result if x is not None]


        return [convert_path_to_url(ITEMS_META[i]) for i in items[:limit]]


    q = u.reshape(1, -1).astype("float32")


    attempts = [
        k_candidates,
        k_candidates * 3,
        k_candidates * 10,
        len(ITEM_IDS)
    ]


    cand = []


    for attempt_k in attempts:
        attempt_k = min(attempt_k, len(ITEM_IDS))
        sims, idxes = index.search(q, attempt_k)
        idxes, sims = idxes[0], sims[0]


        cand = [
            (ITEM_IDS[i], float(sims[j]), i)
            for j, i in enumerate(idxes)
            if 0 <= i < len(ITEM_IDS)
            and ITEM_IDS[i] in ITEMS_META
            and (not exclude_seen or not has_seen(uid, ITEM_IDS[i]))
        ]


        if len(cand) >= limit:
            break


    if len(cand) < limit:
        liked_seen = {
            iid for iid in USER_SEEN.get(uid, set())
            if POPULARITY.get(iid, 0.0) > 0
        }


        sims, idxes = index.search(q, min(k_candidates, len(ITEM_IDS)))
        idxes, sims = idxes[0], sims[0]
        cand = [
            (ITEM_IDS[i], float(sims[j]), i)
            for j, i in enumerate(idxes)
            if 0 <= i < len(ITEM_IDS)
            and ITEM_IDS[i] in ITEMS_META
            and ITEM_IDS[i] not in liked_seen
        ]


    if len(cand) < limit:
        already = {c[0] for c in cand}
        for i, item_id in enumerate(ITEM_IDS):
            if (
                item_id in ITEMS_META
                and item_id not in already
                and item_id not in USER_SEEN.get(uid, set())
            ):
                sim = float(np.dot(u, ITEM_VECS[i]))
                cand.append((item_id, sim, i))
            if len(cand) >= limit * 2:
                break


    if not cand:
        return []


    scored = score_items(
        u,
        np.array([c[2] for c in cand]),
        np.array([c[1] for c in cand]),
        len(cand)
    )
    reranked = mmr_rerank(u, scored, limit, lam=0.7)


    if include_debug:
        result = [
            serialize_item(iid, score=score, row_idx=row_idx)
            for (iid, score, row_idx) in reranked
            if iid in ITEMS_META
        ]
        return [x for x in result if x is not None]


    return [
        convert_path_to_url(ITEMS_META[iid])
        for (iid, _, _) in reranked
        if iid in ITEMS_META
    ]




def get_current_feed_preview(uid: str, limit: int = 8, include_active: bool = True):
    stored_feed = USER_CURRENT_FEED.get(uid)

    if not stored_feed:
        return []

    current_idx = USER_FEED_INDEX.get(uid, 0)

    if include_active:
        feed_slice = stored_feed[current_idx:]
    else:
        feed_slice = stored_feed[current_idx + 1:]

    feed_slice = feed_slice[:limit]

    annotated = []
    for i, item in enumerate(feed_slice):
        item = item.copy()
        item["position"] = current_idx + i + 1
        item["is_active"] = (i == 0)
        annotated.append(item)

    return annotated




@app.post("/feed")
def feed(req: FeedRequest):
    try:
        mark_active(req.user_id)
        items = build_feed_items(
            uid=req.user_id,
            limit=req.limit,
            exclude_seen=req.exclude_seen,
            include_debug=True
        )
        USER_CURRENT_FEED[req.user_id] = items
        clean = [{k: v for k, v in item.items() if k not in ("score", "row_idx")} for item in items]
        return {"items": clean}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/advance")
def advance(req: Advance):
    mark_active(req.user_id)

    stored_feed = USER_CURRENT_FEED.get(req.user_id, [])
    for i, item in enumerate(stored_feed):
        if item.get("item_id") == req.item_id:
            USER_FEED_INDEX[req.user_id] = i
            return {"ok": True}

    return {"ok": True}




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
    state = get_fortune_state(user_id)
    preview = get_preview_fortune(user_id)
    return {
        "user_id": user_id,
        "history_count": len(history),
        "segment": state["segment"],
        "fortune": preview,
        "fortune_preview": preview,
        "fortune_pool_size": len(state["pool"]),
        "fortune_total_interactions": state["total"],
        "fortune_like_ratio": state["like_ratio"],
        "fortune_skip_ratio": state["skip_ratio"],
    }




@app.post("/reset")
def reset(req: ResetRequest):
    global LAST_ACTIVE_USER, LAST_ACTIVITY_AT


    USER_VEC.pop(req.user_id, None)
    USER_SEEN.pop(req.user_id, None)
    USER_HISTORY.pop(req.user_id, None)
    USER_CURRENT_FEED.pop(req.user_id, None)
    USER_FEED_INDEX.pop(req.user_id, None)


    if LAST_ACTIVE_USER == req.user_id:
        LAST_ACTIVE_USER = None
        LAST_ACTIVITY_AT = 0.0


    return {"ok": True}




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
    state = get_fortune_state(user_id)
    current_feed = get_current_feed_preview(user_id, limit=8, include_active=True)


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
        "fortune_segment": state["segment"],
        "fortune_preview": get_preview_fortune(user_id),
        "fortune_pool_size": len(state["pool"]),
        "fortune_total_interactions": state["total"],
        "fortune_like_ratio": state["like_ratio"],
        "fortune_skip_ratio": state["skip_ratio"],
        "current_feed": current_feed,
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
            "fortune_segment": None,
            "fortune_preview": None,
            "fortune_pool_size": 0,
            "fortune_total_interactions": 0,
            "fortune_like_ratio": 0.0,
            "fortune_skip_ratio": 0.0,
            "current_feed": [],
        }


    uid = LAST_ACTIVE_USER
    history = USER_HISTORY.get(uid, [])
    liked = [h["item_id"] for h in history if h["action"] == "like"]
    skipped = [h["item_id"] for h in history if h["action"] == "skip"]
    state = get_fortune_state(uid)
    current_feed = get_current_feed_preview(uid, limit=8, include_active=True)


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
        "fortune_segment": state["segment"],
        "fortune_preview": get_preview_fortune(uid),
        "fortune_pool_size": len(state["pool"]),
        "fortune_total_interactions": state["total"],
        "fortune_like_ratio": state["like_ratio"],
        "fortune_skip_ratio": state["skip_ratio"],
        "current_feed": current_feed,
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




@app.get("/admin/current-feed/{user_id}")
def admin_current_feed(user_id: str, limit: int = 8):
    return {
        "user_id": user_id,
        "current_feed": get_current_feed_preview(user_id, limit=limit, include_active=True)
    }




@app.post("/admin/preview-feed/{user_id}")
def admin_preview_feed(user_id: str, limit: int = 15):
    items = build_feed_items(
        uid=user_id,
        limit=limit,
        exclude_seen=True,
        include_debug=False
    )
    return {"items": items}




@app.get("/admin/feed.csv")
def admin_feed_csv():
    if not LAST_ACTIVE_USER:
        content = "#,status,item_id,title\n"
    else:
        items = get_current_feed_preview(LAST_ACTIVE_USER, limit=15, include_active=True)
        lines = ["#,status,item_id,title"]
        for item in items:
            status  = "ACTIVE" if item.get("is_active") else f"next {(item.get('position', 1) - 1)}"
            item_id = str(item.get("item_id", "")).replace('"', '""')
            title   = str(item.get("title",   "")).replace('"', '""')
            lines.append(f'{item.get("position", "")},{status},"{item_id}","{title}"')
        content = "\n".join(lines)
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Refresh": "2",
        }
    )