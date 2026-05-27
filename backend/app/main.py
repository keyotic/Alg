import os
import json
import time
import traceback
import logging

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alg-backend")


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

ROOT_ARTIFACTS = os.path.join(PROJECT_ROOT, "artifacts")
ROOT_DATA = os.path.join(PROJECT_ROOT, "data")

INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(ROOT_ARTIFACTS, "faiss.index"))
IDS_PATH = os.getenv("IDS_PATH", os.path.join(ROOT_ARTIFACTS, "item_ids.json"))
VECTORS_PATH = os.getenv("VECTORS_PATH", os.path.join(ROOT_ARTIFACTS, "item_vectors.npy"))
ITEMS_JSON = os.getenv("ITEMS_JSON", os.path.join(ROOT_DATA, "items.json"))


index = faiss.read_index(INDEX_PATH)

with open(IDS_PATH, "r") as f:
    ITEM_IDS = json.load(f)

ITEM_VECS = np.load(VECTORS_PATH)

with open(ITEMS_JSON, "r") as f:
    ITEMS_META = {it["item_id"]: it for it in json.load(f)}

missing_in_meta = [iid for iid in ITEM_IDS if iid not in ITEMS_META]
extra_in_meta = [iid for iid in ITEMS_META if iid not in ITEM_IDS]

logger.info("ITEM_IDS count=%s | ITEMS_META count=%s", len(ITEM_IDS), len(ITEMS_META))

if missing_in_meta:
    logger.warning("ITEM_IDS missing from ITEMS_META: %s", missing_in_meta[:20])
    logger.warning("Total ITEM_IDS missing from ITEMS_META: %s", len(missing_in_meta))

if extra_in_meta:
    logger.warning("ITEMS_META missing from ITEM_IDS: %s", extra_in_meta[:20])
    logger.warning("Total ITEMS_META missing from ITEM_IDS: %s", len(extra_in_meta))

EMBED_DIM = ITEM_VECS.shape[1]

USER_VEC = {}
USER_SEEN = {}

POPULARITY = {iid: 0.0 for iid in ITEM_IDS}
CREATED_AT = {iid: time.time() for iid in ITEM_IDS}


class FeedRequest(BaseModel):
    user_id: str
    limit: int = 15
    exclude_seen: bool = True


class Interaction(BaseModel):
    user_id: str
    item_id: str
    action: str


class ResetRequest(BaseModel):
    user_id: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://keyotic.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/images",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "data", "images")),
    name="images"
)


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "count": len(ITEM_IDS),
        "meta_count": len(ITEMS_META),
        "version": "main.py-safe-meta-v2"
    }


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
        if like:
            u = item_vec.copy()
        else:
            u = np.random.randn(EMBED_DIM).astype("float32")
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
        recency = np.exp(-max(0, now - CREATED_AT.get(item_id, now)) / (7 * 24 * 3600))
        score = alpha * float(sim) + beta * pop + gamma * recency
        scored.append((item_id, score, row_idx))

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

    if "path" not in result:
        logger.error("Missing 'path' in item metadata: %s", result)
        return result

    if not isinstance(result["path"], str):
        logger.error("Invalid 'path' type in item metadata: %s", result)
        return result

    if result["path"].startswith("data/"):
        filename = result["path"].split("/")[-1]
        full_path = os.path.join(PROJECT_ROOT, "data", "images", filename)

        if not os.path.exists(full_path):
            logger.error("Image file missing on disk: %s | item=%s", filename, result)

        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        result["path"] = f"{base_url}/images/{filename}"

    return result


@app.post("/feed")
def feed(req: FeedRequest):
    uid = req.user_id
    u = get_user_vec(uid)
    k_candidates = max(req.limit * 5, 100)

    logger.info("---- /feed start ----")
    logger.info("user_id=%s limit=%s exclude_seen=%s", uid, req.limit, req.exclude_seen)
    logger.info("user_vec_exists=%s", u is not None)

    try:
        if u is None:
            items = []
            for item_id in ITEM_IDS:
                if not req.exclude_seen or not has_seen(uid, item_id):
                    items.append(item_id)
                if len(items) >= req.limit:
                    break

            result = []
            for item_id in items:
                meta = ITEMS_META.get(item_id)
                if meta is None:
                    logger.error("Missing metadata during cold start for item_id=%s", item_id)
                    continue
                result.append(convert_path_to_url(meta))

            logger.info("cold_start_items_returned=%s", len(result))
            return {"items": result}

        q = u.reshape(1, -1).astype("float32")

        attempts = [
            k_candidates,
            k_candidates * 3,
            k_candidates * 10,
            len(ITEM_IDS),
        ]

        cand = []

        for attempt_k in attempts:
            attempt_k = min(attempt_k, len(ITEM_IDS))
            sims, idxes = index.search(q, attempt_k)
            idxes, sims = idxes[0], sims[0]

            cand = [
                (ITEM_IDS[i], sims[j], i)
                for j, i in enumerate(idxes)
                if not req.exclude_seen or not has_seen(uid, ITEM_IDS[i])
            ]

            logger.info("attempt_k=%s candidate_count=%s", attempt_k, len(cand))

            if len(cand) >= req.limit:
                break

        if len(cand) < req.limit:
            logger.warning("Not enough candidates; resetting seen set for user=%s", uid)
            USER_SEEN[uid] = set()

            sims, idxes = index.search(q, min(k_candidates, len(ITEM_IDS)))
            idxes, sims = idxes[0], sims[0]
            cand = [(ITEM_IDS[i], sims[j], i) for j, i in enumerate(idxes)]

            logger.info("candidate_count_after_seen_reset=%s", len(cand))

        if not cand:
            logger.error("No candidates found for user=%s", uid)
            return {"items": []}

        scored = score_items(
            u,
            np.array([c[2] for c in cand]),
            np.array([c[1] for c in cand]),
            len(cand)
        )

        logger.info("scored_count=%s", len(scored))

        if not scored:
            logger.error("No scored items for user=%s", uid)
            return {"items": []}

        reranked = mmr_rerank(u, scored, req.limit, lam=0.7)
        logger.info("reranked_count=%s", len(reranked))

        if not reranked:
            logger.warning("No reranked items for user=%s", uid)
            return {"items": []}

        final_items = []
        missing_meta_ids = []

        for (iid, _, _) in reranked:
            meta = ITEMS_META.get(iid)
            if meta is None:
                missing_meta_ids.append(iid)
                logger.error("Missing metadata for reranked item_id=%s", iid)
                continue

            final_items.append(convert_path_to_url(meta))

        if missing_meta_ids:
            logger.warning("Skipped reranked items with missing metadata: %s", missing_meta_ids)

        logger.info("final_items_count=%s", len(final_items))
        logger.info("---- /feed end ----")

        return {"items": final_items}

    except Exception as e:
        logger.error("Feed crashed for user=%s error=%s", uid, str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Feed failed: {str(e)}")


@app.post("/interactions")
def interactions(evt: Interaction):
    logger.info("---- /interactions start ----")
    logger.info("user_id=%s item_id=%s action=%s", evt.user_id, evt.item_id, evt.action)

    if evt.action not in ("like", "skip"):
        raise HTTPException(status_code=400, detail="Invalid action")

    try:
        row_idx = ITEM_IDS.index(evt.item_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown item_id")

    try:
        item_vec = ITEM_VECS[row_idx]
        like = evt.action == "like"

        update_user_vector(evt.user_id, item_vec, like)
        mark_seen(evt.user_id, evt.item_id)
        POPULARITY[evt.item_id] = POPULARITY.get(evt.item_id, 0.0) + (1.0 if like else 0.0)

        logger.info(
            "interaction_applied like=%s seen_count=%s popularity=%s",
            like,
            len(USER_SEEN.get(evt.user_id, set())),
            POPULARITY.get(evt.item_id, 0.0),
        )
        logger.info("---- /interactions end ----")

        return {"ok": True}

    except Exception as e:
        logger.error("Interaction crashed user=%s item_id=%s error=%s", evt.user_id, evt.item_id, str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Interaction failed: {str(e)}")


@app.post("/reset")
def reset_user(req: ResetRequest):
    logger.info("Resetting session for user_id=%s", req.user_id)
    USER_VEC.pop(req.user_id, None)
    USER_SEEN.pop(req.user_id, None)
    return {"ok": True}