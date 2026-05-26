import { useEffect, useState } from "react";
import axios from "axios";
import { useSwipeable } from "react-swipeable"; 
import HeartIcon from '../public/assets/img/LikeLogo.svg';
import SkipIcon from '../public/assets/img/SkipLogo.svg';

const API = import.meta.env.DEV 
  ? "http://localhost:8000"  
  : "https://alg-backend.onrender.com"; 
const USER_ID = "demo-user-1";

function Card({ item, onLike, onSkip }) {
  const [dragX, setDragX] = useState(0);
  const [exitDir, setExitDir] = useState(null);

  const handlers = useSwipeable({
    onSwiping: ({ deltaX }) => setDragX(deltaX),

    onSwipedLeft: () => {
      setExitDir("left");
      setTimeout(() => onSkip(item), 220);
    },

    onSwipedRight: () => {
      setExitDir("right");
      setTimeout(() => onLike(item), 220);
    },

    onSwiped: () => {
      if (!exitDir) setDragX(0);
    },

    trackMouse: true,
    preventScrollOnSwipe: true,
  });

  // Reset animation when switching to next card
  useEffect(() => {
    setDragX(0);
    setExitDir(null);
  }, [item?.item_id || item?.title || item]);

  // --- Button + icon grow animation ---
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

  const maxGrow = 0.20;  // +20% bigger
  const intensity = 220; // px to reach max growth

  const likeScale =
    dragX > 0 ? 1 + clamp(Math.abs(dragX) / intensity, 0, maxGrow) : 1;

  const skipScale =
    dragX < 0 ? 1 + clamp(Math.abs(dragX) / intensity, 0, maxGrow) : 1;

  return (
    <div style={{ width: 1024, overflow: "hidden", marginLeft: "-40em", }}>

      {/* SWIPING AREA — THIS MOVES */}
      <div
        {...handlers}
        style={{
          transform:
            exitDir === "left"
              ? "translateX(-160%) rotate(-10deg)"
              : exitDir === "right"
              ? "translateX(160%) rotate(10deg)"
              : `translateX(${dragX}px) rotate(${dragX * 0.05}deg)`,
          transition: exitDir ? "transform 220ms ease-out" : "none",
          willChange: "transform",
          touchAction: "pan-y",
        }}
      >
        <img
          src={item.path || item.url}
          alt={item.title || item.item_id}
          style={{ width: "90%", height: 1150, objectFit: "cover", marginLeft: "3em", borderRadius: 20, }}
        />

        <div style={{ padding: 12 }}>
          <div
            style={{
              fontWeight: 600,
              fontSize: 24,
              textAlign: "center",
              color: "#213547",
            }}
          >
            {item.title || item.item_id}
          </div>
        </div>
      </div>

      {/* STATIC BUTTONS — THEY DO NOT MOVE WITH SWIPE */}
      <div
        style={{
          padding: 12,
          display: "flex",
          gap: 8,
          justifyContent: "center",
          marginTop: -10, // optional small lift upward
        }}
      >

        {/* SKIP BUTTON */}
        <button
          style={{
            width: 300,
            height: 85,
            marginLeft: -80,
            marginTop: -45, 
            display: "flex",
            alignItems: "right",
            justifyContent: "right",
            borderColor: "#ffffff",
            border: "5px solid #ffffff",
            borderRadius: "25px",
            transform: `scale(${skipScale})`,
            transition: "transform 120ms ease-out",
            backgroundColor: "#000000",
          }}
          onClick={() => onSkip(item)}
        >
          <img
            src={SkipIcon}
            alt="Skip"
            style={{
              width: 55,
              height: 55,

              transform: `scale(${skipScale})`,
              transition: "transform 120ms ease-out",
            }}
          />
        </button>

        {/* LIKE BUTTON */}
        <button
          style={{
            width: 300,
            height: 85,
            marginRight: -80,
            marginTop: -40, // optional small lift upward
            display: "flex",
            alignItems: "left",
            justifyContent: "left",
            marginLeft: "auto",
            borderColor: "#ffffff",
            border: "5px solid #ffffff",
            borderRadius: "25px",
            transform: `scale(${likeScale})`,
            transition: "transform 120ms ease-out",
            backgroundColor: "#000000",
          }}
          onClick={() => onLike(item)}
        >
          <img
            src={HeartIcon}
            alt="Like"
            style={{
              width: 55,
              height: 55,
              
              transform: `scale(${likeScale})`,
              transition: "transform 120ms ease-out",
            }}
          />
        </button>

      </div>
    </div>
  );
}

export default function App() {
  const [queue, setQueue] = useState([]);

  const fetchFeed = async () => {
    const res = await axios.post(`${API}/feed`, {
      user_id: USER_ID,
      limit: 10,
      exclude_seen: true
    });
    setQueue(res.data.items);
  };

  const sendInteraction = (item, action) => {
    axios.post(`${API}/interactions`, {
      user_id: USER_ID,
      item_id: item.item_id,
      action
    }).catch(() => {});
  };

  // Optimistic UI update for instant next card
  const onLike = (item) => {
    setQueue(q => q.slice(1));
    sendInteraction(item, "like");
    if (queue.length < 5) fetchFeed();
  };

  const onSkip = (item) => {
    setQueue(q => q.slice(1));
    sendInteraction(item, "skip");
    if (queue.length < 5) fetchFeed();
  };

  useEffect(() => {
    fetchFeed();
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
      {queue.length ? (
        <Card
          key={queue[0].item_id}
          item={queue[0]}
          onLike={onLike}
          onSkip={onSkip}
        />
      ) : (
        <div>Loading…</div>
      )}
    </div>
  );
}