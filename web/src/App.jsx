import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useSwipeable } from "react-swipeable";
import HeartIcon from "../public/assets/img/LikeLogo.svg";
import SkipIcon from "../public/assets/img/SkipLogo.svg";

const API = import.meta.env.DEV
  ? "http://localhost:8000"
  : "https://alg-backend.onrender.com";

const existingUserId = sessionStorage.getItem("user_id");
const USER_ID = existingUserId || `user-${crypto.randomUUID()}`;

if (!existingUserId) {
  sessionStorage.setItem("user_id", USER_ID);
}

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

  useEffect(() => {
    setDragX(0);
    setExitDir(null);
  }, [item?.item_id || item?.title || item]);

  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const maxGrow = 0.2;
  const intensity = 220;

  const likeScale =
    dragX > 0 ? 1 + clamp(Math.abs(dragX) / intensity, 0, maxGrow) : 1;

  const skipScale =
    dragX < 0 ? 1 + clamp(Math.abs(dragX) / intensity, 0, maxGrow) : 1;

  return (
    <div style={{ width: 1024, overflow: "hidden", marginLeft: "-61em" }}>
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
          style={{
            width: "90%",
            height: 1150,
            objectFit: "cover",
            marginLeft: "3em",
            borderRadius: 20,
          }}
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

      <div
        style={{
          padding: 12,
          display: "flex",
          gap: 8,
          justifyContent: "center",
          marginTop: -10,
        }}
      >
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

        <button
          style={{
            width: 300,
            height: 85,
            marginRight: -80,
            marginTop: -40,
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
  const [interactionCount, setInteractionCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const hasFinishedRef = useRef(false);
  const isInternalNavigationRef = useRef(false);

  const maxInteractions = 20;
  const progressPercent = Math.min(
    (interactionCount / maxInteractions) * 100,
    100
  );

  const fetchFeed = async () => {
    try {
      setLoading(true);
      setError("");

      const res = await axios.post(`${API}/feed`, {
        user_id: USER_ID,
        limit: 10,
        exclude_seen: true,
      });

      setQueue(res.data.items || []);
    } catch (err) {
      console.error("Feed failed:", err);
      setError("Could not load images. Please refresh or try again.");
    } finally {
      setLoading(false);
    }
  };

  const sendInteraction = async (item, action) => {
    try {
      await axios.post(`${API}/interactions`, {
        user_id: USER_ID,
        item_id: item.item_id,
        action,
      });
    } catch (err) {
      console.error("Interaction failed:", err);
    }
  };

  const resetSession = async () => {
    try {
      await axios.post(`${API}/reset`, {
        user_id: USER_ID,
      });
    } catch (err) {
      console.error("Reset failed:", err);
    }
  };

  const goToNextPage = async () => {
    hasFinishedRef.current = true;
    isInternalNavigationRef.current = true;
    window.location.href = `${import.meta.env.BASE_URL}end1.html`;
  };

  const handleInteraction = (item, action) => {
    setQueue((q) => q.slice(1));
    sendInteraction(item, action);

    setInteractionCount((prev) => {
      const next = prev + 1;

      if (next >= maxInteractions) {
        setTimeout(() => {
          goToNextPage();
        }, 250);
      }

      return next;
    });
  };

  const onLike = (item) => handleInteraction(item, "like");
  const onSkip = (item) => handleInteraction(item, "skip");

  useEffect(() => {
    fetchFeed();
  }, []);

  useEffect(() => {
    if (!loading && !error && queue.length < 5 && interactionCount < maxInteractions) {
      fetchFeed();
    }
  }, [queue.length, loading, error, interactionCount]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      if (hasFinishedRef.current || isInternalNavigationRef.current) return;

      const data = JSON.stringify({ user_id: USER_ID });
      navigator.sendBeacon(
        `${API}/reset`,
        new Blob([data], { type: "application/json" })
      );
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
      {loading ? (
        <div style={{ color: "#ffffff" }}>Loading…</div>
      ) : error ? (
        <div style={{ color: "#ffffff", textAlign: "center" }}>
          <p>{error}</p>
          <button onClick={fetchFeed}>Try again</button>
        </div>
      ) : queue.length ? (
        <>
          <Card
            key={queue[0].item_id}
            item={queue[0]}
            onLike={onLike}
            onSkip={onSkip}
          />

          <div
            style={{
              position: "fixed",
              bottom: "2rem",
              left: "50%",
              transform: "translateX(-50%)",
              width: "420px",
              zIndex: 1000,
            }}
          >
            <div
              role="progressbar"
              aria-label="Interaction progress"
              aria-valuemin={0}
              aria-valuemax={maxInteractions}
              aria-valuenow={interactionCount}
              style={{
                width: "100%",
                height: 12,
                backgroundColor: "#000000",
                borderRadius: 999,
                overflow: "hidden",
                border: "5px solid #ffffff",
                padding: 25,
                marginTop: "-11em",
                marginLeft: "-2em",
              }}
            >
              <div
                style={{
                  width: `${progressPercent}%`,
                  height: "100%",
                  backgroundColor: "#C0FD02",
                  borderRadius: 999,
                  transition: "width 180ms ease",
                }}
              />
            </div>
          </div>
        </>
      ) : (
        <div style={{ color: "#ffffff" }}>No more images available.</div>
      )}
    </div>
  );
}