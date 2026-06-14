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

const reportAdvance = (itemId) => {
  axios.post(`${API}/advance`, { user_id: USER_ID, item_id: itemId }).catch(() => {});
};

function Card({
  item,
  onSwipeLeft,
  onSwipeRight,
  isTop = true,
  onDragChange,
  introNudge = false,
  onIntroDone,
}) {
  const [dragX, setDragX] = useState(0);
  const [exitDir, setExitDir] = useState(null);
  const [introX, setIntroX] = useState(0);
  const hasRunIntroRef = useRef(false);

  const handlers = useSwipeable({
    onSwiping: ({ deltaX }) => {
      if (!isTop) return;
      setDragX(deltaX);
      onDragChange?.(deltaX);
    },
    onSwipedLeft: () => {
      if (!isTop) return;
      setExitDir("left");
      onDragChange?.(0);
      setTimeout(() => onSwipeLeft(item), 220);
    },
    onSwipedRight: () => {
      if (!isTop) return;
      setExitDir("right");
      onDragChange?.(0);
      setTimeout(() => onSwipeRight(item), 220);
    },
    onSwiped: () => {
      if (!exitDir) {
        setDragX(0);
        onDragChange?.(0);
      }
    },
    trackMouse: true,
    preventScrollOnSwipe: true,
  });

  useEffect(() => {
    setDragX(0);
    setExitDir(null);
    setIntroX(0);
    hasRunIntroRef.current = false;
    onDragChange?.(0);
  }, [item?.item_id || item?.title || item]);

  useEffect(() => {
    if (!isTop || !introNudge || hasRunIntroRef.current) return;

    hasRunIntroRef.current = true;

    const start = setTimeout(() => setIntroX(16), 700);
    const back = setTimeout(() => setIntroX(-6), 1050);
    const settle = setTimeout(() => setIntroX(0), 1400);
    const done = setTimeout(() => onIntroDone?.(), 1700);

    return () => {
      clearTimeout(start);
      clearTimeout(back);
      clearTimeout(settle);
      clearTimeout(done);
    };
  }, [isTop, introNudge, onIntroDone]);

  return (
    <div style={{ width: 1034, overflow: "hidden", marginLeft: "-61em" }}>
      <div
        {...(isTop ? handlers : {})}
        style={{
          transform:
            exitDir === "left"
              ? "translateX(-160%) rotate(-10deg)"
              : exitDir === "right"
              ? "translateX(160%) rotate(10deg)"
              : isTop
              ? `translateX(${dragX + introX}px) rotate(${(dragX + introX) * 0.05}deg)`
              : "none",
          transition: exitDir
            ? "transform 220ms ease-out"
            : introX !== 0
            ? "transform 450ms ease-in-out"
            : "none",
          willChange: "transform",
          touchAction: "pan-y",
        }}
      >
        <img
          src={item.path || item.url}
          alt={item.title || item.item_id}
          style={{
            width: "90%",
            height: 1000,
            objectFit: "cover",
            marginLeft: "3em",
            borderRadius: 20,
          }}
        />
      </div>
    </div>
  );
}

function ActionButtons({ onSkip, onLike, dragX = 0, disabled = false }) {
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const maxGrow = 0.2;
  const intensity = 220;

  const likeScale =
    dragX > 0 ? 1 + clamp(Math.abs(dragX) / intensity, 0, maxGrow) : 1;

  const skipScale =
    dragX < 0 ? 1 + clamp(Math.abs(dragX) / intensity, 0, maxGrow) : 1;

  return (
    <div
      style={{
        display: "flex",
        gap: 8,
        justifyContent: "center",
        width: 1034,
        marginLeft: "-61em",
        marginTop: -200,
        paddingTop: 45,
        overflow: "hidden",
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
        onClick={onSkip}
        disabled={disabled}
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
        onClick={onLike}
        disabled={disabled}
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
  );
}

export default function App() {
  const [queue, setQueue] = useState([]);
  const [interactionCount, setInteractionCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [dragX, setDragX] = useState(0);
  const [introNudge, setIntroNudge] = useState(true);
  const hasFinishedRef = useRef(false);
  const isInternalNavigationRef = useRef(false);

  const maxInteractions = 20;
  const progressPercent = Math.min((interactionCount / maxInteractions) * 100, 100);

  const fetchFeed = async () => {
    try {
      setLoading(true);
      setError("");

      const res = await axios.post(`${API}/feed`, {
        user_id: USER_ID,
        limit: 10,
        exclude_seen: true,
      });

      const items = res.data.items || [];
      setQueue(items);

      if (items.length > 0) {
        reportAdvance(items[0].item_id);
      }
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

  const goToNextPage = async () => {
    hasFinishedRef.current = true;
    isInternalNavigationRef.current = true;
    window.location.href = `${import.meta.env.BASE_URL}end1.html`;
  };

  const handleInteraction = (item, action) => {
    setIntroNudge(false);

    setQueue((q) => {
      const next = q.slice(1);
      if (next.length > 0) {
        reportAdvance(next[0].item_id);
      }
      return next;
    });

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
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
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
          <div style={{ position: "relative" }}>
            {queue.slice(0, 4).map((item, index) => {
              const isTop = index === 0;

              return (
                <div
                  key={item.item_id}
                  style={{
                    position: isTop ? "relative" : "absolute",
                    top: isTop ? "auto" : 0,
                    left: isTop ? "auto" : 0,
                    zIndex: 100 - index,
                    marginTop: "100px",
                    transform: isTop
                      ? "none"
                      : `translateY(${index * 14}px) scale(${1 - index * 0.03})`,
                    opacity: isTop ? 1 : 0.9 - index * 0.1,
                    pointerEvents: isTop ? "auto" : "none",
                    transition: "transform 180ms ease, opacity 180ms ease",
                  }}
                >
                  <Card
                    item={item}
                    onSwipeLeft={onSkip}
                    onSwipeRight={onLike}
                    isTop={isTop}
                    onDragChange={setDragX}
                    introNudge={isTop && introNudge}
                    onIntroDone={() => setIntroNudge(false)}
                  />
                </div>
              );
            })}
          </div>

          <ActionButtons
            onSkip={() => onSkip(queue[0])}
            onLike={() => onLike(queue[0])}
            dragX={dragX}
            disabled={!queue.length}
          />

          <div
            style={{
              position: "fixed",
              bottom: -10,
              left: "50%",
              transform: "translateX(-50%)",
              width: 1046,
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
                height: 24,
                backgroundColor: "#00000000",
                overflow: "hidden",
                borderTop: "5px solid #ffffff",
                marginTop: -50,
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