import { useCallback, useEffect, useRef, useState } from "react";

export interface Playback {
  tRef: React.MutableRefObject<number>; // live play head (seconds) for useFrame
  t: number;                             // throttled mirror for DOM
  duration: number;
  playing: boolean;
  speed: number;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setSpeed: (s: number) => void;
  seek: (t: number) => void;
}

const MIRROR_MS = 66; // ~15 Hz DOM updates

export function usePlayback(duration: number): Playback {
  const tRef = useRef(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const playingRef = useRef(playing);
  const speedRef = useRef(speed);
  playingRef.current = playing;
  speedRef.current = speed;

  // Reset whenever the trajectory (its duration) changes; auto-play.
  useEffect(() => {
    tRef.current = 0;
    setT(0);
    setPlaying(duration > 0);
  }, [duration]);

  useEffect(() => {
    let raf = 0;
    let lastFrame = 0;
    let lastMirror = 0;
    const tick = (now: number) => {
      if (lastFrame === 0) lastFrame = now;
      const dt = (now - lastFrame) / 1000;
      lastFrame = now;
      if (playingRef.current && duration > 0) {
        let next = tRef.current + dt * speedRef.current;
        if (next >= duration) next = 0; // loop
        tRef.current = next;
      }
      if (now - lastMirror >= MIRROR_MS) {
        lastMirror = now;
        setT(tRef.current);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [duration]);

  const seek = useCallback((to: number) => {
    const clamped = Math.min(Math.max(to, 0), duration);
    tRef.current = clamped;
    setT(clamped);
  }, [duration]);

  return {
    tRef, t, duration, playing, speed,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    toggle: () => setPlaying((p) => !p),
    setSpeed,
    seek,
  };
}
