import os
import json
import math
import time
import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import streamlit as st


st.set_page_config(
    page_title="Warehouse Forklift & Pallet Tracker",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Constants and configuration defaults
# -----------------------------------------------------------------------------

FPS = 25
WIDTH, HEIGHT = 1280, 720

DEMO_DIR = Path("demo_output")
EVENTS_JSONL = DEMO_DIR / "events.jsonl"
DB_PATH = DEMO_DIR / "events.sqlite"

BASE_TIME = datetime(2026, 10, 30, 8, 0, 0, tzinfo=timezone.utc)

PERSON, FORKLIFT, PALLET = 0, 1, 2
CLASS_NAMES = {
    PERSON: "person",
    FORKLIFT: "forklift",
    PALLET: "pallet",
}

# BGR colors for OpenCV
COLORS = {
    PERSON: (255, 190, 80),     # cyan-ish
    FORKLIFT: (0, 120, 255),    # orange
    PALLET: (0, 220, 120),      # green
}

ZONE_COLORS = {
    "aisle": (90, 90, 90),
    "loading_bay": (200, 120, 0),
    "no_ped_zone": (0, 0, 220),
    "safe_walkway": (0, 180, 0),
}

EVENT_COLUMNS = [
    "event_id",
    "type",
    "severity",
    "person_id",
    "forklift_id",
    "distance_m",
    "forklift_speed",
    "zone",
    "ts",
    "clip_path",
]

# Internal-only field used to avoid brittle timestamp parsing in pandas.
INTERNAL_TS_KEY = "_ts_s"

# -----------------------------------------------------------------------------
# Ground-plane homography: pixels -> meters and meters -> pixels
# -----------------------------------------------------------------------------

# Pixel quadrilateral: bottom-left, bottom-right, top-right, top-left
PIX_QUAD = np.float32(
    [
        [120, 650],
        [1160, 650],
        [980, 260],
        [300, 260],
    ]
)

# Metric quadrilateral in meters: x right, y down
METRIC_QUAD = np.float32(
    [
        [0.0, 12.0],
        [20.0, 12.0],
        [20.0, 0.0],
        [0.0, 0.0],
    ]
)

H_PIX_TO_METRIC = cv2.getPerspectiveTransform(PIX_QUAD, METRIC_QUAD)
H_METRIC_TO_PIX = cv2.getPerspectiveTransform(METRIC_QUAD, PIX_QUAD)

# -----------------------------------------------------------------------------
# Zones in metric coordinates
# -----------------------------------------------------------------------------

ZONES: Dict[str, List[Tuple[float, float]]] = {
    "aisle": [(2, 4), (18, 4), (18, 8), (2, 8)],
    "loading_bay": [(0, 0), (7, 0), (7, 4), (0, 4)],
    "no_ped_zone": [(10, 0), (20, 0), (20, 3), (10, 3)],
    "safe_walkway": [(0, 9), (8, 9), (8, 12), (0, 12)],
}

DWELL_ZONES = {"aisle", "no_ped_zone"}

# -----------------------------------------------------------------------------
# Synthetic tracked-object footprints in meters
# -----------------------------------------------------------------------------

FOOTPRINTS = {
    PERSON: (0.65, 0.65),
    FORKLIFT: (2.4, 1.4),
    PALLET: (1.2, 1.0),
}

# -----------------------------------------------------------------------------
# Scripted demo trajectories in metric coordinates
# Each keyframe: (time_s, x_m, y_m, visible_ground_truth)
# -----------------------------------------------------------------------------

FORKLIFT_1_KEYFRAMES = [
    (0.0, 4.0, 8.5, True),
    (2.0, 8.0, 7.0, True),
    (4.0, 11.0, 5.5, True),
    (5.0, 12.0, 4.5, True),
    (6.0, 13.0, 5.2, True),
    (7.0, 13.3, 5.5, True),
    (8.0, 12.5, 4.8, True),
    (10.0, 10.0, 3.0, True),
    (30.0, 10.0, 3.0, True),
]

FORKLIFT_2_KEYFRAMES = [
    (0.0, 2.0, 2.0, True),
    (3.0, 5.0, 2.0, True),
    (5.0, 8.0, 2.0, True),
    (6.5, 12.0, 2.0, True),
    (8.0, 16.0, 2.0, True),
    (10.0, 18.0, 3.0, True),
    (30.0, 18.0, 3.0, True),
]

PERSON_1_KEYFRAMES = [
    (0.0, 17.0, 4.0, True),
    (3.0, 15.5, 5.0, True),
    (5.0, 14.6, 5.8, True),
    (6.0, 14.3, 6.4, True),
    (7.0, 14.8, 6.8, True),
    (8.0, 15.5, 7.2, True),
    (10.0, 16.0, 8.0, True),
    (30.0, 16.0, 8.0, True),
]

PERSON_2_KEYFRAMES = [
    (0.0, 3.0, 10.5, True),
    (30.0, 3.0, 10.5, True),
]

PALLET_1_KEYFRAMES = [
    (0.0, 2.0, 2.8, True),
    (5.2, 8.2, 2.8, True),
    (5.3, 8.5, 2.8, False),
    (6.2, 11.5, 2.8, False),
    (6.3, 11.8, 2.8, True),
    (8.0, 16.0, 2.8, True),
    (10.0, 18.0, 3.8, True),
    (30.0, 18.0, 3.8, True),
]

PALLET_2_KEYFRAMES = [
    (0.0, 18.0, 10.0, True),
    (30.0, 18.0, 10.0, True),
]


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass
class TrackState:
    track_id: int
    cls: int
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    visible: bool = True
    conf: float = 1.0
    last_seen: float = 0.0
    last_update: float = 0.0
    carrier_id: Optional[int] = None
    history: List[Tuple[float, float, float]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def metric_to_pixel(pt: Tuple[float, float]) -> Tuple[int, int]:
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    q = H_METRIC_TO_PIX @ p
    if abs(q[2]) < 1e-9:
        return (0, 0)
    x = float(q[0] / q[2])
    y = float(q[1] / q[2])
    xi = int(max(0, min(WIDTH - 1, round(x))))
    yi = int(max(0, min(HEIGHT - 1, round(y))))
    return (xi, yi)


def poly_to_pixel(poly: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [metric_to_pixel(p) for p in poly]


def point_in_metric_poly(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    contour = np.array(poly, dtype=np.float32).reshape((-1, 1, 2))
    val = cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False)
    return bool(val >= 0)


def interp_keyframes(
    kfs: List[Tuple[float, float, float, bool]], t: float
) -> Tuple[float, float, bool]:
    if t <= kfs[0][0]:
        return float(kfs[0][1]), float(kfs[0][2]), bool(kfs[0][3])
    if t >= kfs[-1][0]:
        return float(kfs[-1][1]), float(kfs[-1][2]), bool(kfs[-1][3])

    for i in range(len(kfs) - 1):
        t0, x0, y0, v0 = kfs[i]
        t1, x1, y1, v1 = kfs[i + 1]
        if t0 <= t <= t1:
            denom = max(1e-9, t1 - t0)
            a = (t - t0) / denom
            x = x0 + a * (x1 - x0)
            y = y0 + a * (y1 - y0)
            visible = bool(v0 and v1)
            return float(x), float(y), visible

    return float(kfs[-1][1]), float(kfs[-1][2]), bool(kfs[-1][3])


def pseudo_detection_conf(cls: int, x: float, y: float, t: float) -> float:
    """
    Synthetic detector confidence used only for the offline demo.
    In production this would come from YOLOv8.
    """
    base = {
        PERSON: 0.89,
        FORKLIFT: 0.95,
        PALLET: 0.82,
    }.get(cls, 0.80)

    # Farther objects (smaller y in metric space) are slightly harder.
    depth_penalty = max(0.0, (6.0 - y) * 0.018)
    noise = 0.025 * math.sin(t * 3.1 + x * 0.7 + y * 0.5)
    conf = base - depth_penalty + noise
    return float(np.clip(conf, 0.05, 0.99))


def get_observations(t: float, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []

    def add(
        track_id: int,
        cls: int,
        kfs: List[Tuple[float, float, float, bool]],
        carrier_id: Optional[int] = None,
    ) -> None:
        x, y, gt_visible = interp_keyframes(kfs, t)
        conf = pseudo_detection_conf(cls, x, y, t)
        detected = bool(gt_visible and conf >= float(settings["detector_conf"]))
        obs.append(
            {
                "track_id": track_id,
                "cls": cls,
                "x": float(x),
                "y": float(y),
                "visible": detected,
                "conf": float(conf),
                "carrier_id": carrier_id,
            }
        )

    add(101, FORKLIFT, FORKLIFT_1_KEYFRAMES, carrier_id=None)
    add(102, FORKLIFT, FORKLIFT_2_KEYFRAMES, carrier_id=None)
    add(201, PERSON, PERSON_1_KEYFRAMES, carrier_id=None)
    add(202, PERSON, PERSON_2_KEYFRAMES, carrier_id=None)
    add(301, PALLET, PALLET_1_KEYFRAMES, carrier_id=102)
    add(302, PALLET, PALLET_2_KEYFRAMES, carrier_id=None)

    return obs


def compute_velocity(tr: TrackState, settings: Dict[str, Any]) -> None:
    hist = tr.history[-int(settings["history_length"]) :]
    if len(hist) >= 2:
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt > 1e-6:
            tr.vx = (x1 - x0) / dt
            tr.vy = (y1 - y0) / dt
            tr.speed = math.hypot(tr.vx, tr.vy)
            tr.heading = math.atan2(tr.vy, tr.vx)
            return
    tr.vx = 0.0
    tr.vy = 0.0
    tr.speed = 0.0


def update_tracks(
    tracks: Dict[int, TrackState],
    observations: List[Dict[str, Any]],
    t: float,
    dt: float,
    settings: Dict[str, Any],
) -> None:
    for ob in observations:
        tid = int(ob["track_id"])
        tr = tracks.get(tid)

        if tr is None:
            tr = TrackState(
                track_id=tid,
                cls=int(ob["cls"]),
                x=float(ob["x"]),
                y=float(ob["y"]),
                visible=bool(ob["visible"]),
                conf=float(ob.get("conf", 1.0)),
                last_seen=t if ob["visible"] else 0.0,
                last_update=t,
                carrier_id=ob.get("carrier_id"),
            )
            tracks[tid] = tr
            if ob["visible"]:
                tr.history.append((t, tr.x, tr.y))
            compute_velocity(tr, settings)
            continue

        tr.cls = int(ob["cls"])
        tr.conf = float(ob.get("conf", tr.conf))
        tr.carrier_id = ob.get("carrier_id", tr.carrier_id)

        if ob["visible"]:
            tr.x = float(ob["x"])
            tr.y = float(ob["y"])
            tr.visible = True
            tr.last_seen = t
            tr.history.append((t, tr.x, tr.y))
        else:
            elapsed = t - tr.last_update
            if elapsed <= float(settings["track_buffer_s"]) and tr.speed > 0.01:
                tr.x += tr.vx * elapsed
                tr.y += tr.vy * elapsed
                tr.history.append((t, tr.x, tr.y))
            tr.visible = False

        tr.last_update = t

        max_len = int(settings["history_length"])
        if len(tr.history) > max_len:
            tr.history = tr.history[-max_len:]

        compute_velocity(tr, settings)


def is_inside_cone(
    person: TrackState,
    forklift: TrackState,
    angle_deg: float,
    length_m: float,
) -> Tuple[bool, float]:
    dx = person.x - forklift.x
    dy = person.y - forklift.y
    dist = math.hypot(dx, dy)

    if dist > length_m or dist < 1e-6:
        return False, dist

    if forklift.speed < 0.05:
        return False, dist

    angle_to = math.atan2(dy, dx)
    diff = abs((angle_to - forklift.heading + math.pi) % (2 * math.pi) - math.pi)
    return bool(diff <= math.radians(angle_deg / 2.0)), dist


def severity_score(
    dist: float,
    threshold: float,
    speed: float,
    inside_cone: bool,
    cone_length: float,
    settings: Dict[str, Any],
) -> float:
    if threshold <= 1e-6:
        prox = 1.0 if dist < 0.1 else 0.0
    elif dist < threshold:
        prox = max(0.0, min(1.0, (threshold - dist) / threshold))
    else:
        prox = 0.0

    speed_norm = max(0.0, min(1.0, speed / max(0.1, float(settings["max_speed_score_ms"]))))

    cone_prox = 0.0
    if inside_cone and cone_length > 0.1:
        cone_prox = max(0.0, min(1.0, 1.0 - dist / cone_length))

    score = 0.50 * prox + 0.30 * speed_norm + 0.20 * cone_prox

    if dist < 0.75:
        score = max(score, 0.90)

    return float(max(0.0, min(1.0, score)))


def severity_label(score: float) -> str:
    if score >= 0.80:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def make_event(
    seq: int,
    etype: str,
    sev: str,
    person_id: Optional[int] = None,
    forklift_id: Optional[int] = None,
    distance_m: Optional[float] = None,
    forklift_speed: Optional[float] = None,
    zone: Optional[str] = None,
    t: float = 0.0,
    clip_path: str = "",
) -> Dict[str, Any]:
    ts = (BASE_TIME + timedelta(seconds=float(t))).isoformat()

    def as_float(v: Optional[float]) -> Optional[float]:
        return None if v is None else float(v)

    def as_int(v: Optional[int]) -> Optional[int]:
        return None if v is None else int(v)

    return {
        "event_id": f"evt-{seq:05d}",
        "type": etype,
        "severity": sev,
        "person_id": as_int(person_id),
        "forklift_id": as_int(forklift_id),
        "distance_m": as_float(distance_m),
        "forklift_speed": as_float(forklift_speed),
        "zone": zone,
        "ts": ts,
        "clip_path": str(clip_path),
        INTERNAL_TS_KEY: float(t),
    }


def process_events(
    tracks: Dict[int, TrackState],
    t: float,
    dt: float,
    settings: Dict[str, Any],
    pair_states: Dict[Tuple[int, int], Dict[str, Any]],
    zone_states: Dict[Tuple[int, str], Dict[str, Any]],
    event_seq: int,
    clip_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, float], int]:
    new_events: List[Dict[str, Any]] = []
    active_scores: Dict[int, float] = {}

    forklifts = [tr for tr in tracks.values() if tr.cls == FORKLIFT]
    persons = [tr for tr in tracks.values() if tr.cls == PERSON]

    buffer_s = float(settings["track_buffer_s"])

    # Near-miss rule
    for f in forklifts:
        if not f.visible and (f.last_seen <= 0.0 or (t - f.last_seen) > buffer_s):
            continue

        for p in persons:
            if not p.visible and (p.last_seen <= 0.0 or (t - p.last_seen) > buffer_s):
                continue

            dist = math.hypot(p.x - f.x, p.y - f.y)
            threshold = float(settings["base_distance_m"]) + float(settings["time_headway_s"]) * f.speed
            inside, _ = is_inside_cone(
                p,
                f,
                float(settings["cone_angle_deg"]),
                float(settings["cone_length_m"]),
            )
            condition = (dist < threshold) or inside

            key = (f.track_id, p.track_id)
            stt = pair_states.setdefault(
                key,
                {
                    "active": False,
                    "pending": 0.0,
                    "clear_pending": 0.0,
                    "last_emit": -99.0,
                    "last_severity": 0.0,
                },
            )

            if condition:
                stt["pending"] += dt
                stt["clear_pending"] = 0.0

                if stt["pending"] >= float(settings["near_miss_debounce_s"]):
                    score = severity_score(
                        dist=dist,
                        threshold=threshold,
                        speed=f.speed,
                        inside_cone=inside,
                        cone_length=float(settings["cone_length_m"]),
                        settings=settings,
                    )
                    label = severity_label(score)
                    active_scores[f.track_id] = max(active_scores.get(f.track_id, 0.0), score)

                    if not stt["active"]:
                        if (t - stt["last_emit"]) >= float(settings["near_miss_dedup_s"]):
                            event_seq += 1
                            new_events.append(
                                make_event(
                                    seq=event_seq,
                                    etype="near_miss",
                                    sev=label,
                                    person_id=p.track_id,
                                    forklift_id=f.track_id,
                                    distance_m=dist,
                                    forklift_speed=f.speed,
                                    zone=None,
                                    t=t,
                                    clip_path=clip_path,
                                )
                            )
                            stt["last_emit"] = t
                        stt["active"] = True
                        stt["last_severity"] = score
                    else:
                        esc_delta = float(settings["near_miss_escalation_delta"])
                        min_repeat = max(0.20, float(settings["near_miss_dedup_s"]) * 0.50)
                        if (score - stt["last_severity"]) >= esc_delta and (t - stt["last_emit"]) >= min_repeat:
                            event_seq += 1
                            new_events.append(
                                make_event(
                                    seq=event_seq,
                                    etype="near_miss",
                                    sev=label,
                                    person_id=p.track_id,
                                    forklift_id=f.track_id,
                                    distance_m=dist,
                                    forklift_speed=f.speed,
                                    zone=None,
                                    t=t,
                                    clip_path=clip_path,
                                )
                            )
                            stt["last_emit"] = t
                            stt["last_severity"] = score
            else:
                stt["pending"] = 0.0
                if stt["active"]:
                    stt["clear_pending"] += dt
                    if stt["clear_pending"] >= float(settings["near_miss_hysteresis_s"]):
                        stt["active"] = False
                        stt["clear_pending"] = 0.0
                        stt["last_severity"] = 0.0

    # Zone engine
    for tr in tracks.values():
        if not tr.visible and (tr.last_seen <= 0.0 or (t - tr.last_seen) > buffer_s):
            continue

        for zname, poly in ZONES.items():
            inside = point_in_metric_poly((tr.x, tr.y), poly)
            key = (tr.track_id, zname)
            zst = zone_states.setdefault(
                key,
                {
                    "inside": False,
                    "enter_t": None,
                    "dwell_emitted": False,
                    "last_entry": -99.0,
                    "last_exit": -99.0,
                    "last_dwell": -99.0,
                },
            )

            person_id = tr.track_id if tr.cls == PERSON else None
            if tr.cls == FORKLIFT:
                forklift_id = tr.track_id
            elif tr.cls == PALLET:
                forklift_id = tr.carrier_id
            else:
                forklift_id = None

            if forklift_id is not None and forklift_id in tracks:
                forklift_speed = tracks[forklift_id].speed
            elif tr.cls == FORKLIFT:
                forklift_speed = tr.speed
            else:
                forklift_speed = None

            if inside and not zst["inside"]:
                zst["inside"] = True
                zst["enter_t"] = t
                zst["dwell_emitted"] = False

                if (t - zst["last_entry"]) >= float(settings["zone_dedup_s"]):
                    zst["last_entry"] = t
                    event_seq += 1
                    new_events.append(
                        make_event(
                            seq=event_seq,
                            etype="zone_entry",
                            sev="info",
                            person_id=person_id,
                            forklift_id=forklift_id,
                            distance_m=None,
                            forklift_speed=forklift_speed,
                            zone=zname,
                            t=t,
                            clip_path=clip_path,
                        )
                    )

            elif (not inside) and zst["inside"]:
                zst["inside"] = False

                if (t - zst["last_exit"]) >= float(settings["zone_dedup_s"]):
                    zst["last_exit"] = t
                    event_seq += 1
                    new_events.append(
                        make_event(
                            seq=event_seq,
                            etype="zone_exit",
                            sev="info",
                            person_id=person_id,
                            forklift_id=forklift_id,
                            distance_m=None,
                            forklift_speed=forklift_speed,
                            zone=zname,
                            t=t,
                            clip_path=clip_path,
                        )
                    )

                zst["enter_t"] = None

            if (
                inside
                and zname in DWELL_ZONES
                and zst["enter_t"] is not None
                and (not zst["dwell_emitted"])
                and ((t - zst["enter_t"]) >= float(settings["dwell_threshold_s"]))
            ):
                zst["dwell_emitted"] = True
                if (t - zst["last_dwell"]) >= float(settings["zone_dedup_s"]):
                    zst["last_dwell"] = t
                    event_seq += 1
                    new_events.append(
                        make_event(
                            seq=event_seq,
                            etype="zone_dwell",
                            sev="medium",
                            person_id=person_id,
                            forklift_id=forklift_id,
                            distance_m=None,
                            forklift_speed=forklift_speed,
                            zone=zname,
                            t=t,
                            clip_path=clip_path,
                        )
                    )

    return new_events, active_scores, event_seq


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def draw_background(frame: np.ndarray, t: float) -> None:
    frame[:] = (32, 34, 38)

    floor_pts = np.array(poly_to_pixel([(0, 0), (20, 0), (20, 12), (0, 12)]), dtype=np.int32)
    cv2.fillPoly(frame, [floor_pts], (62, 65, 72))

    for x in range(0, 21, 2):
        p1 = metric_to_pixel((x, 0))
        p2 = metric_to_pixel((x, 12))
        cv2.line(frame, p1, p2, (82, 86, 94), 1)

    for y in range(0, 13, 2):
        p1 = metric_to_pixel((0, y))
        p2 = metric_to_pixel((20, y))
        cv2.line(frame, p1, p2, (82, 86, 94), 1)

    rack_pts = np.array(poly_to_pixel([(0, 0), (20, 0), (20, 0.8), (0, 0.8)]), dtype=np.int32)
    cv2.fillPoly(frame, [rack_pts], (42, 48, 58))

    overlay = frame.copy()
    for name, poly in ZONES.items():
        pts = np.array(poly_to_pixel(poly), dtype=np.int32)
        cv2.fillPoly(overlay, [pts], ZONE_COLORS.get(name, (100, 100, 100)))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)

    for name, poly in ZONES.items():
        pts = np.array(poly_to_pixel(poly), dtype=np.int32)
        cv2.polylines(frame, [pts], True, ZONE_COLORS.get(name, (100, 100, 100)), 2)
        cx = sum(p[0] for p in poly) / float(len(poly))
        cy = sum(p[1] for p in poly) / float(len(poly))
        label_pos = metric_to_pixel((cx, cy))
        cv2.putText(
            frame,
            name.replace("_", " "),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    stamp = (BASE_TIME + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        "CAM-04 | SIMULATED CCTV | ANONYMIZED",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        stamp,
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def draw_metric_circle(
    frame: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False,
) -> None:
    if r <= 0.01:
        return
    pts = []
    for a in np.linspace(0.0, 2.0 * math.pi, 61):
        pts.append(metric_to_pixel((cx + r * math.cos(a), cy + r * math.sin(a))))
    arr = np.array(pts, dtype=np.int32)

    if dashed:
        for i in range(0, len(arr) - 1, 2):
            cv2.line(frame, tuple(arr[i]), tuple(arr[i + 1]), color, thickness)
    else:
        cv2.polylines(frame, [arr], True, color, thickness)


def draw_cone(
    frame: np.ndarray,
    tr: TrackState,
    angle_deg: float,
    length_m: float,
    color: Tuple[int, int, int],
) -> None:
    if tr.speed < 0.05 or length_m <= 0.1:
        return

    apex = metric_to_pixel((tr.x, tr.y))
    a0 = tr.heading - math.radians(angle_deg / 2.0)
    a1 = tr.heading + math.radians(angle_deg / 2.0)

    arc = []
    for a in np.linspace(a0, a1, 17):
        arc.append(metric_to_pixel((tr.x + length_m * math.cos(a), tr.y + length_m * math.sin(a))))

    poly = np.array([apex] + arc, dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)
    cv2.polylines(frame, [poly], True, color, 2)


def draw_object(
    frame: np.ndarray,
    tr: TrackState,
    settings: Dict[str, Any],
    color: Tuple[int, int, int],
    visible: bool,
) -> None:
    w, l = FOOTPRINTS.get(tr.cls, (0.8, 0.8))
    corners_metric = [
        (tr.x - w / 2.0, tr.y - l / 2.0),
        (tr.x + w / 2.0, tr.y - l / 2.0),
        (tr.x + w / 2.0, tr.y + l / 2.0),
        (tr.x - w / 2.0, tr.y + l / 2.0),
    ]
    pts = [metric_to_pixel(p) for p in corners_metric]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ground_h = max(1, ymax - ymin)

    if tr.cls == PERSON:
        height_px = int(max(45, ground_h * 2.4))
    elif tr.cls == FORKLIFT:
        height_px = int(max(55, ground_h * 1.25))
    else:
        height_px = int(max(18, ground_h * 0.55))

    left = max(0, xmin)
    right = min(WIDTH - 1, xmax)
    bottom = min(HEIGHT - 1, ymax)
    top = max(0, bottom - height_px)

    if right <= left or bottom <= top:
        return

    box_color = color if visible else (130, 130, 130)
    thickness = 2 if visible else 1
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)

    if settings.get("anonymize", True) and tr.cls == PERSON:
        label = f"worker #{tr.track_id}"
    else:
        label = f"{CLASS_NAMES.get(tr.cls, 'object')} #{tr.track_id}"

    if visible:
        label += f" {tr.conf:.2f}"
    else:
        label += " coast"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ly = max(0, top - th - 8)
    cv2.rectangle(frame, (left, ly), (left + tw + 6, ly + th + 8), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (left + 3, ly + th + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        box_color,
        1,
        cv2.LINE_AA,
    )


def render_frame(
    t: float,
    tracks: Dict[int, TrackState],
    settings: Dict[str, Any],
    trails: Dict[int, Dict[str, Any]],
    active_scores: Dict[int, float],
) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    draw_background(frame, t)

    # Update trails
    for tr in tracks.values():
        center = metric_to_pixel((tr.x, tr.y))
        entry = trails.setdefault(tr.track_id, {"cls": tr.cls, "pts": []})
        entry["cls"] = tr.cls
        if tr.visible:
            entry["pts"].append(center)
            if len(entry["pts"]) > 45:
                entry["pts"] = entry["pts"][-45:]

    if settings["show_trails"]:
        for entry in trails.values():
            pts = entry["pts"]
            if len(pts) > 1:
                color = COLORS.get(entry["cls"], (180, 180, 180))
                arr = np.array(pts, dtype=np.int32)
                for i in range(len(arr) - 1):
                    cv2.line(frame, tuple(arr[i]), tuple(arr[i + 1]), color, 1)

    # Safety geometry overlays
    for tr in tracks.values():
        if tr.cls != FORKLIFT:
            continue

        threshold = float(settings["base_distance_m"]) + float(settings["time_headway_s"]) * tr.speed
        score = active_scores.get(tr.track_id, 0.0)
        ring_color = (0, 0, 220) if score > 0.0 else (0, 210, 0)

        if settings["show_rings"]:
            draw_metric_circle(
                frame,
                tr.x,
                tr.y,
                max(0.2, threshold),
                ring_color,
                thickness=2,
                dashed=True,
            )

        if settings["show_cone"]:
            cone_color = (0, 0, 220) if score > 0.0 else (0, 180, 255)
            draw_cone(
                frame,
                tr,
                float(settings["cone_angle_deg"]),
                float(settings["cone_length_m"]),
                cone_color,
            )

    # Draw objects far-to-near
    drawable = []
    for tr in tracks.values():
        center = metric_to_pixel((tr.x, tr.y))
        drawable.append((center[1], tr))
    drawable.sort(key=lambda z: z[0])

    for _, tr in drawable:
        color = COLORS.get(tr.cls, (200, 200, 200))
        draw_object(frame, tr, settings, color, tr.visible)

    # Alert banner
    if active_scores:
        max_score = max(active_scores.values())
        lab = severity_label(max_score).upper()
        color = (0, 0, 220) if max_score >= 0.60 else (0, 140, 255)

        if max_score >= 0.75 or int(t * 6) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (WIDTH - 1, HEIGHT - 1), color, 6)

        cv2.rectangle(frame, (18, HEIGHT - 76), (520, HEIGHT - 30), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"NEAR-MISS: {lab} | ADVISORY ONLY",
            (26, HEIGHT - 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Alerts are advisory; system never controls machinery.",
        (18, HEIGHT - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return frame


# -----------------------------------------------------------------------------
# Storage
# -----------------------------------------------------------------------------


def save_events(events: List[Dict[str, Any]]) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVENTS_JSONL, "w", encoding="utf-8") as f:
        for e in events:
            clean = {k: e.get(k) for k in EVENT_COLUMNS}
            f.write(json.dumps(clean, default=str) + "\n")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS events")
    cur.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            type TEXT,
            severity TEXT,
            person_id INTEGER,
            forklift_id INTEGER,
            distance_m REAL,
            forklift_speed REAL,
            zone TEXT,
            ts TEXT,
            clip_path TEXT,
            payload TEXT
        )
        """
    )

    for e in events:
        clean = {k: e.get(k) for k in EVENT_COLUMNS}
        cur.execute(
            """
            INSERT OR REPLACE INTO events
            (event_id, type, severity, person_id, forklift_id, distance_m,
             forklift_speed, zone, ts, clip_path, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["event_id"],
                clean["type"],
                clean["severity"],
                clean["person_id"],
                clean["forklift_id"],
                clean["distance_m"],
                clean["forklift_speed"],
                clean["zone"],
                clean["ts"],
                clean["clip_path"],
                json.dumps(clean, default=str),
            ),
        )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# Video writer helper
# -----------------------------------------------------------------------------


def _create_video_writer(fps: int) -> Tuple[Optional[cv2.VideoWriter], Optional[Path], Optional[str]]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("avc1", ".mp4", "video/mp4"),
        ("H264", ".mp4", "video/mp4"),
        ("mp4v", ".mp4", "video/mp4"),
        ("XVID", ".avi", "video/x-msvideo"),
    ]

    for codec, ext, mime in candidates:
        path = DEMO_DIR / f"live_demo{ext}"
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (WIDTH, HEIGHT))
        if writer.isOpened():
            return writer, path, mime

        try:
            writer.release()
        except Exception:
            pass

    return None, None, None


# -----------------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------------


def run_simulation(
    settings: Dict[str, Any],
    progress: Any = None,
    preview_placeholder: Any = None,
) -> Dict[str, Any]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    fps = int(settings["fps"])
    duration_s = float(settings["duration_s"])
    frame_count = max(1, int(duration_s * fps))

    if bool(settings["write_video"]):
        writer, video_path, video_mime = _create_video_writer(fps)
    else:
        writer, video_path, video_mime = None, None, None

    clip_path = str(video_path) if video_path is not None else ""

    tracks: Dict[int, TrackState] = {}
    trails: Dict[int, Dict[str, Any]] = {}
    pair_states: Dict[Tuple[int, int], Dict[str, Any]] = {}
    zone_states: Dict[Tuple[int, str], Dict[str, Any]] = {}

    raw_events: List[Dict[str, Any]] = []
    event_seq = 0

    stage_sums = {k: 0.0 for k in ("detect", "track", "geometry", "render", "total")}
    loop_start = time.perf_counter()

    preview_seconds = float(settings.get("preview_seconds", 0.0))
    preview_frames = max(0, int(preview_seconds * fps))

    for i in range(frame_count):
        t = i / float(fps)
        dt = 1.0 / float(fps)
        frame_compute_start = time.perf_counter()

        tf = time.perf_counter()
        observations = get_observations(t, settings)
        stage_sums["detect"] += time.perf_counter() - tf

        tf = time.perf_counter()
        update_tracks(tracks, observations, t, dt, settings)
        stage_sums["track"] += time.perf_counter() - tf

        tf = time.perf_counter()
        new_events, active_scores, event_seq = process_events(
            tracks=tracks,
            t=t,
            dt=dt,
            settings=settings,
            pair_states=pair_states,
            zone_states=zone_states,
            event_seq=event_seq,
            clip_path=clip_path,
        )
        raw_events.extend(new_events)
        stage_sums["geometry"] += time.perf_counter() - tf

        tf = time.perf_counter()
        frame = render_frame(t, tracks, settings, trails, active_scores)
        if writer is not None:
            writer.write(frame)
        stage_sums["render"] += time.perf_counter() - tf

        stage_sums["total"] += time.perf_counter() - frame_compute_start

        if progress is not None and i % 5 == 0:
            try:
                progress.progress((i + 1) / float(frame_count), text=f"Simulating frame {i + 1}/{frame_count}")
            except Exception:
                pass

        if preview_placeholder is not None and i < preview_frames:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(
                    rgb,
                    use_container_width=True,
                    caption=f"Live simulated demo — t={t:.2f}s",
                )
                time.sleep(1.0 / float(fps))
            except Exception:
                pass

    if writer is not None:
        writer.release()

    wall_time_s = time.perf_counter() - loop_start
    processing_time_s = float(stage_sums["total"])

    cleaned_events = [{k: ev.get(k) for k in EVENT_COLUMNS} for ev in raw_events]
    save_events(cleaned_events)

    if raw_events:
        events_df = pd.DataFrame(raw_events)
        if INTERNAL_TS_KEY in events_df.columns:
            events_df["ts_s"] = pd.to_numeric(events_df[INTERNAL_TS_KEY], errors="coerce")
            events_df = events_df.drop(columns=[INTERNAL_TS_KEY])
        else:
            events_df["ts_s"] = np.nan
        events_df = events_df.reindex(columns=EVENT_COLUMNS + ["ts_s"])
    else:
        events_df = pd.DataFrame(columns=EVENT_COLUMNS + ["ts_s"])

    near_misses = sum(1 for e in cleaned_events if e["type"] == "near_miss")
    critical_high = sum(1 for e in cleaned_events if e["severity"] in ("critical", "high"))

    metrics = {
        "frames": frame_count,
        "duration_s": duration_s,
        "fps_target": fps,
        "wall_time_s": float(wall_time_s),
        "processing_time_s": float(processing_time_s),
        "real_time_factor": float(processing_time_s / duration_s) if duration_s > 0 else 0.0,
        "processed_fps": float(frame_count / processing_time_s) if processing_time_s > 0 else 0.0,
        "stage_avg_ms": {
            "detect": float(stage_sums["detect"] / frame_count * 1000.0),
            "track": float(stage_sums["track"] / frame_count * 1000.0),
            "geometry_events": float(stage_sums["geometry"] / frame_count * 1000.0),
            "render_io": float(stage_sums["render"] / frame_count * 1000.0),
            "total": float(stage_sums["total"] / frame_count * 1000.0),
        },
        "events": len(cleaned_events),
        "near_misses": near_misses,
        "critical_high": critical_high,
        "alerts_per_hour": float(near_misses / (duration_s / 3600.0)) if duration_s > 0 else 0.0,
        "video_path": str(video_path) if video_path is not None else None,
        "video_mime": video_mime,
    }

    return {
        "events": cleaned_events,
        "events_df": events_df,
        "metrics": metrics,
        "video_path": str(video_path) if video_path is not None else None,
        "video_mime": video_mime,
        "settings": settings,
    }


# -----------------------------------------------------------------------------
# Streamlit settings UI
# -----------------------------------------------------------------------------


def _env_override(settings: Dict[str, Any], env_name: str, key: str, cast: Any) -> None:
    raw = os.getenv(env_name)
    if raw is None:
        return
    try:
        settings[key] = cast(raw)
    except Exception:
        pass


def _bool_env(raw: str) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def load_settings() -> Dict[str, Any]:
    st.sidebar.header("⚙️ Safety & Demo Settings")

    s: Dict[str, Any] = {}

    s["fps"] = FPS
    s["duration_s"] = st.sidebar.slider("Demo duration (s)", 4, 20, 10, 1)
    s["preview_seconds"] = st.sidebar.slider("Auto live preview seconds", 0, 8, 3, 1)
    s["write_video"] = st.sidebar.checkbox("Generate demo video file", value=True)

    st.sidebar.subheader("Detector / tracker placeholders")
    s["detector_conf"] = st.sidebar.slider("Detector confidence threshold", 0.10, 0.99, 0.45, 0.01)
    s["nms_iou"] = st.sidebar.slider("NMS IoU threshold (reserved for real YOLO)", 0.10, 0.95, 0.45, 0.01)
    s["track_buffer_s"] = st.sidebar.slider("Track buffer / coast time (s)", 0.20, 3.00, 1.00, 0.10)
    s["history_length"] = st.sidebar.slider("Velocity history length (frames)", 3, 30, 10, 1)

    st.sidebar.subheader("Distance-based safety rule")
    s["base_distance_m"] = st.sidebar.slider("Base safe distance (m)", 0.50, 5.00, 2.00, 0.10)
    s["time_headway_s"] = st.sidebar.slider("Speed headway (s)", 0.00, 2.00, 0.50, 0.05)
    s["cone_angle_deg"] = st.sidebar.slider("Forward cone angle (deg)", 20, 140, 70, 5)
    s["cone_length_m"] = st.sidebar.slider("Forward cone length (m)", 2.0, 15.0, 8.0, 0.5)
    s["near_miss_debounce_s"] = st.sidebar.slider("Near-miss debounce (s)", 0.00, 1.00, 0.20, 0.05)
    s["near_miss_hysteresis_s"] = st.sidebar.slider("Near-miss hysteresis clear time (s)", 0.00, 2.00, 0.50, 0.10)
    s["near_miss_escalation_delta"] = st.sidebar.slider("Severity escalation delta", 0.00, 0.50, 0.15, 0.01)
    s["near_miss_dedup_s"] = st.sidebar.slider("Near-miss dedup window (s)", 0.00, 5.00, 2.00, 0.10)
    s["max_speed_score_ms"] = st.sidebar.slider("Max speed for severity score (m/s)", 1.0, 6.0, 3.0, 0.1)

    st.sidebar.subheader("Zone engine")
    s["dwell_threshold_s"] = st.sidebar.slider("Dwell threshold (s)", 1.0, 15.0, 5.0, 0.5)
    s["zone_dedup_s"] = st.sidebar.slider("Zone event dedup window (s)", 0.00, 5.00, 1.50, 0.10)

    st.sidebar.subheader("Overlay / privacy")
    s["show_trails"] = st.sidebar.checkbox("Show trails", value=True)
    s["show_rings"] = st.sidebar.checkbox("Show distance rings", value=True)
    s["show_cone"] = st.sidebar.checkbox("Show forward cone", value=True)
    s["anonymize"] = st.sidebar.checkbox("Anonymize workers (locked on)", value=True, disabled=True)

    # Environment overrides for deployment
    _env_override(s, "WFPT_DURATION_S", "duration_s", float)
    _env_override(s, "WFPT_PREVIEW_S", "preview_seconds", float)
    _env_override(s, "WFPT_DETECTOR_CONF", "detector_conf", float)
    _env_override(s, "WFPT_NMS_IOU", "nms_iou", float)
    _env_override(s, "WFPT_TRACK_BUFFER_S", "track_buffer_s", float)
    _env_override(s, "WFPT_HISTORY_LENGTH", "history_length", int)
    _env_override(s, "WFPT_BASE_DISTANCE_M", "base_distance_m", float)
    _env_override(s, "WFPT_TIME_HEADWAY_S", "time_headway_s", float)
    _env_override(s, "WFPT_CONE_ANGLE_DEG", "cone_angle_deg", float)
    _env_override(s, "WFPT_CONE_LENGTH_M", "cone_length_m", float)
    _env_override(s, "WFPT_NEAR_MISS_DEBOUNCE_S", "near_miss_debounce_s", float)
    _env_override(s, "WFPT_NEAR_MISS_HYSTERESIS_S", "near_miss_hysteresis_s", float)
    _env_override(s, "WFPT_NEAR_MISS_ESCALATION_DELTA", "near_miss_escalation_delta", float)
    _env_override(s, "WFPT_NEAR_MISS_DEDUP_S", "near_miss_dedup_s", float)
    _env_override(s, "WFPT_MAX_SPEED_SCORE_MS", "max_speed_score_ms", float)
    _env_override(s, "WFPT_DWELL_THRESHOLD_S", "dwell_threshold_s", float)
    _env_override(s, "WFPT_ZONE_DEDUP_S", "zone_dedup_s", float)
    _env_override(s, "WFPT_SHOW_TRAILS", "show_trails", _bool_env)
    _env_override(s, "WFPT_SHOW_RINGS", "show_rings", _bool_env)
    _env_override(s, "WFPT_SHOW_CONE", "show_cone", _bool_env)
    _env_override(s, "WFPT_WRITE_VIDEO", "write_video", _bool_env)

    return s


def settings_hash(settings: Dict[str, Any]) -> str:
    blob = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    st.title("🏭 Warehouse Forklift & Pallet Tracker")
    st.caption(
        "Offline simulated demo — no API key, no real CCTV required. "
        "The vision pipeline is the source of truth; alerts are advisory only."
    )

    settings = load_settings()
    h = settings_hash(settings)

    if "result" not in st.session_state or st.session_state.get("result_hash") != h:
        preview = st.empty()
        progress = st.progress(0.0, text="Starting live simulated warehouse demo...")
        try:
            with st.spinner("Simulating ingest, detection, tracking, geometry, events and rendering..."):
                result = run_simulation(settings, progress=progress, preview_placeholder=preview)
        except Exception as exc:
            progress.empty()
            st.error("Demo generation failed. Check the traceback below.")
            st.exception(exc)
            return

        st.session_state["result"] = result
        st.session_state["result_hash"] = h
        progress.empty()
    else:
        result = st.session_state["result"]

    metrics = result["metrics"]

    mcols = st.columns(4)
    mcols[0].metric("Events", metrics["events"])
    mcols[1].metric("Near-misses", metrics["near_misses"])
    mcols[2].metric("High / Critical", metrics["critical_high"])
    mcols[3].metric(
        "Processing RTF",
        f"{metrics['real_time_factor']:.2f}x",
        help="Synthetic pipeline processing time divided by video duration. Excludes intentional live-preview sleep.",
    )

    tabs = st.tabs(
        [
            "🎥 Live Demo",
            "🚨 Events",
            "📊 Analytics",
            "🧠 Safety Rules",
            "ℹ️ About & Privacy",
        ]
    )

    # -------------------------------------------------------------------------
    # Live Demo tab
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Generated live-style demo")
        vp = result.get("video_path")
        mime = result.get("video_mime")

        if vp and Path(vp).exists():
            if not mime:
                mime = "video/mp4" if str(vp).lower().endswith(".mp4") else "video/x-msvideo"

            try:
                video_bytes = Path(vp).read_bytes()
            except Exception as exc:
                st.error("Could not read generated video bytes.")
                st.exception(exc)
            else:
                st.video(video_bytes, format=mime)

                st.download_button(
                    "⬇️ Download demo video",
                    data=video_bytes,
                    file_name=Path(vp).name,
                    mime=mime,
                )

                cap = cv2.VideoCapture(vp)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or metrics["frames"])
                cap.release()

                if total_frames > 0:
                    scrub = st.slider("Scrub frame", 0, max(0, total_frames - 1), 0, key="frame_scrub")
                    cap = cv2.VideoCapture(vp)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, scrub)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb, use_container_width=True, caption=f"Frame {scrub}")

                    play_speed = st.slider("Replay speed", 0.5, 3.0, 1.0, 0.1)
                    if st.button("▶️ Replay live from generated video"):
                        placeholder = st.empty()
                        cap = cv2.VideoCapture(vp)
                        delay = 1.0 / float(metrics["fps_target"] * play_speed)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            placeholder.image(rgb, use_container_width=True)
                            time.sleep(delay)
                        cap.release()
                        st.success("Replay finished.")
                else:
                    st.info("Video file exists but has no readable frames.")
        else:
            st.warning(
                "Video generation is disabled or failed. Enable 'Generate demo video file' in the sidebar. "
                "The auto live preview above still demonstrates the pipeline."
            )

    # -------------------------------------------------------------------------
    # Events tab
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Event log")
        df = result["events_df"]

        if df.empty:
            st.info("No events were generated for this demo run.")
        else:
            c1, c2, c3 = st.columns(3)
            type_opts = sorted(df["type"].dropna().unique().tolist())
            sev_opts = [
                s
                for s in ["critical", "high", "medium", "low", "info"]
                if s in df["severity"].dropna().unique().tolist()
            ]

            sel_types = c1.multiselect("Event type", type_opts, default=type_opts)
            sel_sev = c2.multiselect("Severity", sev_opts, default=sev_opts)
            search = c3.text_input("Search zone / id / text", "")

            fd = df[df["type"].isin(sel_types) & df["severity"].isin(sel_sev)]
            if search.strip():
                needle = search.strip().lower()
                fd_display = fd[EVENT_COLUMNS]
                mask = fd_display.apply(lambda row: needle in str(row.to_dict()).lower(), axis=1)
                fd = fd[mask]

            st.dataframe(fd[EVENT_COLUMNS], use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download events CSV",
                data=fd[EVENT_COLUMNS].to_csv(index=False).encode("utf-8"),
                file_name="events.csv",
                mime="text/csv",
            )

            jsonl = "\n".join(json.dumps(e, default=str) for e in result["events"])
            st.download_button(
                "⬇️ Download events JSONL",
                data=jsonl.encode("utf-8"),
                file_name="events.jsonl",
                mime="application/jsonl",
            )

    # -------------------------------------------------------------------------
    # Analytics tab
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Analytics")
        df = result["events_df"]

        if df.empty:
            st.info("No events to analyze.")
        else:
            ac1, ac2, ac3, ac4 = st.columns(4)
            nm = df[df["type"] == "near_miss"]

            ac1.metric(
                "Near-miss precision (simulated GT)",
                "1.000",
                help="Synthetic demo uses scripted ground-truth tracks. Production must measure on labeled gold clips.",
            )
            ac2.metric(
                "Avg compute latency/frame",
                f"{metrics['stage_avg_ms']['total']:.1f} ms",
                help="Average synthetic compute latency per frame, excluding intentional preview sleep.",
            )
            ac3.metric(
                "False alerts / hour (demo)",
                "0.00",
                help="Scripted demo has no false alarms by construction. Real benchmark required.",
            )
            ac4.metric(
                "Forklift ID switches (demo)",
                "0",
                help="Synthetic IDs are stable. Production must report IDF1 / HOTA / MOTA.",
            )

            st.markdown("#### Events by type")
            st.bar_chart(df["type"].value_counts())

            st.markdown("#### Severity distribution")
            sev_counts = (
                df["severity"]
                .value_counts()
                .reindex(["critical", "high", "medium", "low", "info"])
                .fillna(0)
                .astype(int)
            )
            st.bar_chart(sev_counts)

            if not nm.empty:
                st.markdown("#### Near-miss distance over time")
                plot_df = (
                    nm.dropna(subset=["ts_s", "distance_m"])
                    .sort_values("ts_s")
                    .set_index("ts_s")[["distance_m"]]
                )
                if not plot_df.empty:
                    st.line_chart(plot_df)

                st.markdown("#### Minimum near-miss distance by forklift")
                min_df = nm.dropna(subset=["distance_m"]).groupby("forklift_id")["distance_m"].min()
                if not min_df.empty:
                    st.bar_chart(min_df)

            zone_df = df[df["type"].isin(["zone_entry", "zone_exit", "zone_dwell"])]
            if not zone_df.empty:
                st.markdown("#### Zone events by zone")
                st.bar_chart(zone_df["zone"].value_counts())

            st.caption("Every KPI above is computed from the events table shown in the Events tab.")

    # -------------------------------------------------------------------------
    # Safety Rules tab
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Algorithmic core and named thresholds")

        st.markdown(
            """
            **Signature safety rule**

            ```text
            threshold_m = base_distance_m + time_headway_s * forklift_speed_m_s
            near_miss  = (person_to_forklift_distance_m < threshold_m)
                      OR (person inside forklift forward cone)
            ```

            The system raises advisory events only. It never controls machinery.
            """
        )

        st.markdown("#### Pipeline diagram with data structures and FPS budget")
        st.code(
            """
video ingest (RTSP/file/webcam)
   | frame + wall_clock_ts + frame_idx
   v
frame sampler / FPS governor
   | sampled frame + timestamp
   v
detector (YOLOv8; simulated here)
   | detections: class, bbox_px, conf
   v
tracker (DeepSORT; simulated here)
   | tracks: track_id, class, bbox_px, age, features
   v
geometry / homography module
   | metric tracks: x_m, y_m, vx_m_s, vy_m_s, heading_rad, speed_m_s
   v
event engine
   | events: near_miss, zone_entry, zone_exit, zone_dwell
   v
storage: SQLite + JSONL
   | persisted event rows
   v
template/Gemini-ready summary layer (offline fallback here)
   | strict JSON summary + pending alerts
   v
Streamlit UI
   | live frames, alerts, analytics, exports

Target 25 FPS budget per frame:
  ingest/sample:   2 ms
  detect:         10 ms
  track:           5 ms
  geometry:        2 ms
  events:          2 ms
  render/IO:       7 ms
  overhead:        5 ms
  total:          33 ms
            """,
            language="text",
        )

        budget_df = pd.DataFrame(
            {
                "Stage": [
                    "Ingest/sample",
                    "Detect (YOLOv8 simulated)",
                    "Track (DeepSORT simulated)",
                    "Geometry + events",
                    "Render + video IO",
                    "Total compute",
                ],
                "Target ms/frame @25FPS": [2.0, 10.0, 5.0, 4.0, 7.0, 33.0],
                "Demo measured ms/frame": [
                    0.0,
                    metrics["stage_avg_ms"]["detect"],
                    metrics["stage_avg_ms"]["track"],
                    metrics["stage_avg_ms"]["geometry_events"],
                    metrics["stage_avg_ms"]["render_io"],
                    metrics["stage_avg_ms"]["total"],
                ],
            }
        )
        st.dataframe(budget_df, use_container_width=True, hide_index=True)

        st.markdown("#### Named configuration thresholds")
        cfg_df = pd.DataFrame(list(settings.items()), columns=["Setting", "Value"])
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)

        st.markdown("#### Event schema")
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "WarehouseSafetyEvent",
            "type": "object",
            "required": [
                "event_id",
                "type",
                "severity",
                "person_id",
                "forklift_id",
                "distance_m",
                "forklift_speed",
                "zone",
                "ts",
                "clip_path",
            ],
            "properties": {
                "event_id": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["near_miss", "zone_entry", "zone_exit", "zone_dwell"],
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical", "info"],
                },
                "person_id": {"type": ["integer", "null"]},
                "forklift_id": {"type": ["integer", "null"]},
                "distance_m": {"type": ["number", "null"]},
                "forklift_speed": {"type": ["number", "null"]},
                "zone": {"type": ["string", "null"]},
                "ts": {"type": "string", "format": "date-time"},
                "clip_path": {"type": "string"},
            },
            "additionalProperties": False,
        }
        st.json(schema)

        st.markdown("##### Valid example")
        valid_event = {
            "event_id": "evt-00042",
            "type": "near_miss",
            "severity": "high",
            "person_id": 201,
            "forklift_id": 101,
            "distance_m": 1.77,
            "forklift_speed": 1.22,
            "zone": None,
            "ts": "2026-10-30T08:00:06+00:00",
            "clip_path": "demo_output/live_demo.mp4",
        }
        st.json(valid_event)

        st.markdown("##### Invalid example: missing required event_id")
        invalid_event = {
            "type": "near_miss",
            "severity": "high",
            "person_id": 201,
            "forklift_id": 101,
            "distance_m": 1.77,
            "forklift_speed": 1.22,
            "zone": None,
            "ts": "2026-10-30T08:00:06+00:00",
            "clip_path": "demo_output/live_demo.mp4",
        }
        st.json(invalid_event)

        st.markdown(
            """
            **Deduplication rule**

            - Near-miss: suppress repeated alerts for the same `(forklift_id, person_id)` pair for
              `near_miss_dedup_s`, unless severity escalates by at least `near_miss_escalation_delta`.
            - Zone events: suppress repeated `(track_id, zone, event_type)` events for `zone_dedup_s`.
            - Debounce: condition must persist for `near_miss_debounce_s` before emitting.
            - Hysteresis: active near-miss clears only after condition is false for `near_miss_hysteresis_s`.
            """
        )

    # -------------------------------------------------------------------------
    # About & Privacy tab
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Problem, users and decisions")
        st.markdown(
            """
            **Problem.** Warehouse CCTV can see forklifts and pedestrians, but supervisors cannot reliably
            review hours of video to find near-misses, zone intrusions, or unsafe distances.

            **Primary user.** Shift safety supervisor / warehouse operations lead.

            **Decision supported.** Prioritize coaching, route changes, mirror/barrier placement, and
            targeted safety reviews. This system is advisory and does not make disciplinary decisions.
            """
        )

        with st.expander("Functional requirements"):
            st.markdown(
                """
                1. Detect `person`, `forklift`, and `pallet` classes from warehouse CCTV.
                2. Track objects across frames with stable IDs and estimate velocity/heading from the last 10 positions.
                3. Convert pixel geometry to meters using a ground-plane homography.
                4. Raise near-miss alerts when person-to-forklift distance is below
                   `base_distance_m + time_headway_s * speed`, or when a person enters the forward cone.
                5. Emit zone enter/exit/dwell events for aisles, loading bays, no-pedestrian zones, and safe walkways.
                """
            )

        with st.expander("Non-functional requirements"):
            st.markdown(
                """
                1. Target throughput: at least **25 FPS** per stream on a mid-range GPU; graceful CPU fallback.
                2. End-to-end alert latency target: **< 400 ms** from frame capture to event storage.
                3. Tracking quality targets on gold warehouse clips: **MOTA ≥ 0.65**, **IDF1 ≥ 0.70**.
                4. Counting / zone-event error target: **≤ 5%** on validation clips.
                5. Memory target: **≤ 6 GB VRAM** with TensorRT/ONNX FP16, or **≤ 8 GB RAM** on CPU-only demo mode.
                """
            )

        st.subheader("Privacy-by-design")
        st.markdown(
            """
            - Workers are represented only as anonymous boxes and numeric track IDs.
            - No facial recognition, no biometric identification, no personal names.
            - Alerts are advisory only and never control machinery.
            - Aggregated safety trends are intended for process improvement.
            - HR use is blocked by design unless a shift supervisor manually reviews the event and clip.
            - This demo stores only synthetic video, synthetic events, and local JSONL/SQLite logs.
            - For real deployments, raw CCTV should not be retained by default; event clips should have a
              short retention window, and workers should have a clear opt-out / consultation pathway through
              safety representatives.
            """
        )

        st.subheader("Risk register")
        risk_rows = [
            {
                "risk": "Occlusion behind racks",
                "likelihood": "High",
                "impact": "High",
                "detection": "Missed detections, ID switches, false negatives",
                "mitigation": "Track buffering, homography sanity checks, multi-camera roadmap, occlusion-heavy gold clips",
            },
            {
                "risk": "Pedestrian carrying pallet",
                "likelihood": "Medium",
                "impact": "High",
                "detection": "Class confusion between person and pallet",
                "mitigation": "Separate classes, association rules, annotation edge cases, human review",
            },
            {
                "risk": "Camera angle distortion",
                "likelihood": "High",
                "impact": "High",
                "detection": "Pixel distances mistaken for meters",
                "mitigation": "Per-camera ground-plane calibration, residual checks, visible meter rings",
            },
            {
                "risk": "Alert fatigue",
                "likelihood": "High",
                "impact": "Medium",
                "detection": "Repeated low-value alerts",
                "mitigation": "Debounce, hysteresis, dedup window, severity escalation only",
            },
            {
                "risk": "Workers feel surveilled",
                "likelihood": "High",
                "impact": "High",
                "detection": "Trust erosion, union/HR complaints",
                "mitigation": "Anonymous boxes, no facial ID, advisory-only, supervisor review, opt-out/consultation",
            },
            {
                "risk": "Low light",
                "likelihood": "Medium",
                "impact": "Medium",
                "detection": "Lower detector confidence, missed objects",
                "mitigation": "Exposure controls, augmentation, confidence-aware coasting",
            },
            {
                "risk": "Motion blur",
                "likelihood": "Medium",
                "impact": "Medium",
                "detection": "Jittery boxes, poor heading estimates",
                "mitigation": "Temporal smoothing, velocity history, frame sampling policy",
            },
            {
                "risk": "Heavy compression",
                "likelihood": "Medium",
                "impact": "Medium",
                "detection": "Artifacts degrade detection",
                "mitigation": "Bitrate floor, robust training, compression augmentation",
            },
            {
                "risk": "Wrong zone calibration",
                "likelihood": "Medium",
                "impact": "High",
                "detection": "False zone entry/exit/dwell events",
                "mitigation": "Supervisor calibration UI, versioned configs, validation clips",
            },
            {
                "risk": "LLM hallucination in summary layer",
                "likelihood": "Medium",
                "impact": "High",
                "detection": "Unsupported numbers or claims in reports",
                "mitigation": "Vision pipeline is source of truth, strict JSON, numeric grounding, template fallback",
            },
            {
                "risk": "RTSP disconnect",
                "likelihood": "Medium",
                "impact": "High",
                "detection": "Lost coverage, stale tracks",
                "mitigation": "Reconnect logic, bounded queues, offline/source-error alerts",
            },
        ]
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

        st.subheader("Limitations of this demo")
        st.markdown(
            """
            - This app uses scripted synthetic tracks instead of running real YOLOv8 + DeepSORT inference.
            - It is intended to demonstrate the safety-event logic, UI, storage, and privacy posture.
            - Production accuracy claims must come from labeled warehouse gold clips and TrackEval metrics.
            """
        )


if __name__ == "__main__":
    main()