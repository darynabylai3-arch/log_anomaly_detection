import re
import sys
from datetime import datetime

import pandas as pd

src = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

LINE = re.compile(
    r"^(?P<d>\d{8})-(?P<h>\d+):(?P<m>\d+):(?P<s>\d+):(?P<ms>\d+)\|"
    r"(?P<component>[^|]*)\|(?P<pid>\d+)\|(?P<msg>.*)$"
)

rows, bad = [], []
with open(src, encoding="utf-8", errors="replace") as f:
    for n, line in enumerate(f, 1):
        line = line.rstrip("\n")
        m = LINE.match(line)
        if not m:
            bad.append((n, line))
            continue
        g = m.groupdict()
        ts = datetime.strptime(g["d"], "%Y%m%d").replace(
            hour=int(g["h"]), minute=int(g["m"]), second=int(g["s"]),
            microsecond=int(g["ms"]) * 1000,
        )
        rows.append({"line_no": n, "ts": ts, "component": g["component"],
                     "pid": int(g["pid"]), "msg": g["msg"].strip()})

df = pd.DataFrame(rows)

# ---- тип события + числа, вытащенные из сообщения ----------------------
RULES = [
    ("step_changed", re.compile(r"^onStandStepChanged (\d+)$"), ["stand_steps"]),
    ("extend", re.compile(r"^onExtend:(\d+) (\d+) (\d+) (\d+)$"),
     ["ext_ts_ms", "ext_steps", "ext_b", "ext_state"]),
    ("report", re.compile(r"^REPORT : (\d+) (\d+) (\d+) (\d+)$"),
     ["rep_steps", "rep_distance", "rep_calories", "rep_altitude"]),
    ("get_detail", re.compile(r"^getTodayTotalDetailSteps = (\d+)##(\d+)##(\d+)##(\d+)##(\d+)##(\d+)$"),
     ["gd_ts_ms", "gd_steps", "gd_b", "gd_c", "gd_d", "gd_counter"]),
    ("set_detail", re.compile(r"^setTodayTotalDetailSteps=(\d+)##(\d+)##(\d+)##(\d+)##(\d+)##(\d+)$"),
     ["sd_ts_ms", "sd_steps", "sd_b", "sd_c", "sd_d", "sd_counter"]),
    ("calories", re.compile(r"^calculateCaloriesWithCache totalCalories=(\d+)$"), ["total_calories"]),
    ("altitude", re.compile(r"^calculateAltitudeWithCache totalAltitude=(\d+)$"), ["total_altitude"]),
    ("screen_receiver", re.compile(r"^onReceive action: android\.intent\.action\.(\w+)$"), ["action"]),
    ("broadcast", re.compile(r"^processHandleBroadcastAction action:android\.intent\.action\.(\w+)$"), ["action"]),
    ("db_upload", re.compile(r"^Alarm uploadStaticsToDB totalSteps=(\d+)Calories:(\d+)Floor:(\d+)Distance:(\d+)$"),
     ["up_steps", "up_calories", "up_floor", "up_distance"]),
    ("minute_upload", re.compile(r"^upLoadOneMinuteDataToEngine time=(\d+),(\d+),(\d+),(\d+),(\d+)$"),
     ["mu_minute", "mu_a", "mu_steps", "mu_c", "mu_type"]),
    ("stat_update", re.compile(r"^new date =(\d+), type=(\d+),([\d.]+),old=([\d.]+)$"),
     ["st_date", "st_type", "st_new", "st_old"]),
]

event = []
extracted = []
for msg in df["msg"]:
    ev, vals = "other", {}
    for name, rx, cols in RULES:
        mm = rx.match(msg)
        if mm:
            ev = name
            vals = dict(zip(cols, mm.groups()))
            break
    event.append(ev)
    extracted.append(vals)

df["event"] = event
feat = pd.DataFrame(extracted)
for c in feat.columns:
    if c != "action":
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
df = pd.concat([df, feat], axis=1)

# ---- полезные производные колонки ---------------------------------------
df["is_error"] = df["msg"].str.contains(r"fail|ERR_", case=False, regex=True)
df["date"] = df["ts"].dt.date
df["hour"] = df["ts"].dt.hour
df["minute_ts"] = df["ts"].dt.floor("min")
df["dt_prev_s"] = df["ts"].diff().dt.total_seconds()
df["screen_on_event"] = (df["action"] == "SCREEN_ON").astype(int)
df["screen_off_event"] = (df["action"] == "SCREEN_OFF").astype(int)

df.to_csv(f"{out_dir}/events.csv", index=False)

# ---- таблица по минутам для модели ---------------------------------------
rep = df[df.event == "report"]
g = df.groupby("minute_ts")
minute = pd.DataFrame({
    "n_events": g.size(),
    "n_step_changed": g.apply(lambda x: (x.event == "step_changed").sum()),
    "n_errors": g["is_error"].sum().astype(int),
    "screen_on": g["screen_on_event"].sum(),
    "screen_off": g["screen_off_event"].sum(),
    "ext_steps_sum": g["ext_steps"].sum(min_count=1),
    "stand_steps_max": g["stand_steps"].max(),
})
r = rep.groupby("minute_ts").agg(
    rep_steps=("rep_steps", "last"), rep_distance=("rep_distance", "last"),
    rep_calories=("rep_calories", "last"), rep_altitude=("rep_altitude", "last"),
)
minute = minute.join(r)
full = pd.date_range(minute.index.min(), minute.index.max(), freq="min")
minute = minute.reindex(full)
minute.index.name = "minute_ts"
for c in ["n_events", "n_step_changed", "n_errors", "screen_on", "screen_off"]:
    minute[c] = minute[c].fillna(0).astype(int)
minute["ext_steps_sum"] = minute["ext_steps_sum"].fillna(0)
for c in ["rep_steps", "rep_distance", "rep_calories", "rep_altitude", "stand_steps_max"]:
    minute[c] = minute[c].ffill()
minute["steps_delta"] = minute["rep_steps"].diff().fillna(0).clip(lower=0)
minute.reset_index().to_csv(f"{out_dir}/minute_features.csv", index=False)

# ---- краткая сводка ------------------------------------------------------
print("rows parsed:", len(df), "| unparsed lines:", len(bad))
print(df["event"].value_counts().to_string())
print("range:", df.ts.min(), "->", df.ts.max())
print("minute table:", minute.shape)