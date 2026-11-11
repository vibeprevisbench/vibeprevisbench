#!/usr/bin/env python3
"""VLM-based Screen Check: does the proxy video show what the 4D scene got right?

    python3 judge/screen_check.py --data data --run runs/<system> --scores results/<system> --out results/<system>/screen

For each task with a delivered proxy video the judge (Gemini 4 Argon in the paper) gets two turns:
  1. blind: only the proxy video (photo tasks: also the evaluator's orbit video). It writes a recap of what it sees:
     places, objects, and for videos who does what when.
  2. it then gets the brief, the reference media and four checklists built from the hidden ground truth facts, and
     answers every entry with yes / partial / no / n/a:
       V1 shape and place identity  (named in the blind recap = yes)
       V2 action readability        (events; an event whose moment is not visible in V3 is left out of V2)
       V3 visibility at key moments (every event, and every shot without one)
       V4 story                     (story beats, judged against the blind recap only)
     Photo tasks are judged on V1 only.
Only entries the Rule-based Scene Check found correct are asked (an entry it has no check for is asked too): the
Screen Check measures whether a correct scene can be seen, not whether it is correct. A dimension's share is the mean
of yes = 1, partial = 0.5, no = 0 over its answered entries; the task's Screen score is 100 x the mean share of the
dimensions that have entries.

--scores is the output of `vpbench.py score` (scorer/score.py) for the same run: the per-check report of each task decides which entries
are asked, and scores.csv says which tasks were scored. Results are written per task to --out/<task_id>.json; a task
that already has one is skipped, so a stopped run can be started again.

The judge is any endpoint that speaks the Gemini generateContent API:
  VPB_JUDGE_API_KEY    API key (or GEMINI_API_KEY)
  VPB_JUDGE_BASE_URL   default https://generativelanguage.googleapis.com/v1beta
  VPB_JUDGE_MODEL      default gemini-4-argon
  VPB_JUDGE_AUTH       "goog" (x-goog-api-key header, default) or "bearer" (Authorization: Bearer)
Photo tasks: the judge sees the photo frames of the proxy video (the first frame_count frames of the pinned brief,
rubrics/<ID>/task.json) and the evaluator's orbit, <run>/<ID>/orbit/orbit.mp4 (vpbench.py orbit, orbit/render_orbit.py),
as in the paper; the orbit the 228-frame brief asks the agent for is not judged.
Videos are re-encoded to 2 fps, at most 640 px wide (photos at most 1024 px) and sent inline; ffmpeg and ffprobe must be
on PATH.
"""
import argparse, base64, csv, hashlib, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

FPS = 2.0
JUDGE_MODEL = os.environ.get("VPB_JUDGE_MODEL", "gemini-4-argon")
BASE_URL = os.environ.get("VPB_JUDGE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
AUTH = os.environ.get("VPB_JUDGE_AUTH", "goog")
MEDIA = Path("screen_cache")   # set by main(): re-encoded copies of the videos and photos sent to the judge

HEAD = ("You are watching a video rendered from a simplified 3D previsualization (simple shapes, mannequin-like "
        "characters, at most flat colours; this look is intended). Timestamps are relative to the start of the video.")
P1 = HEAD + """
Describe ONLY what you can see; do not invent a story.
1. RECAP: in time order, who is where, what each person does, to whom / with what. Refer to people by what they
   look like (e.g. "the shorter figure", "figure on the left"). Name objects by what their SHAPE looks like; if a
   shape is not recognisable write "unidentified shape". If a movement is not clear write "unclear movement".
2. PLACES: what kind of place(s) this looks like, or "unclear".
3. OBJECTS: every object / character you can identify from its SHAPE (category names only, e.g. "sofa", "car",
   "person"); do not list shapes you cannot identify.
Output JSON {"recap":[{"t":"<seconds or range>","text":"..."}],"places":["..."],"objects":["..."]}.
Keep it under 300 words."""
P1_IMG = HEAD + """ The first video is the model's own camera view(s); the second (if any) is an evaluation-side orbit
of the same scene (a 360-degree look from the room centre, then a top-down orbit with the ceiling hidden).
Describe ONLY what you can see.
1. PLACES: what kind of place(s) this looks like, or "unclear".
2. OBJECTS: every object you can identify from its SHAPE (category names only, e.g. "sofa", "sink", "bookshelf");
   do not list shapes you cannot identify.
Output JSON {"places":["..."],"objects":["..."]}."""

RULES = """Automatic checks already verify object presence, positions, layout, camera placement, timing, shot count and
physics — DO NOT judge those, and DO NOT judge materials, lighting, realism, aesthetics or composition. Judge the RENDER
only{refnote}. For every checklist entry answer "yes", "partial", "no" or "n/a" (n/a = that entity / actor / shot does not
appear in the render at all; that is penalised elsewhere) and give the render timestamp(s) you used.
Be strict: when unsure between two verdicts, choose the lower one."""
DIM = {
 "V1": """V1 SHAPE & PLACE IDENTITY (KEY ENTITIES, PLACES): was it recognised from its shape BEFORE you saw the brief?
   yes = YOUR previous-turn PLACES / OBJECTS / RECAP already named that category or kind of place (same category counts,
   e.g. armchair = sofa; any "person"/"figure" counts for people); partial = it was not named blind, but now that you
   know what it should be, a shape at a plausible spot fits it; no = it reads as something else / cannot be found.""",
 "V2": """V2 ACTION READABILITY (EVENTS): would a viewer who has NOT read the brief describe the render footage as that action /
   interaction? yes = action and contact are clear; partial = pose changes but the action or contact is unclear;
   no = miming in the air, standing still, or a different action. Ignore exact timing, order and body-part details.""",
 "V3": """V3 VISIBILITY AT KEY MOMENTS (MOMENTS): at that moment of the render (times are the reference's; the render should
   be at about the same time), are the people / objects involved clearly visible — the bodies and the contact area?
   Judge visibility only, not whether the action is right. yes = clearly visible; partial = visible but small, partly
   blocked or partly out of frame; no = mostly blocked by foreground geometry, out of frame, too small to see, the frame
   is black / white / only a wall, the camera is inside geometry, flicker or broken faces, leftover helper objects or
   text.""",
 "V4": """V4 STORY (STORY BEATS): does YOUR RECAP from the previous turn contain this beat (same who / what, wording may
   differ; a beat lists several actions — yes = the recap has its gist, partial = only part of it)? yes / partial / no.
   Judge only against the recap text you already wrote; do not use the brief or the videos to fill gaps."""}
WORDS = {"turn": "turns around", "open_door": "opens the door of", "close_door": "closes the door of", "sit": "sits down",
         "crouch": "crouches down", "contact": "touches / holds", "hand_over": "hands over", "carry": "carries",
         "walk": "walks", "run": "runs", "stand_up": "stands up", "pick_up": "picks up", "put_down": "puts down",
         "lie_down": "lies down", "jump": "jumps", "look_at": "looks at", "talk": "talks with", "enter": "enters",
         "exit": "leaves", "land": "lands", "stop": "stops", "start": "starts moving", "lie": "lies down", "fall": "falls", "kneel": "kneels", "push": "pushes", "wave": "waves"}


class Retryable(RuntimeError):
    pass


def first(paths):
    return next((p for p in paths if p.exists()), None)


def video_part(path, fps):
    data = Path(path).read_bytes()
    return {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(data).decode()},
            "videoMetadata": {"fps": round(float(fps), 3)}}


def image_part(path):
    return {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(Path(path).read_bytes()).decode()}}


def small(path, kind):
    """a 2 fps, at most 640 px wide, low-bitrate copy of a video (the judge samples at 2 fps anyway), photos at most
    1024 px wide; cached by source path, size and mtime"""
    st = path.stat()
    h = hashlib.sha1(("%s|%d|%d" % (path, st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]
    dst = MEDIA / (h + (".mp4" if kind == "video" else ".jpg"))
    if not dst.exists():
        MEDIA.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp%d%s" % (threading.get_ident(), dst.suffix))  # several threads may share a reference
        vf = ["-vf", "fps=%g,scale='min(640,iw)':-2" % FPS, "-an", "-c:v", "libx264", "-crf", "28", "-preset", "fast",
              "-pix_fmt", "yuv420p"] if kind == "video" else ["-vf", "scale='min(1024,iw)':-2", "-q:v", "4"]
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path)] + vf + [str(tmp)], check=True)
        tmp.replace(dst)
    return video_part(dst, FPS) if kind == "video" else image_part(dst)


def checklists(g, kind, rep):
    """the V1-V4 entries of one task from its hidden ground truth facts (gt.json) and the rule report; an entry is
    judged when the rule scorer passed it or has no check for it"""
    ents = {e["id"]: e for e in g.get("entities") or []}
    objs = {o["id"]: o for o in g.get("objects") or [] if o.get("id")}

    def lab(i):
        if i in ents:
            return "%s (%s)" % (i, ents[i]["name"].split(",")[0].rstrip("."))
        if i in objs:
            return "%s (%s)" % (i, objs[i]["class"].replace("other:", ""))
        return i

    def judged(key, ok):  # option A
        r = rep.get(key)
        return True if r is None else ok(r)

    L = {"V1": [], "V2": [], "V3": [], "V4": []}
    for e in g.get("entities") or []:
        L["V1"].append({"id": e["id"], "text": "%s %s: %s" % (e["id"], e["class"], e["name"]),
                        "judge": judged("entity:" + e["id"], lambda r: bool(r.get("value")))})
    seen = set()
    for o in g.get("objects") or []:
        cls = o["class"].replace("other:", "")
        if cls in seen or len(seen) >= 12 or cls in ("wall", "floor", "ceiling", "ground", "room", "door_frame"):
            continue
        seen.add(cls)
        r = [x for n, x in rep.items() if x["dim"] == "A1" and x["block"] == "counts" and n.endswith(":" + o["class"])]
        L["V1"].append({"id": "obj:" + cls, "text": "a " + cls.replace("_", " "),
                        "judge": (max(x["score"] for x in r) > 0) if r else True})
    for s in g.get("spaces") or []:
        L["V1"].append({"id": "place:" + s["id"], "text": "%s: %s" % (s["id"], s.get("name")), "judge": True})
    evs = []
    for e in g.get("events") or []:
        a, ob = e.get("agents") or [], e.get("objects") or []
        if e["type"] == "hand_over" and len(a) == 2:
            txt = "%s hands %s over to %s" % (lab(a[0]), " / ".join(map(lab, ob)) or "something", lab(a[1]))
        elif e["type"] == "contact" and len(a) == 2:
            txt = "%s and %s touch / hold each other" % (lab(a[0]), lab(a[1]))
        else:
            txt = ("%s %s %s" % (" and ".join(map(lab, a)), WORDS.get(e["type"], e["type"].replace("_", " ")),
                                 " / ".join(map(lab, ob)))).strip()
        ok = judged("event:%s:%s" % (e["id"], e["type"]), lambda r: r["score"] > 0)
        evs.append((e, txt, ok))
        L["V2"].append({"id": e["id"], "text": "%s (reference %.1f-%.1f s)" % (txt, e["t"][0], e["t"][1]), "judge": ok})
    shots = g.get("shots") or []
    if kind != "image":
        for e, txt, ok in evs:
            L["V3"].append({"id": "M:" + e["id"], "text": "%.1f-%.1f s: %s" % (e["t"][0], e["t"][1], txt), "judge": ok})
        for s in shots:
            if any(s["t"][0] <= e["t"][0] < s["t"][1] for e, _, _ in evs):
                continue
            subj = ", ".join(map(lab, s.get("subjects") or [])) or "the shot's subject"
            L["V3"].append({"id": "M:" + s["id"], "text": "%.1f-%.1f s (shot %s): %s" % (s["t"][0], s["t"][1], s["id"], subj),
                            "judge": True})
    if evs and kind != "image":
        n = len(shots) or 1
        chunks = [shots[round(i * n / min(5, n)):round((i + 1) * n / min(5, n))] for i in range(min(5, n))] if shots else [[]]
        for k, ch in enumerate(chunks):
            t0, t1 = (ch[0]["t"][0], ch[-1]["t"][1]) if ch else (0, 1e9)
            inside = [(e, txt, ok) for e, txt, ok in evs if t0 <= e["t"][0] < t1 or (k == len(chunks) - 1 and e["t"][0] >= t1)]
            done = [txt for e, txt, ok in inside if ok]
            if inside:
                L["V4"].append({"id": "B%d" % (k + 1), "text": "; ".join(done) if done else "(none)", "judge": bool(done)})
    return L


def photo_frames(video, n):
    """photo tasks: the judge sees the photo cameras only (the first n frames, n = frame_count of the pinned brief) and
    the evaluator's orbit; a delivery that also renders the orbit of the 228-frame brief is cut to its photo frames"""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
    if int(r.stdout.strip().split(",")[0]) <= n:
        return video
    st = video.stat()
    h = hashlib.sha1(("%s|%d|%d|%d" % (video, st.st_size, int(st.st_mtime), n)).encode()).hexdigest()[:16]
    dst = MEDIA / ("photos_" + h + ".mp4")
    if not dst.exists():
        MEDIA.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp%d.mp4" % threading.get_ident())
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                        "trim=end_frame=%d,setpts=PTS-STARTPTS" % n, "-an", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(tmp)], check=True)
        tmp.replace(dst)
    return dst


def item_kind(kind):
    """index.json kind (script, spatial_arkit, spatial_keyframe, continuation, video) -> checklist kind"""
    return "image" if kind.startswith("spatial") else kind


# ------------------------------------------------------------------------------------------------- judge API
def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            return json.loads(t[i:j + 1])
        raise


def api_key():
    k = os.environ.get("VPB_JUDGE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not k:
        sys.exit("set VPB_JUDGE_API_KEY (or GEMINI_API_KEY) to the judge's API key")
    return k


def gemini(contents):
    url = "%s/models/%s:generateContent" % (BASE_URL, requests.utils.quote(JUDGE_MODEL, safe=""))
    payload = {"contents": contents, "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 32768,
                                                          "thinkingConfig": {"thinkingLevel": "LOW"}}}
    head = {"Content-Type": "application/json"}
    head.update({"Authorization": "Bearer " + api_key()} if AUTH == "bearer" else {"x-goog-api-key": api_key()})
    err = None
    for k in range(6):
        try:
            r = requests.post(url, json=payload, timeout=(20, 900), headers=head)
            if r.status_code == 429 or r.status_code >= 500:
                raise Retryable("HTTP %d %s" % (r.status_code, r.text[:120]))
            if r.status_code != 200:
                raise RuntimeError("HTTP %d %s" % (r.status_code, r.text[:300]))
            data = r.json()
            if data.get("error"):
                raise Retryable("upstream error %s" % str(data["error"])[:200])
            cand = data["candidates"][0]
            parts = [p for p in cand["content"]["parts"] if not p.get("thought")]
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip() or not cand.get("finishReason"):
                raise Retryable("empty / unfinished")
            return parts, parse_json(text), data.get("usageMetadata") or {}
        except (Retryable, requests.RequestException, ValueError, KeyError) as e:
            err = e
            print("  retry", k + 1, type(e).__name__, str(e)[:120].replace("\n", " "), flush=True)
            time.sleep(15 * (k + 1))
    raise RuntimeError("gave up: %s" % err)


# ------------------------------------------------------------------------------------------------- scoring
def band(share):
    return 4 if share >= .9 else 3 if share >= .7 else 2 if share >= .5 else 1 if share >= .2 else 0


def nid(i):
    """some judges drop the "M:" prefix of V3 moment ids (E03 for M:E03): compare without it"""
    i = str(i).strip()
    return i[2:] if i.startswith("M:") else i


def score_dims(dims, lists, verdicts):
    val = {"yes": 1.0, "partial": .5, "no": 0.0}
    out = {}
    v3 = {nid(v.get("id")): str(v.get("v", "")).lower() for v in verdicts.get("V3") or []}
    for k in dims:
        asked = [nid(x["id"]) for x in lists[k] if x["judge"]]
        if k == "V2":  # readability given visible: drop events whose moment is not visible (counted in V3 already)
            asked = [i for i in asked if v3.get(i) != "no"]
        got = {nid(v.get("id")): str(v.get("v", "")).lower() for v in verdicts.get(k) or []}
        vs = [val[got[i]] for i in asked if got.get(i) in val]
        out[k] = {"asked": len(asked), "scored": len(vs), "na_by_judge": sum(1 for i in asked if got.get(i) == "n/a"),
                  "missing": sum(1 for i in asked if i not in got), "na_by_rule": sum(1 for x in lists[k] if not x["judge"]),
                  "share": (sum(vs) / len(vs)) if vs else None, "band": band(sum(vs) / len(vs)) if vs else None}
    return out


def screen_score(dims):
    shares = [d["share"] for d in dims.values() if d["share"] is not None]
    return (sum(shares) / len(shares) * 100) if shares else None


# ------------------------------------------------------------------------------------------------- one task
def build_request(item, kind, gt, task, rep, ref_media):
    """the checklists and the turn-2 text of one task (no API call)"""
    lists = checklists(gt, kind, rep or {})
    ref_parts = [small(p, "video") if p.suffix == ".mp4" else small(p, "image") for p in ref_media]
    dims = ["V1"] if kind == "image" else ["V1", "V2", "V3", "V4"]
    fmt = lambda k, keep=lambda x: True: "\n".join("%s: %s" % (x["id"], x["text"]) for x in lists[k]
                                                   if x["judge"] and keep(x)) or "(none)"
    isplace = lambda x: x["id"].startswith("place:")
    blocks = {"V1": "KEY ENTITIES:\n" + fmt("V1", lambda x: not isplace(x)) + "\nPLACES:\n" + fmt("V1", isplace),
              "V2": "EVENTS:\n" + fmt("V2"),
              "V3": "MOMENTS:\n" + fmt("V3"),
              "V4": "STORY BEATS:\n" + fmt("V4")}
    refnote = "; the reference media only shows what each entry means" if ref_parts else ""
    body = (RULES.format(refnote=refnote) + "\n\n" + "\n".join(DIM[d] for d in dims) + "\n\nTASK BRIEF:\n" + task["prompt"]
            + "\n\n" + "\n".join(blocks[d] for d in dims) + "\n\nOutput JSON {" +
            ",".join('"%s":[{"id":"..","v":"yes|partial|no|n/a","t":"..","note":"<=15 words"}]' % d for d in dims) +
            "} with exactly one entry per checklist id.")
    return lists, ref_parts, dims, body


def judge_task(item, kind, gt, task, rep, render, orbit, ref_media, rule_total, call=None):
    """two turns (blind recap -> verdicts); returns the task's record. call(contents) -> (parts, json, usage) is the
    judge (default gemini)"""
    call = call or gemini
    lists, ref_parts, dims, body = build_request(item, kind, gt, task, rep, ref_media)
    render = small(render, "video")
    usage = []
    if kind == "image":
        vids = [render] + ([small(orbit, "video")] if orbit is not None and orbit.exists() else [])
        t1 = [{"role": "user", "parts": vids + [{"text": P1_IMG}]}]
        parts1, recap, u1 = call(t1)
        lead = "Now you also get the task brief, the reference photo(s) (this turn's first part) and the checklist.\n"
    else:
        t1 = [{"role": "user", "parts": [render, {"text": P1}]}]
        parts1, recap, u1 = call(t1)
        lead = ("Now you also get the task brief" + (", the REFERENCE media it was meant to rebuild (after this turn's "
                                                     "first part)" if ref_parts else "") + " and checklists.\n")
    t2 = t1 + [{"role": "model", "parts": parts1}, {"role": "user", "parts": ref_parts + [{"text": lead + body}]}]
    _, verdicts, u2 = call(t2)
    usage += [u1, u2]
    res_dims = score_dims(dims, lists, verdicts)
    return {"task_id": item, "kind": kind, "rule_total": rule_total, "screen": screen_score(res_dims), "dims": res_dims,
            "recap": recap, "verdicts": verdicts, "lists": lists, "usage": usage, "fps": FPS, "judge": JUDGE_MODEL}


# ------------------------------------------------------------------------------------------------- run
def rule_rows(scores):
    """scores.csv of `vpbench.py score`: {task_id: row}"""
    with open(Path(scores) / "scores.csv", newline="", encoding="utf-8") as f:
        return {r["task_id"]: r for r in csv.DictReader(f)}


def rule_report(scores, item):
    """the per-check report the rule scorer wrote for this task: {check name: check}"""
    hits = sorted(Path(scores).glob("engine/reports_5011/*/%s/report.json" % item))
    if not hits:
        return None
    return {x["name"]: x for x in json.loads(hits[0].read_text())["items"]}


def main():
    global MEDIA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"), help="the dataset folder (tasks/, rubrics/)")
    ap.add_argument("--run", required=True, type=Path, help="<task_id>/video.mp4 (+ orbit/orbit.mp4 for photo tasks)")
    ap.add_argument("--scores", required=True, type=Path, help="output of vpbench.py score for this run")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", help="comma-separated task ids; default every scored task")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    MEDIA = a.out / "_media"
    index = {t["task_id"]: t for t in json.loads((a.data / "tasks/index.json").read_text())["tasks"]}
    rows = rule_rows(a.scores)
    want = set(a.tasks.split(",")) if a.tasks else None
    if want and not want <= set(index):
        ap.error("unknown tasks: " + ",".join(sorted(want - set(index))))
    todo = [i for i in sorted(index) if (want is None or i in want) and (rows.get(i) or {}).get("status") == "scored"]
    a.out.mkdir(parents=True, exist_ok=True)
    api_key()

    def one(item):
        f = a.out / (item + ".json")
        if f.exists():
            return json.loads(f.read_text())
        t = index[item]
        rb = a.data / "rubrics" / item
        task = json.loads((rb / "task.json").read_text())     # the brief the agent was given
        media = [a.data / "tasks" / item / m["path"] for m in task.get("media") or []]
        missing = [str(p) for p in media if not p.exists()]
        if missing:
            raise FileNotFoundError("input media missing (run fetch_external.py): " + ", ".join(missing))
        kind = item_kind(t["type"])
        video = a.run / item / "video.mp4"
        if kind == "image":
            if not (a.run / item / "orbit/orbit.mp4").is_file():
                raise FileNotFoundError("evaluation orbit missing: run vpbench.py orbit before judging photo tasks")
            video = photo_frames(video, task["output_spec"]["frame_count"])
        res = judge_task(item, kind, json.loads((rb / "gt.json").read_text()), task, rule_report(a.scores, item), video,
                         a.run / item / "orbit/orbit.mp4", media, float(rows[item]["rule"]))
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        print("%s %-12s rule %5.1f screen %5s" % (item, res["kind"], res["rule_total"],
              "%.1f" % res["screen"] if res["screen"] is not None else "-"), flush=True)
        return res

    def safe(item):
        try:
            return one(item)
        except Exception as e:
            print("FAILED", item, type(e).__name__, str(e)[:300], flush=True)
            return None
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(safe, todo))
    print("judged %d / %d tasks" % (sum(r is not None for r in res), len(todo)), flush=True)
    return 0 if all(r is not None for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
