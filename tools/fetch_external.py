#!/usr/bin/env python3
"""Fetch the input media of VibePrevisBench (photos, first frames, reference videos) from their original sources.

The dataset is released without its photos and videos (third-party copyright). external.json lists, for
every media file, where the original comes from (a web page / file URL, a dataset file or a member of a dataset
archive) and the exact recipe that produced it (time window in the original, crop as fractions of the frame, output
size, fps, encoder settings, frame choice). This script downloads the originals, applies the recipe with ffmpeg,
writes the files to tasks/<task_id>/media/<name>, and checks each result against the sha256 and the content
fingerprints stored in the manifest (perceptual dHash, no image content). Each file is reported as
    identical   byte-identical to the file used in the benchmark (same sha256)
    ok          same content: fingerprints match at the recorded time and size (bytes differ after re-encoding)
    aligned     same content after the best offset (and zoom) for a shifted / slightly rescaled upload
    lowres      same content, but the source served a lower resolution than the one used in the benchmark
    mismatch    produced, but the content or the source file differs from the one used in the benchmark
    unavailable the original could not be downloaded (removed, region-locked, network)
Dataset photos are checked twice: the downloaded source file must have the recorded sha256 before it is processed.
Requirements: python3, ffmpeg + ffprobe (>= 5.1) on PATH, yt-dlp (CLI, recent) for YouTube / Bilibili, and
numpy + opencv-python(-headless) for the photos: their steps are replayed with OpenCV as they were run, giving
byte-identical photos (without OpenCV they are rebuilt with ffmpeg, same content, and fisheye ones are skipped).

    python3 fetch_external.py                       fetch everything that is missing (verified files are skipped)
    python3 fetch_external.py --check               only report what is missing (exit code 1 if anything is)
    python3 fetch_external.py --tasks B250_190,M50_038 --kinds video,continuation --jobs 4 --cache fetch_cache
            [--datasets-root /data]

See docs/source-reconstruction.md for every file, its source and how it is checked.

Network: the script uses the proxy settings of your own environment (HTTP(S)_PROXY for downloads; pass
--ytdlp-arg=--proxy=URL to yt-dlp if needed). Privacy: yt-dlp always runs with --ignore-config --no-cookies
--no-cookies-from-browser --no-cache-dir and a fresh empty cache directory; the script never logs in, never reads
browser data, and refuses cookie options.
"""
import argparse, bisect, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.parse
import urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STEP = 0.5            # fingerprint sampling step (s)
T_OK = 7.0            # mean Hamming distance (of 64 bits) accepted as the same content
T_IMG = 9.0           # single image (one 64-bit hash) threshold
T256_VID = 24.0       # final check: mean Hamming distance of the 256-bit track (17x16 dHash) per 0.5 s
T256_IMG = 28.0       # final check: one 256-bit hash of an image
LOWRES = 0.95         # a source below 95% of the recorded source height counts as lower resolution
ZOOMS = (1.0, 0.97, 1.03, 0.94, 1.06)
UA = "Mozilla/5.0 (X11; Linux x86_64) vibeprevisbench-fetch/1.0"
YTDLP_FMT = "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]/bv*/b"
_locks, _locks_guard = {}, threading.Lock()


# ----------------------------------------------------------------------------------------------- small helpers
def log(*a):
    print(*a, file=sys.stderr, flush=True)


def lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def run(cmd, timeout=None, **kw):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)


def slug(s, n=60):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:n] + "_" + hashlib.sha1(s.encode()).hexdigest()[:8]


def ham(a, b):
    return bin(a ^ b).count("1")


def hx(h):
    return "%016x" % h


def unhx(s):
    return int(s, 16)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hdr_args(headers):
    if not headers:
        return []
    return ["-headers", "".join("%s: %s\r\n" % kv for kv in headers.items())]


# ----------------------------------------------------------------------------------------------- probing / hashing
def probe(src, headers=None):
    r = run(["ffprobe", "-v", "error"] + hdr_args(headers) + ["-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,avg_frame_rate:stream_side_data=rotation:format=duration,start_time",
             "-of", "json", str(src)], timeout=600)
    j = json.loads(r.stdout or b"{}")
    if not j.get("streams"):
        raise RuntimeError("ffprobe: no video stream in %s: %s" % (src, r.stderr.decode(errors="replace")[-300:]))
    s = j["streams"][0]
    fs = s.get("r_frame_rate") or "0/1"
    if s.get("avg_frame_rate") not in (None, "0/0") and fs in ("0/0", "1000/1", "90000/1"):
        fs = s["avg_frame_rate"]
    a, b = fs.split("/")
    fps = float(a) / float(b) if float(b) else 0.0
    f = j.get("format") or {}
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "fps_str": fs,
            "duration": float(f.get("duration") or 0), "start": float(f.get("start_time") or 0)}


def dhash72(b):
    h = 0
    for r in range(8):
        row = b[r * 9:(r + 1) * 9]
        for c in range(8):
            h = (h << 1) | (1 if row[c] > row[c + 1] else 0)
    return h


_START = {}


def start_time(src, headers=None):
    if src not in _START:
        _START[src] = probe(src, headers)["start"]
    return _START[src]


def frame_hashes(src, ss=None, dur=None, pre=None, headers=None, inopts=(), n_max=None, size=(9, 8)):
    """[(t, dhash64)] for every decoded frame of src (from input seek ss, reading dur s). t = the frame's true
    presentation time minus the file's start_time (-copyts), i.e. the same file timeline ffmpeg -ss uses, and
    independent of how a codec snaps the seek. pre: filters before the 9x8 area downscale (crop / rotation)."""
    vf = ",".join([x for x in (pre, "scale=%d:%d:flags=area,format=gray,showinfo" % size) if x])
    nb = size[0] * size[1]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "info", "-copyts"] + list(inopts) + hdr_args(headers)
    if ss is not None:
        cmd += ["-ss", "%.4f" % max(0.0, ss)]
    if dur is not None:
        cmd += ["-t", "%.4f" % dur]
    cmd += ["-i", str(src)]
    if n_max:
        cmd += ["-frames:v", str(n_max)]
    cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo",
            "-pix_fmt", "gray", "pipe:1"]
    r = run(cmd, timeout=7200)
    data = r.stdout
    pts = [float(m.group(1)) for m in re.finditer(rb"Parsed_showinfo.*? pts_time:\s*(-?[0-9.]+)", r.stderr)]
    n = min(len(pts), len(data) // nb)
    if n == 0:
        raise RuntimeError("decode failed (%s): %s" % (src, r.stderr.decode(errors="replace")[-400:]))
    st = start_time(str(src), headers)
    return [(pts[i] - st, dhash_wh(data[i * nb:(i + 1) * nb], size[0] - 1, size[1])) for i in range(n)]


def seek_args(rec, ss, length, headers=None, src=None):
    """the original pipeline's own seek, shifted to the downloaded copy: input -ss (accurate seek: the first frame
    is the first one at/after ss, as the original encode had), read length s. 'input_to' = bench250 media.py
    (-ss a -to b before -i); 'output_t' = miniset50 trim (-ss s -i SRC -t len)"""
    if rec.get("seek_mode") == "output_t":
        return hdr_args(headers) + ["-ss", "%.4f" % max(0.0, ss), "-i", str(src)], ["-t", "%.4f" % length]
    return hdr_args(headers) + ["-ss", "%.4f" % max(0.0, ss), "-t", "%.4f" % length, "-i", str(src)], []


def dhash_wh(b, w, h):
    """dHash of an (w+1) x h grey thumbnail: bit = cell brighter than its right neighbour; w*h bits"""
    v = 0
    for r in range(h):
        row = b[r * (w + 1):(r + 1) * (w + 1)]
        for c in range(w):
            v = (v << 1) | (1 if row[c] > row[c + 1] else 0)
    return v


def strong_track(path):
    """256-bit dHash (17x16 area thumbnail) of the frame shown every STEP s from the first frame, for a whole file"""
    r = run(["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "info", "-i", str(path), "-map", "0:v:0", "-an",
             "-vf", "scale=17:16:flags=area,format=gray,showinfo", "-fps_mode", "passthrough", "-f", "rawvideo",
             "-pix_fmt", "gray", "pipe:1"], timeout=7200)
    pts = [float(m.group(1)) for m in re.finditer(rb"Parsed_showinfo.*? pts_time:\s*(-?[0-9.]+)", r.stderr)]
    n = min(len(pts), len(r.stdout) // 272)
    if n == 0:
        raise RuntimeError("decode failed (%s)" % path)
    fr = [(pts[i], dhash_wh(r.stdout[i * 272:(i + 1) * 272], 16, 16)) for i in range(n)]
    t0 = fr[0][0]
    k = int((fr[-1][0] - t0) / STEP + 1e-6) + 1
    return ["%064x" % h for h in track_of(fr, t0, k)]


def strong_image(path):
    r = run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-frames:v", "1", "-vf",
             "scale=17:16:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], timeout=600)
    if len(r.stdout) < 272:
        raise RuntimeError("cannot decode image %s" % path)
    return "%064x" % dhash_wh(r.stdout[:272], 16, 16)


def strong_frame_distance(ref, path):
    """mean Hamming distance (of 256 bits) between the reference track (frame shown every STEP s from the first frame)
    and a produced video, each sample compared with the frames within one frame period of its time"""
    fr = frame_hashes(path, size=(17, 16))
    t0 = fr[0][0]
    fr = [(t - t0, h) for t, h in fr]
    period = (fr[-1][0] - fr[0][0]) / max(1, len(fr) - 1) if len(fr) > 1 else 0.04
    if abs(len(ref) - (int(fr[-1][0] / STEP + 1e-6) + 1)) > 2:
        return 256.0
    times = [t for t, _ in fr]
    tot = 0
    for k, rh in enumerate(ref):
        t = k * STEP
        i = bisect.bisect_right(times, t + 1e-4) - 1
        cand = [fr[j][1] for j in range(max(0, i - 1), min(len(fr), i + 2)) if abs(fr[j][0] - t) <= 1.5 * period + 1e-6]
        cand = cand or [fr[max(0, i)][1]]
        tot += min(ham(int(rh, 16), h) for h in cand)
    return tot / len(ref)


def frame_exact(src, pre, ref, P, span, single=False):
    """refine a start time P (file time, from the 0.5 s coarse alignment) to the frame whose 256-bit hashes match
    the reference best: ref is the strong track (or, single=True, the hash of one frame); candidates within 0.25 s"""
    fr = frame_hashes(src.path, P - 0.5, span + 1.0, pre, src.headers, size=(17, 16))
    times = [t for t, _ in fr]
    cands = [t for t in times if abs(t - P) <= 0.25]
    if not cands:
        return P, None
    def score(Pc):
        if single:
            return ham(int(ref, 16), fr[times.index(Pc)][1])
        tot = n = 0
        for k, rh in enumerate(ref):
            t = Pc + k * STEP
            if t > times[-1] + STEP:
                break
            tot += ham(int(rh, 16), displayed(fr, times, t))
            n += 1
        return tot / max(1, n)
    best = min(cands, key=lambda c: (score(c), abs(c - P)))
    return best, score(best)


def strong_distance(ref, got):
    """mean Hamming distance (of 256 bits) between two hex tracks (or two single hashes)"""
    if isinstance(ref, str):
        return float(ham(int(ref, 16), int(got, 16)))
    n = min(len(ref), len(got))
    if n == 0 or abs(len(ref) - len(got)) > 2:
        return 256.0
    return sum(ham(int(a, 16), int(b, 16)) for a, b in zip(ref[:n], got[:n])) / n


def verdict(e, out, coarse_ok, off=0.0, zoom=1.0, src_size=None):
    """final status of a produced file: identical / ok / aligned / lowres / mismatch, plus the numbers behind it"""
    d = {"sha256_equal": sha256(out) == e.get("sha256")}
    fp = e["fingerprint"]
    if fp.get("dhash256"):
        if isinstance(fp["dhash256"], list):
            d["strong"] = round(strong_frame_distance(fp["dhash256"], out), 2)
        else:
            d["strong"] = round(strong_distance(fp["dhash256"], strong_image(out)), 2)
        thr = T256_VID if isinstance(fp["dhash256"], list) else T256_IMG
        same = d["strong"] <= thr and coarse_ok
    else:
        same = coarse_ok
    ref = e["recipe"].get("source_size_ref")
    low = bool(src_size and ref and src_size[1] < LOWRES * ref[1])
    if src_size and ref:
        d["src_size"], d["src_size_ref"] = src_size, ref
    if d["sha256_equal"]:
        d["status"] = "identical"
    elif not same:
        d["status"] = "mismatch"
    elif low:
        d["status"] = "lowres"
    else:
        d["status"] = "ok" if abs(off) < 0.07 and zoom == 1.0 else "aligned"
    return d


def image_hash(path, pre=None):
    vf = ",".join([x for x in (pre, "scale=9:8:flags=area,format=gray") if x])
    r = run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-frames:v", "1", "-vf", vf, "-f", "rawvideo",
             "-pix_fmt", "gray", "pipe:1"], timeout=600)
    if len(r.stdout) < 72:
        raise RuntimeError("cannot decode image %s: %s" % (path, r.stderr.decode(errors="replace")[-300:]))
    return dhash72(r.stdout[:72])


def image_size(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
             "json", str(path)], timeout=120)
    s = json.loads(r.stdout)["streams"][0]
    return [int(s["width"]), int(s["height"])]


def displayed(frames, times, t):
    i = bisect.bisect_right(times, t + 1e-4) - 1
    return frames[max(0, i)][1]


def track_of(frames, t0, n, step=STEP):
    times = [f[0] for f in frames]
    return [displayed(frames, times, t0 + k * step) for k in range(n)]


def video_fingerprint(path):
    info = probe(path)
    fr = frame_hashes(path)
    t0 = fr[0][0]
    n = int((fr[-1][0] - t0) / STEP + 1e-6) + 1
    return {"step_s": STEP, "dhash": [hx(h) for h in track_of(fr, t0, n)], "duration_s": round(info["duration"], 3),
            "fps": info["fps_str"], "size": [info["width"], info["height"]], "frames": len(fr)}


def compare_tracks(ref_hex, hashes):
    n = min(len(ref_hex), len(hashes))
    if n == 0:
        return 64.0
    return sum(ham(unhx(a), b) for a, b in zip(ref_hex[:n], hashes[:n])) / n


def align(frames, ref_hex, anchor, rng, step=STEP, min_cover=0.8):
    """best candidate frame time P (|P-anchor|<=rng; rng=None: anywhere) such that the source sampled at
    P + k*step matches the reference track; returns (score, P) or None"""
    ref = [unhx(h) for h in ref_hex]
    times = [f[0] for f in frames]
    last = times[-1]
    best = None
    for P, _ in frames:
        if rng is not None and abs(P - anchor) > rng + 1e-6:
            continue
        tot = n = 0
        for k, rh in enumerate(ref):
            t = P + k * step
            if t > last + step:
                break
            tot += ham(rh, displayed(frames, times, t))
            n += 1
        if n < max(1, int(min_cover * len(ref))):
            continue
        s = tot / n
        if best is None or s < best[0] - 1e-9 or (abs(s - best[0]) < 1e-9 and abs(P - anchor) < abs(best[1] - anchor)):
            best = (s, P)
    return best


# ----------------------------------------------------------------------------------------------- geometry
def crop_px(rec, W, H, zoom=1.0):
    """crop box in pixels of a W x H copy: exact recorded pixels when the size matches, else from the fractions"""
    ref = rec.get("source_size_ref")
    if zoom == 1.0 and ref and [W, H] == list(ref) and rec.get("crop_px_ref"):
        x0, y0, x1, y1 = rec["crop_px_ref"]
    else:
        fx0, fy0, fx1, fy1 = rec.get("crop_frac") or [0, 0, 1, 1]
        cx, cy, hw, hh = (fx0 + fx1) / 2, (fy0 + fy1) / 2, (fx1 - fx0) / 2 * zoom, (fy1 - fy0) / 2 * zoom
        fx0, fx1, fy0, fy1 = max(0, cx - hw), min(1, cx + hw), max(0, cy - hh), min(1, cy + hh)
        x0, y0, x1, y1 = (int(round(fx0 * W)) // 2 * 2, int(round(fy0 * H)) // 2 * 2,
                          int(round(fx1 * W)) // 2 * 2, int(round(fy1 * H)) // 2 * 2)
    return [max(0, x0), max(0, y0), min(W, x1), min(H, y1)]


def crop_filter(box, W, H):
    x0, y0, x1, y1 = box
    if [x0, y0, x1, y1] == [0, 0, W, H]:
        return None
    return "crop=%d:%d:%d:%d" % (x1 - x0, y1 - y0, x0, y0)


def encode_filters(rec, box, info):
    """crop -> scale to the recorded output size -> fps, as in the original pipeline"""
    W, H = rec["out_size"]
    cw, ch = box[2] - box[0], box[3] - box[1]
    vf = [f for f in [crop_filter(box, info["width"], info["height"])] if f]
    if (cw, ch) != (W, H):
        vf.append("scale=%d:%d%s" % (W, H, ":flags=lanczos" if rec.get("scale_flags") == "lanczos" else ""))
    if rec.get("out_fps"):
        a, b = rec["out_fps"].split("/") if "/" in rec["out_fps"] else (rec["out_fps"], "1")
        want = float(a) / float(b)
        if abs(info["fps"] - want) / want > 0.001:
            vf.append("fps=%s" % rec["out_fps"])
    return vf


def x264_args(rec):
    c = rec.get("codec") or {}
    return ["-an", "-c:v", "libx264", "-preset", c.get("preset", "slow"), "-crf", str(c.get("crf", 18)),
            "-pix_fmt", c.get("pix_fmt", "yuv420p"), "-movflags", "+faststart"]


# ----------------------------------------------------------------------------------------------- sources
class Src:
    """a local file or URL readable by ffmpeg; orig_time = file_time + offset"""

    def __init__(self, path, offset=0.0, headers=None, note=""):
        self.path, self.offset, self.headers, self.note = str(path), float(offset), headers, note


class Fetcher:
    def __init__(self, a):
        self.a = a
        self.cache = Path(a.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.local = {}
        if a.local_source_map:
            self.local = json.load(open(a.local_source_map))
        self.ytdlp = a.ytdlp
        bad = [x for x in a.ytdlp_arg if re.search(r"cookie|username|password|netrc|login", x, re.I)]
        if bad:
            sys.exit("refusing yt-dlp options that use personal credentials: %s" % bad)
        self.zips = {}

    def zip_member(self, zip_url, member=None, suffix=None):
        """bytes of one member of a remote zip, read by HTTP ranges (the central directory is read once per zip)"""
        with lock_for("zip:" + zip_url):
            if zip_url not in self.zips:
                self.zips[zip_url] = zipfile.ZipFile(io.BufferedReader(RangeFile(zip_url), buffer_size=1 << 20))
            z = self.zips[zip_url]
            if member is None:
                names = [n for n in z.namelist() if n.endswith(suffix)]
                if not names:
                    raise RuntimeError("member *%s not in %s" % (suffix, zip_url))
                member = names[0]
            return z.read(member)

    def file_get(self, url, dst):
        """a whole file (cached)"""
        with lock_for(str(dst)):
            if not dst.exists():
                self.http_get(url, dst)
        return dst

    # -- developer stand-ins (self-test): source id -> local original
    def from_local(self, sid, need=None):
        for e in self.local.get(sid) or []:
            cov = e.get("covers")
            if need and cov and not (cov[0] - 1e-3 <= need[0] and need[1] <= cov[1] + 1e-3):
                continue
            if os.path.exists(e["path"]):
                return Src(e["path"], e.get("offset", 0.0), note="local stand-in")
        return None

    def http_get(self, url, dst, headers=None):
        tmp = Path("%s.%d.part" % (dst, os.getpid()))      # per process: two runs may share one cache
        tmp.unlink(missing_ok=True)
        total = None
        for attempt in range(6):          # a dropped connection resumes with a Range request; size is checked
            have = tmp.stat().st_size if tmp.exists() else 0
            h = dict({"User-Agent": UA}, **(headers or {}))
            if have:
                h["Range"] = "bytes=%d-" % have
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                    if have and r.status != 206:
                        have = 0                                 # server ignored the range: start over
                    if total is None and r.headers.get("Content-Length"):
                        total = have + int(r.headers["Content-Length"])
                    with open(tmp, "ab" if have else "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
            except Exception as ex:  # noqa: BLE001
                log("  download interrupted (%s), resuming" % ex)
            if total is None or tmp.stat().st_size >= total:
                break
            time.sleep(3 + 5 * attempt)
        if total is not None and tmp.stat().st_size != total:
            raise RuntimeError("incomplete download of %s (%d of %d bytes)" % (url, tmp.stat().st_size, total))
        os.replace(tmp, dst)
        return dst

    def eyepetizer(self, vid, dst):
        """public play-URL API of Eyepetizer (kaiyanapp): a GET answers with a redirect to the CDN mp4 (HEAD is not
        allowed); no login. Tries the 720p ('high') edition, the video's default edition, then 480p ('normal')."""
        err = None
        for ed in ("high", "default", "normal"):
            u = ("https://baobab.kaiyanapp.com/api/v1/playUrl?vid=%s&resourceType=video&editionType=%s&source=aliyun"
                 "&playUrlType=url_oss&udid=" % (vid, ed))
            try:
                return self.http_get(u, dst)
            except Exception as e:  # noqa: BLE001
                err = e
                log("eyepetizer", vid, ed, e)
        raise RuntimeError("eyepetizer play URL failed: %s" % err)

    def ytdlp_section(self, url, a, b, min_h=None):
        key = slug("%s_%s" % (url, "full" if a is None else "%.2f_%.2f" % (a, b)))
        side = self.cache / ("ytdlp_%s.json" % key)
        with lock_for(key):
            if side.exists():
                d = json.load(open(side))
                if os.path.exists(d["path"]):
                    return Src(d["path"], d["offset"], note="yt-dlp section")
            tmpc = tempfile.mkdtemp(prefix="ytdlp_cache_")
            env = dict(os.environ, XDG_CACHE_HOME=tmpc)
            out = self.cache / ("ytdlp_%s.%%(ext)s" % key)
            fmt = YTDLP_FMT
            if min_h:      # the stream height used in the benchmark (same framing and padding), else a taller one
                fmt = ("bv*[height=%d][vcodec^=avc1]/bv*[height=%d]/bv*[height>=%d][height<=1080][vcodec^=avc1]/"
                       "bv*[height>=%d][height<=1080]/bv*[height>=%d]/%s" % (min_h, min_h, min_h, min_h, min_h, YTDLP_FMT))
            cmd = [self.ytdlp, "--ignore-config", "--no-cookies", "--no-cookies-from-browser", "--no-cache-dir",
                   "--no-playlist", "--no-part", "--no-progress", "-f", fmt, "-o", str(out)]
            if a is not None:
                cmd += ["--download-sections", "*%.2f-%.2f" % (a, b)]
            cmd += list(self.a.ytdlp_arg) + [url]
            try:
                for attempt in range(4):     # Bilibili answers 412 when requests come too fast: wait and retry
                    r = run(cmd, env=env, timeout=7200)
                    if r.returncode == 0 or b"412" not in (r.stderr or b"") or attempt == 3:
                        break
                    log("  yt-dlp got HTTP 412, retrying in %d s" % (60 * (attempt + 1)))
                    time.sleep(60 * (attempt + 1))
            finally:
                shutil.rmtree(tmpc, ignore_errors=True)
            got = [p for p in self.cache.glob("ytdlp_%s.*" % key) if p.suffix in (".mp4", ".mkv", ".webm", ".flv", ".mov")]
            if not got:
                raise RuntimeError("yt-dlp failed: " + (r.stderr or r.stdout).decode(errors="replace")[-500:])
            d = {"url": url, "section": [a, b], "path": str(got[0]), "offset": a or 0.0}
            json.dump(d, open(side, "w"))
            return Src(d["path"], d["offset"], note="yt-dlp section")

    def get(self, s, need, core=None, min_h=None):
        """Src for source spec s covering orig-time window need=[t0,t1] (with margins; a local stand-in only has
        to cover core), or raise"""
        loc = self.from_local(s["id"], core or need)
        if loc or self.local and self.a.local_only:
            if not loc:
                raise RuntimeError("no local stand-in")
            return loc
        kind = s["fetch"]
        if kind == "ytdlp":
            if s.get("offset_known") is False or need is None:
                return self.ytdlp_section(s["url"], None, None, min_h)
            return self.ytdlp_section(s["url"], max(0.0, need[0]), need[1], min_h)
        if kind == "http":
            if self.a.http_download_full or s.get("download_full"):
                dst = self.cache / ("http_%s%s" % (slug(s["url"]), Path(urllib.parse.urlparse(s["url"]).path).suffix))
                with lock_for(str(dst)):
                    if not dst.exists():
                        self.http_get(s["url"], dst, s.get("headers"))
                return Src(dst, s.get("time_offset_s", 0.0))
            # time_offset_s: original-timeline time minus this file's time (another release of the same film)
            return Src(s["url"], s.get("time_offset_s", 0.0), headers=dict({"User-Agent": UA}, **(s.get("headers") or {})),
                       note="http stream")
        if kind == "archive_item":
            # an Internet Archive item: its video files are listed by the metadata API; the largest MP4 is used
            meta = json.load(urllib.request.urlopen(urllib.request.Request(
                "https://archive.org/metadata/%s" % s["identifier"], headers={"User-Agent": UA}), timeout=120))
            files = [f for f in meta.get("files", []) if f.get("name", "").lower().endswith((".mp4", ".m4v"))]
            if not files:
                raise RuntimeError("no MP4 in archive.org item %s" % s["identifier"])
            f = max(files, key=lambda f: int(f.get("size") or 0))
            url = "https://archive.org/download/%s/%s" % (s["identifier"], urllib.parse.quote(f["name"]))
            return Src(url, s.get("time_offset_s", 0.0), headers={"User-Agent": UA}, note="archive.org " + f["name"])
        if kind == "eyepetizer":
            dst = self.cache / ("eyepetizer_%s.mp4" % s["vid"])
            with lock_for(str(dst)):
                if not dst.exists():
                    self.eyepetizer(s["vid"], dst)
            return Src(dst, 0.0)
        raise RuntimeError("unknown fetch kind %s" % kind)


# ----------------------------------------------------------------------------------------------- remote zip
class RangeFile(io.RawIOBase):
    """seekable read-only file over HTTP Range requests (enough for zipfile)"""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": "bytes=%d-%d" % (self.pos, end)})
        for i in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    b = r.read()
                break
            except Exception:  # noqa: BLE001
                if i == 3:
                    raise
                time.sleep(2 + 3 * i)
        self.pos += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)


# ----------------------------------------------------------------------------------------------- media kinds
def search_variants(fetcher, src, rec, ref_hex, anchor_file, R, dur, rot_pre=None):
    info = probe(src.path, src.headers)
    wide = rec.get("_wide")
    best = None
    for z in ZOOMS:
        box = crop_px(rec, info["width"], info["height"], z)
        pre = ",".join([x for x in (rot_pre, crop_filter(box, info["width"], info["height"])) if x]) or None
        if wide:
            fr = frame_hashes(src.path, None, None, pre, src.headers)
            b = align(fr, ref_hex, anchor_file, None)
        else:
            fr = frame_hashes(src.path, anchor_file - R - 0.5, dur + 2 * R + 1.0, pre, src.headers)
            b = align(fr, ref_hex, anchor_file, R)
        if b and (best is None or b[0] < best[0] - 0.25):
            best = (b[0], b[1], z, box)
        if best and best[0] <= T_OK:
            break
    return info, best


def gray_integral(src, t, headers, w, h):
    """integral image of the grey w x h frame shown at file time t"""
    r = run(["ffmpeg", "-v", "error", "-nostdin"] + hdr_args(headers) + ["-ss", "%.4f" % max(0.0, t), "-i", str(src),
             "-frames:v", "1", "-vf", "scale=%d:%d:flags=area,format=gray" % (w, h), "-f", "rawvideo", "-pix_fmt", "gray",
             "pipe:1"], timeout=600)
    if len(r.stdout) < w * h:
        raise RuntimeError("frame decode failed at %.2f s" % t)
    b, W1 = r.stdout, w + 1
    I = [0] * (W1 * (h + 1))
    for y in range(h):
        row, base, prev, o = 0, (y + 1) * W1, y * W1, y * w
        for x in range(w):
            row += b[o + x]
            I[base + x + 1] = I[prev + x + 1] + row
    return I


def box_hash(I, w, box):
    """256-bit dHash of the area-averaged 17x16 grid over a (fractional) pixel box of an integral image"""
    x0, y0, x1, y1 = box
    xs = [int(round(x0 + (x1 - x0) * c / 17.0)) for c in range(18)]
    ys = [int(round(y0 + (y1 - y0) * r / 16.0)) for r in range(17)]
    W1, v = w + 1, 0
    for r in range(16):
        ya, yb = ys[r] * W1, ys[r + 1] * W1
        hh = max(1, ys[r + 1] - ys[r])
        cells = [(I[yb + xs[c + 1]] - I[ya + xs[c + 1]] - I[yb + xs[c]] + I[ya + xs[c]]) / (max(1, xs[c + 1] - xs[c]) * hh)
                 for c in range(17)]
        for c in range(16):
            v = (v << 1) | (1 if cells[c] > cells[c + 1] else 0)
    return v


def refine_framing(src, info, rec, samples):
    """When a platform now serves the video framed differently (another scale or position of the picture), find the
    crop that matches the 256-bit fingerprints: zoom 0.80-1.25 and shifts of up to 5% of the frame, keeping the
    recorded aspect ratio. samples: [(file time, reference 256-bit hash)]. Returns (mean distance, zoom, pixel box)."""
    W, H = info["width"], info["height"]
    w = min(480, W)
    h = max(16, int(round(w * H / float(W))))
    Is = [(gray_integral(src.path, t, src.headers, w, h), ref) for t, ref in samples]
    fx0, fy0, fx1, fy1 = rec.get("crop_frac") or [0, 0, 1, 1]
    cx, cy, hw, hh = (fx0 + fx1) / 2, (fy0 + fy1) / 2, (fx1 - fx0) / 2, (fy1 - fy0) / 2
    best = None
    for zi in range(-8, 11):
        z = 1.0 + 0.025 * zi
        for dxi in range(-4, 5):
            for dyi in range(-4, 5):
                ccx, ccy = cx + 0.0125 * dxi, cy + 0.0125 * dyi
                bx = ((ccx - hw * z) * w, (ccy - hh * z) * h, (ccx + hw * z) * w, (ccy + hh * z) * h)
                if bx[0] < -0.5 or bx[1] < -0.5 or bx[2] > w + 0.5 or bx[3] > h + 0.5:
                    continue
                bx = (max(0.0, bx[0]), max(0.0, bx[1]), min(float(w), bx[2]), min(float(h), bx[3]))
                d = sum(ham(box_hash(I, w, bx), ref) for I, ref in Is) / len(Is)
                if best is None or d < best[0] - 1e-9:
                    best = (d, z, ccx, ccy)
    d, z, ccx, ccy = best
    box = [int(round((ccx - hw * z) * W)) // 2 * 2, int(round((ccy - hh * z) * H)) // 2 * 2,
           int(round((ccx + hw * z) * W)) // 2 * 2, int(round((ccy + hh * z) * H)) // 2 * 2]
    return d, z, [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]


def status_of(score, off, zoom, thr):
    if score > thr:
        return "mismatch"
    return "ok" if abs(off) < 0.07 and zoom == 1.0 else "aligned"


def do_video(fetcher, e, out, R):
    """cut, crop, scale and encode as recorded; if the result does not match, retry from the exact first frame
    (another copy can put its frames at slightly other times), then with a framing search"""
    rec, fp = e["recipe"], e["fingerprint"]
    dur = rec["duration_s"]
    need = [rec["anchor_s"] - R - 5.0, rec["anchor_s"] + dur + R + 5.0]
    last_err = None
    mism = None
    min_h = (rec.get("source_size_ref") or [0, 0])[1] or None
    for s in e["sources"]:
        try:
            src = fetcher.get(s, need, [rec["anchor_s"], rec["anchor_s"] + dur - 0.5], min_h)
            r2 = dict(rec, _wide=s.get("offset_known") is False)
            info, best = search_variants(fetcher, src, r2, fp["dhash"], rec["anchor_s"] - src.offset, R, dur)
            if not best:
                raise RuntimeError("no alignment candidate")
            score, P, z, box = best
            off = P + src.offset - rec["anchor_s"]
            length0 = rec["t_end_s"] - rec["t_start_s"]

            def make(box, z, ss, length):
                inp, outp = seek_args(rec, ss, length, src.headers, src.path)
                vf = encode_filters(rec, box, info)
                tmp = out.with_suffix(".tmp" + out.suffix)
                cmd = (["ffmpeg", "-nostdin", "-v", "error", "-y"] + inp + ["-map", "0:v:0"] + outp +
                       (["-vf", ",".join(vf)] if vf else []) + x264_args(rec) + [str(tmp)])
                r = run(cmd, timeout=7200)
                if r.returncode:
                    raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[-300:])
                os.replace(tmp, out)
                got = video_fingerprint(out)
                v = compare_tracks(fp["dhash"], [unhx(h) for h in got["dhash"]])
                res = verdict(e, out, v <= T_OK and got["size"] == fp.get("size", got["size"]), off, z,
                              [info["width"], info["height"]])
                res.update(offset_s=round(off, 3), zoom=z, score=round(v, 2), search_score=round(score, 2),
                           source=s["id"], via=src.note, out_size=got["size"], duration_s=got["duration_s"])
                return res

            ss0 = rec["t_start_s"] + off - src.offset
            res = make(box, z, ss0, length0)
            if res["status"] == "mismatch" and isinstance(fp.get("dhash256"), list):
                # exact first frame by the 256-bit track; start just before it and keep the recorded end time
                Pe, _ = frame_exact(src, crop_filter(box, info["width"], info["height"]), fp["dhash256"], P, dur)
                ss_end = ss0 + length0 + (Pe - P)
                res2 = make(box, z, Pe - 0.001, ss_end - (Pe - 0.001))
                res2["exact_frame_retry"] = round(Pe - P, 4)
                if res2["status"] != "mismatch" or res2.get("strong", 256) < res.get("strong", 256):
                    res, P, ss0 = res2, Pe, Pe - 0.001
                    length0 = ss_end - ss0
            if res["status"] == "mismatch" and isinstance(fp.get("dhash256"), list):
                n = len(fp["dhash256"])
                ks = sorted({int(round(i * (n - 1) / 7.0)) for i in range(8)})
                d, z2, box2 = refine_framing(src, info, rec, [(P + k * STEP, int(fp["dhash256"][k], 16)) for k in ks])
                log("   %s %s framing search: zoom %.3f box %s distance %.1f" % (e["item"], e["file"], z2, box2, d))
                if d <= 1.5 * T256_VID and box2 != box:
                    res2 = make(box2, z2 if z2 != 1.0 else 0.999, ss0, length0)
                    res2["framing_search"] = round(d, 1)
                    if res2["status"] != "mismatch" or res2.get("strong", 256) < res.get("strong", 256):
                        res = res2
            if res["status"] != "mismatch":
                return res
            mism = mism or res
            last_err = "%s: mismatch (strong %s)" % (s["id"], res.get("strong"))
            log("  ", e["item"], e["file"], last_err)
        except Exception as ex:  # noqa: BLE001
            last_err = "%s: %s" % (s["id"], ex)
            log("  ", e["item"], e["file"], last_err[:300])
    return mism or {"status": "unavailable", "reason": last_err}


def pick_fps_frame(src, ss, win_len, n, pre, out, headers=None, rot=None):
    vf = ",".join([x for x in ("fps=15", "select=eq(n\\,%d)" % n, pre) if x])
    r = run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + hdr_args(headers) + ["-ss", "%.4f" % max(0.0, ss), "-i", src,
             "-t", "%.4f" % win_len, "-map", "0:v:0", "-vf", vf, "-frames:v", "1", "-q:v", "2", str(out)], timeout=1800)
    if r.returncode or not out.exists():
        raise RuntimeError("frame extraction failed: " + r.stderr.decode(errors="replace")[-300:])


def do_frame(fetcher, e, out, R):
    rec, fp = e["recipe"], e["fingerprint"]
    tr = fp.get("align_track")
    last_err = None
    mism = None
    min_h = (rec.get("source_size_ref") or [0, 0])[1] or None
    R0 = R
    for s in e["sources"]:
        try:
            R = max(R0, s.get("search_range_s", 0.0))    # another release of a film: search a wider time window
            t_need = rec["anchor_s"]
            span = (len(tr["dhash"]) * STEP) if tr else 2.0
            src = fetcher.get(s, [t_need - R - 5.0, t_need + span + R + 5.0], [t_need, t_need + 0.5], min_h)
            r2 = dict(rec, _wide=s.get("offset_known") is False)
            if tr:
                info, best = search_variants(fetcher, src, r2, tr["dhash"], tr["anchor_s"] - src.offset, R, span)
                if not best:
                    raise RuntimeError("no alignment candidate")
                score, P, z, box = best
                off = P + src.offset - tr["anchor_s"]
            else:
                info = probe(src.path, src.headers)
                score, off, z = 0.0, 0.0, 1.0
                box = crop_px(rec, info["width"], info["height"])
            m = rec["method"]
            if m not in ("first_frame_of_encode", "fps15_pick"):
                raise RuntimeError("unknown frame method " + m)

            def make(box, z, off):
                if m == "first_frame_of_encode":
                    tmpv = out.with_suffix(".tmp.mp4")
                    inp, outp = seek_args(rec, rec["t_frame_s"] + off - src.offset, 1.0, src.headers, src.path)
                    vf = encode_filters(rec, box, info)
                    cmd = (["ffmpeg", "-nostdin", "-v", "error", "-y"] + inp + ["-map", "0:v:0"] + outp +
                           (["-vf", ",".join(vf)] if vf else []) + x264_args(rec) + [str(tmpv)])
                    r = run(cmd, timeout=1800)
                    if r.returncode:
                        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[-300:])
                    r = run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmpv), "-frames:v", "1", "-q:v", "2", str(out)])
                    tmpv.unlink(missing_ok=True)
                    if r.returncode:
                        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[-300:])
                else:
                    vf = encode_filters(rec, box, info) if rec.get("out_size") else []
                    pick_fps_frame(src.path, rec["seek_s"] + off - src.offset, rec["window_s"], rec["index"],
                                   ",".join(vf) or None, out, src.headers)
                v = ham(image_hash(out), unhx(fp["dhash"]))
                res = verdict(e, out, v <= T_IMG, off, z, [info["width"], info["height"]])
                res.update(offset_s=round(off, 3), zoom=z, score=v, search_score=round(score, 2), source=s["id"],
                           via=src.note, out_size=image_size(out))
                return res

            res = make(box, z, off)
            t_file = (rec["t_frame_s"] if m == "first_frame_of_encode" else rec["seek_s"] + rec["index"] / 15.0) + off - src.offset
            if res["status"] == "mismatch" and m == "first_frame_of_encode" and isinstance(fp.get("dhash256"), str):
                # exact frame by its 256-bit hash, then extract from just before it
                t1, _ = frame_exact(src, crop_filter(box, info["width"], info["height"]), fp["dhash256"], t_file, 0.0, single=True)
                off2 = off + (t1 - 0.001) - t_file
                res2 = make(box, z, off2)
                res2["exact_frame_retry"] = round(t1 - t_file, 4)
                if res2["status"] != "mismatch" or res2.get("strong", 256) < res.get("strong", 256):
                    res, off, t_file = res2, off2, t1
            if res["status"] == "mismatch" and isinstance(fp.get("dhash256"), str):
                d, z2, box2 = refine_framing(src, info, rec, [(t_file, int(fp["dhash256"], 16))])
                log("   %s %s framing search: zoom %.3f box %s distance %.1f" % (e["item"], e["file"], z2, box2, d))
                if d <= T256_IMG and box2 != box:
                    res2 = make(box2, z2 if z2 != 1.0 else 0.999, off)
                    res2["framing_search"] = round(d, 1)
                    if res2["status"] != "mismatch" or res2.get("strong", 256) < res.get("strong", 256):
                        res = res2
            if res["status"] != "mismatch":
                return res
            mism = mism or res
            last_err = "%s: mismatch (strong %s)" % (s["id"], res.get("strong"))
            log("  ", e["item"], e["file"], last_err)
        except Exception as ex:  # noqa: BLE001
            last_err = "%s: %s" % (s["id"], ex)
            log("  ", e["item"], e["file"], last_err[:300])
    return mism or {"status": "unavailable", "reason": last_err}


ROT = {0: None, 1: "transpose=2", 2: "hflip,vflip", 3: "transpose=1"}   # np.rot90(img, k): k quarter turns CCW


def replay_cv2(src, ops, out):
    """Replay the photo pipeline as it was run (OpenCV decode, undistort / resize / rotate, each JPEG save re-read),
    writing the bytes of the last JPEG save. Returns False when numpy + opencv-python are not installed (the ffmpeg
    path then gives the same content, not the same bytes)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError("cannot decode %s" % src)
    data = None
    for op in ops:
        if op["op"] == "fisheye_undistort":
            H, W = im.shape[:2]
            im = cv2.fisheye.undistortImage(im, np.array(op["K_raw"]), np.array(op["dist"]), Knew=np.array(op["K_new"]),
                                            new_size=(W, H))
        elif op["op"] == "resize":
            im = cv2.resize(im, (op["w"], op["h"]), interpolation=cv2.INTER_AREA)
        elif op["op"] == "rot90":
            im = np.ascontiguousarray(np.rot90(im, op["k"]))
        elif op["op"] == "jpeg":
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(op["quality"])])
            if not ok:
                raise RuntimeError("jpeg encode failed")
            data = buf.tobytes()
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            raise RuntimeError("unknown op %s" % op["op"])
    if data is None:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        data = buf.tobytes()
    Path(out).write_bytes(data)
    return True


def dataset_path(root, s):
    if not root:
        return None
    for rel in [s.get("path")] + list(s.get("alt_paths") or []):
        if rel:
            p = Path(root) / s.get("root_dir", "") / rel
            if p.exists():
                return p
    return None


def do_dataset(fetcher, e, out, R):
    rec, fp = e["recipe"], e["fingerprint"]
    last_err = None
    work = Path(tempfile.mkdtemp(prefix="pv_ds_", dir=fetcher.cache))
    try:
        for s in e["sources"]:
            try:
                loc = fetcher.from_local(s["id"])
                p = Path(loc.path) if loc else dataset_path(fetcher.a.datasets_root, s)
                ext = Path(s.get("member") or s.get("path") or "").suffix if s["fetch"] in ("zip_member", "hf_file") else ""
                img = work / ("src" + (ext or ".png"))      # a frame taken from a movie is always written as PNG
                if s["fetch"] == "zip_member":
                    if p is None:                      # remote zip: read only the member, by HTTP ranges
                        img.write_bytes(fetcher.zip_member(s["zip_url"], s.get("member"), s.get("member_suffix")))
                        p = img
                elif s["fetch"] == "hf_file":
                    if p is None:
                        p = fetcher.file_get(s["url"], fetcher.cache / ("hf_%s%s" % (slug(s["url"]), Path(s["url"]).suffix)))
                if p is not None and s.get("source_sha256") and sha256(p) != s["source_sha256"]:
                    raise RuntimeError("source_differs: %s is not the file used in the benchmark (sha256)" % s["id"])
                elif s["fetch"] == "movie_frame":
                    if p is None and s.get("public_url"):
                        dst = fetcher.cache / ("movie_%s" % Path(urllib.parse.urlparse(s["public_url"]).path).name)
                        with lock_for(str(dst)):
                            if not dst.exists():
                                log("  downloading", s["public_url"])
                                fetcher.http_get(s["public_url"], dst)
                        p = dst
                    if p is None:
                        raise RuntimeError("movie not found")
                    r = run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-noautorotate", "-i", str(p), "-map", "0:v:0",
                             "-vf", "select=eq(n\\,%d)" % s["frame_index"], "-fps_mode", "passthrough", "-frames:v", "1",
                             str(img)], timeout=3600)
                    if r.returncode or not img.exists():
                        raise RuntimeError("frame %d: %s" % (s["frame_index"], r.stderr.decode(errors="replace")[-300:]))
                    p = img
                if p is None:
                    raise RuntimeError("needs_dataset: %s %s under --datasets-root" % (s.get("dataset"), s.get("path")))
                if replay_cv2(p, rec["ops"], out):    # the original OpenCV steps: byte-identical photos
                    h = image_hash(out)
                    v = ham(h, unhx(fp["dhash"]))
                    sz = image_size(out)
                    res = verdict(e, out, v <= T_IMG and sz == fp["size"])
                    res.update(score=v, source=s["id"], via=("local stand-in" if loc else s["fetch"]) + " +opencv", out_size=sz)
                    return res
                cur = p
                for op in rec["ops"]:
                    if op["op"] == "fisheye_undistort":
                        try:
                            import cv2
                            import numpy as np
                        except ImportError:
                            raise RuntimeError("needs_opencv: pip install numpy opencv-python (ScanNet++ fisheye)")
                        im = cv2.imread(str(cur))
                        H, W = im.shape[:2]
                        im = cv2.fisheye.undistortImage(im, np.array(op["K_raw"]), np.array(op["dist"]),
                                                        Knew=np.array(op["K_new"]), new_size=(W, H))
                        nxt = work / ("u%d.png" % len(list(work.iterdir())))
                        cv2.imwrite(str(nxt), im)
                        cur = nxt
                    elif op["op"] in ("rot90", "resize"):
                        vf = ROT[op["k"] % 4] if op["op"] == "rot90" else "scale=%d:%d:flags=%s" % (op["w"], op["h"], op.get("flags", "area"))
                        nxt = work / ("o%d.png" % len(list(work.iterdir())))
                        if vf:
                            r = run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(cur), "-vf", vf, str(nxt)])
                            if r.returncode:
                                raise RuntimeError(r.stderr.decode(errors="replace")[-300:])
                            cur = nxt
                    elif op["op"] == "jpeg":
                        pass
                    else:
                        raise RuntimeError("unknown op %s" % op["op"])
                r = run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(cur), "-frames:v", "1", "-q:v", "2", str(out)])
                if r.returncode:
                    raise RuntimeError(r.stderr.decode(errors="replace")[-300:])
                h = image_hash(out)
                v = ham(h, unhx(fp["dhash"]))
                sz = image_size(out)
                res = verdict(e, out, v <= T_IMG and sz == fp["size"])
                res.update(score=v, source=s["id"], via="local stand-in" if loc else s["fetch"], out_size=sz)
                return res
            except Exception as ex:  # noqa: BLE001
                last_err = "%s: %s" % (s["id"], ex)
                log("  ", e["item"], e["file"], last_err[:300])
        return {"status": "mismatch" if "source_differs" in str(last_err) else "unavailable", "reason": last_err}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------------------------- main
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=str(here / "external.json"))
    ap.add_argument("--out", default=str(here / "tasks"), help="the dataset's tasks/ directory (default: next to this script)")
    ap.add_argument("--cache", default=str(here / "fetch_cache"), help="download cache dir")
    ap.add_argument("--tasks", "--items", dest="items", default="", help="comma list of task ids (default all)")
    ap.add_argument("--kinds", default="", help="comma list of input kinds: image,continuation,video")
    ap.add_argument("--check", action="store_true", help="download nothing; report which files are missing")
    ap.add_argument("--datasets-root", default=None, help="root holding licensed datasets (see docs/source-reconstruction.md)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--search-range", type=float, default=3.0, help="time offset search range (s)")
    ap.add_argument("--ytdlp", default="yt-dlp", help="yt-dlp executable")
    ap.add_argument("--ytdlp-arg", action="append", default=[], help="extra yt-dlp argument (e.g. --proxy=URL)")
    ap.add_argument("--http-download-full", action="store_true", help="download whole files instead of reading ranges")
    ap.add_argument("--report", default=None, help="write the JSON report here (default fetch_report.json next to this script)")
    ap.add_argument("--overwrite", action="store_true", help="redo files that already verify")
    ap.add_argument("--local-source-map", default=None, help=argparse.SUPPRESS)   # developer self-test
    ap.add_argument("--local-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    for e in man["media"]:              # external.json names the fields task_id / task_kind / media_kind
        e.setdefault("item", e.get("task_id"))
        e.setdefault("task", e.get("task_kind"))
        e.setdefault("kind", e.get("media_kind"))
    items = set(filter(None, a.items.split(",")))
    tasks = set(filter(None, a.kinds.split(",")))
    todo = [e for e in man["media"] if (not items or e["item"] in items) and (not tasks or e["task"] in tasks)]
    outroot = Path(a.out)
    if a.check:
        miss, summ = [], {}
        for e in todo:
            out = outroot / e["item"] / e["file"]
            st = ("missing" if not out.exists() else
                  "identical" if e.get("sha256") and sha256(out) == e["sha256"] else "present")
            summ[st] = summ.get(st, 0) + 1
            if st == "missing":
                miss.append("%s/%s" % (e["item"], e["file"]))
        for m in miss:
            print("missing", m)
        print(json.dumps(summ), "(present = rebuilt with the same content but different bytes, see fetch_report.json)")
        sys.exit(1 if miss else 0)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit("%s not found on PATH" % tool)
    fetcher = Fetcher(a)

    def one(e):
        out = outroot / e["item"] / e["file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if out.exists() and not a.overwrite and e.get("sha256") and sha256(out) == e["sha256"]:
            res = {"status": "identical", "note": "already present"}
        else:
            fn = {"video_clip": do_video, "frame_from_video": do_frame, "dataset_frame": do_dataset}[e["kind"]]
            try:
                res = fn(fetcher, e, out, a.search_range)
            except Exception as ex:  # noqa: BLE001
                res = {"status": "unavailable", "reason": str(ex)[:500]}
        res.update(task_id=e["item"], task_kind=e["task"], file=e["file"], media_kind=e["kind"],
                   seconds=round(time.time() - t, 1))
        log("%-9s %-13s %-22s %-10s %s" % (e["item"], e["task"], e["file"], res["status"],
                                          {k: res[k] for k in ("offset_s", "zoom", "score", "via") if k in res}))
        return res

    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        results = list(ex.map(one, todo))
    summ = {}
    for r in results:
        summ[r["status"]] = summ.get(r["status"], 0) + 1
    rep = {"manifest": Path(a.manifest).name, "n": len(results), "summary": summ, "results": results}
    rp = Path(a.report) if a.report else here / "fetch_report.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(rp, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ), "report:", rp)
    return int(any(r["status"] not in ("identical", "ok", "aligned") for r in results))


if __name__ == "__main__":
    sys.exit(main())
