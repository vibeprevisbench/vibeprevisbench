"""5.2 (M15): the clip_start of every camera object in a saved .blend file (stdlib; zstd via compression.zstd or
zstandard when the file is compressed).

    camera_clips(path) -> {object name: {"clip_start": float, "animated": bool}}

Reads the file's own SDNA (struct layout) so it does not depend on one Blender version: the Object blocks give the
object name and its data pointer, the Camera blocks give clip_start and the AnimData pointer (animated = the camera
data has animation data and the file holds an RNA path "clip_start" (an F-curve or driver), then its clip_start can
change over time and is not used). Raises ValueError on a file it cannot read.
"""
import struct
from pathlib import Path


def _decompress(b):
    if b[:4] != b"\x28\xb5\x2f\xfd":
        return b
    try:
        from compression import zstd
        return zstd.decompress(b)
    except ImportError:
        import io
        import zstandard                          # 5.2 (addendum 3): Blender writes several zstd frames; read all of them
        return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(b), read_across_frames=True).read()


def _blocks(b):
    if b[:7] != b"BLENDER":
        raise ValueError("not a .blend file")
    if b[7:9].isdigit() and b[9:10] == b"-":          # 5.x: BLENDER17-01v0502, 32-byte block heads
        hsize = int(b[7:9])
        if b[10:12] != b"01" or b[12:13] != b"v":
            raise ValueError("unknown .blend header %r" % b[:hsize])
        pos, end = hsize, len(b)
        while pos + 32 <= end:
            code, sdna, old, ln, nr = struct.unpack_from("<4siQqq", b, pos)
            pos += 32
            yield code, sdna, old, b[pos:pos + ln], nr
            pos += ln
            if code == b"ENDB":
                return
        return
    ptr = 8 if b[7:8] == b"-" else 4                  # legacy: BLENDER-v405 / BLENDER_v279
    if b[8:9] != b"v":
        raise ValueError("big-endian .blend not supported")
    pos, end = 12, len(b)
    fmt = "<4siQii" if ptr == 8 else "<4siIii"
    hl = struct.calcsize(fmt)
    while pos + hl <= end:
        code, ln, old, sdna, nr = struct.unpack_from(fmt, b, pos)
        pos += hl
        yield code, sdna, old, b[pos:pos + ln], nr
        pos += ln
        if code == b"ENDB":
            return


def _sdna(d):
    p = 8 if d[:8] == b"SDNANAME" else None
    if p is None:
        raise ValueError("no SDNA")

    def strings(p, n):
        out = []
        for _ in range(n):
            e = d.index(b"\x00", p)
            out.append(d[p:e].decode("latin-1"))
            p = e + 1
        return out, p

    def align(p):
        return (p + 3) & ~3

    n, = struct.unpack_from("<i", d, p)
    names, p = strings(p + 4, n)
    p = align(p)
    if d[p:p + 4] != b"TYPE":
        raise ValueError("SDNA TYPE")
    n, = struct.unpack_from("<i", d, p + 4)
    types, p = strings(p + 8, n)
    p = align(p)
    if d[p:p + 4] != b"TLEN":
        raise ValueError("SDNA TLEN")
    tlen = struct.unpack_from("<%dh" % len(types), d, p + 4)
    p = align(p + 4 + 2 * len(types))
    if d[p:p + 4] != b"STRC":
        raise ValueError("SDNA STRC")
    n, = struct.unpack_from("<i", d, p + 4)
    p += 8
    structs = []
    for _ in range(n):
        t, nf = struct.unpack_from("<hh", d, p)
        f = struct.unpack_from("<%dh" % (2 * nf), d, p + 4)
        structs.append((t, [(f[2 * i], f[2 * i + 1]) for i in range(nf)]))
        p += 4 + 4 * nf
    return names, types, tlen, structs


def _layout(names, types, tlen, structs, ptr=8):
    """struct name -> (sdna index, {field name: (offset, type name, is pointer, array length)})"""
    out = {}
    for si, (t, fields) in enumerate(structs):
        off, lay = 0, {}
        for ft, fn in fields:
            nm = names[fn]
            is_ptr = nm.startswith("*") or nm.startswith("(*")
            arr = 1
            for part in nm.split("[")[1:]:
                arr *= int(part.split("]")[0])
            base = nm.lstrip("*").split("[")[0].strip("()*")
            size = (ptr if is_ptr else tlen[ft]) * arr
            lay[base] = (off, types[ft], is_ptr, arr)
            off += size
        out[types[t]] = (si, lay)
    return out


def camera_clips(path):
    b = _decompress(Path(path).read_bytes())
    blocks = list(_blocks(b))
    dna = [blk for blk in blocks if blk[0] == b"DNA1"]
    if not dna:
        raise ValueError("no DNA1 block")
    L = _layout(*_sdna(dna[0][3]))
    if "Camera" not in L or "Object" not in L or "ID" not in L:
        raise ValueError("SDNA lacks Camera / Object / ID")
    _, id_l = L["ID"]
    cam_si, cam_l = L["Camera"]
    ob_si, ob_l = L["Object"]
    noff, _, _, nlen = id_l["name"]
    fld = "clip_start" if "clip_start" in cam_l else "clipsta"   # the file keeps the old DNA name
    co, ct = cam_l[fld][0], cam_l[fld][1]
    if ct != "float":
        raise ValueError("Camera.%s is %s" % (fld, ct))
    aoff = cam_l["adt"][0] if "adt" in cam_l else None
    # an F-curve or driver on clip_start stores the RNA path "clip_start" (the SDNA itself says "clipsta")
    clip_path = any(blk[0] != b"DNA1" and b"clip_start" in blk[3] for blk in blocks)
    cams = {}
    for code, sdna, old, data, nr in blocks:
        if sdna == cam_si and code[:2] == b"CA":
            clip, = struct.unpack_from("<f", data, co)
            adt = struct.unpack_from("<Q", data, aoff)[0] if aoff is not None else 0
            cams[old] = {"clip_start": float(clip), "animated": adt != 0 and clip_path}
    out = {}
    doff = ob_l["data"][0]
    oid = ob_l["id"][0]
    for code, sdna, old, data, nr in blocks:
        if sdna == ob_si and code[:2] == b"OB":
            dp, = struct.unpack_from("<Q", data, doff)
            if dp in cams:
                raw = data[oid + noff:oid + noff + nlen]
                name = raw.split(b"\x00", 1)[0][2:].decode("utf-8", "replace")
                out[name] = dict(cams[dp])
    return out


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(camera_clips(sys.argv[1]), indent=1))
