"""pvtools: semantic tags for previs-benchmark scenes (runs inside Blender).

    from pvtools import tag, tag_faces, tag_space, pv_check

    tag(obj, "chair")                  # class only
    tag(obj, "person", "P01")          # class + public entity id from input/entities.json
    tag(collection, "shelf")           # every object in the collection inherits it
    tag_faces(obj, "door", faces=[4, 5])   # part of one mesh is a different thing
    tag_space(collection, "S02")       # optional: which listed space these objects belong to
    pv_check()                         # prints and returns a self-check report (no score, no reference data)

A tag is a (class, id) pair stored as custom properties `pv_class` / `pv_id`. For every rendered piece of
geometry the nearest tag wins as a whole: face tag > the object's own tag > nearest tagged parent object >
nearest tagged collection. `pv_space` resolves through the same chain independently. Classes come from
input/categories.json; anything not listed is written as `other:<word>`.

The scorer imports this same file to resolve tags, so model-side checks and scoring agree.
"""
import json
import os
import re
from pathlib import Path

VERSION = "pvtools-5.0"
_OTHER = re.compile(r"^other:[a-z][a-z0-9_]{0,31}$")
_ID = re.compile(r"^[A-Z]\d{2,3}$")
_SPACE = re.compile(r"^S\d{2}$")
_tables = {}


# ------------------------------------------------------------------ public tables
def _input_dir():
    for d in (os.environ.get("PV_INPUT_DIR"), "input", str(Path(__file__).resolve().parent.parent / "input")):
        if d and Path(d).is_dir():
            return Path(d)
    return None


def _load(name):
    if name not in _tables:
        d = _input_dir()
        f = d / name if d else None
        _tables[name] = json.loads(f.read_text()) if f and f.is_file() else None
    return _tables[name]


def categories():
    """{class: {"group": str, "size_m": [lo, hi]}} from input/categories.json (empty if absent)."""
    t = _load("categories.json") or {}
    return {c["class"]: c for c in t.get("classes", [])}


def entities():
    """{id: {"class": str, "description": str}} from input/entities.json (empty if absent)."""
    t = _load("entities.json") or {}
    return {e["id"]: e for e in t.get("entities", [])}


def spaces():
    t = _load("entities.json") or {}
    return {s["id"]: s for s in t.get("spaces", [])}


def valid_class(cls):
    return isinstance(cls, str) and (cls in categories() or bool(_OTHER.match(cls)) or not categories())


def _warn(msg):
    print("PV_WARNING " + msg, flush=True)


# ------------------------------------------------------------------ writing tags
def _target(x):
    import bpy
    if isinstance(x, str):
        t = bpy.data.objects.get(x) or bpy.data.collections.get(x)
        if t is None:
            raise ValueError(f"pvtools: no object or collection named {x!r}")
        return t
    if isinstance(x, (bpy.types.Object, bpy.types.Collection)):
        return x
    raise TypeError("pvtools: expected a bpy Object, Collection or its name")


def tag(target, cls, id=None):
    """Tag an object or collection with a class and optional public entity id. Returns the target."""
    t = _target(target)
    if not valid_class(cls):
        _warn(f"{t.name}: class {cls!r} is not in input/categories.json; use a listed class or 'other:<word>'")
    if id is not None:
        ents = entities()
        if not _ID.match(str(id)):
            _warn(f"{t.name}: id {id!r} should look like P01 / V01 / O01")
        elif ents and id not in ents:
            _warn(f"{t.name}: id {id!r} is not listed in input/entities.json")
        elif ents and ents[id].get("class") and ents[id]["class"] != cls:
            _warn(f"{t.name}: {id} is listed as {ents[id]['class']!r}, tagged {cls!r}")
    t["pv_class"] = cls
    if id is None:
        if "pv_id" in t:
            del t["pv_id"]
    else:
        t["pv_id"] = str(id)
    return t


def tag_space(target, space):
    """Mark which listed space (S01, S02, ...) an object or collection belongs to."""
    t = _target(target)
    if not _SPACE.match(str(space)):
        _warn(f"{t.name}: space {space!r} should look like S01")
    elif spaces() and space not in spaces():
        _warn(f"{t.name}: space {space!r} is not listed in input/entities.json")
    t["pv_space"] = str(space)
    return t


def tag_faces(target, cls, id=None, faces=None, material=None):
    """Tag some faces of a mesh object (by polygon indices or by material slot name/index).

    Face tags are stored on the mesh data: an INT face attribute `pv_tag` (0 = inherit the object's tag,
    k >= 1 = entry k-1 of the JSON list mesh["pv_face_tags"], each [class, id-or-null]).
    """
    import bpy
    obj = _target(target)
    if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
        raise TypeError("pvtools.tag_faces: expected a mesh object")
    if not valid_class(cls):
        _warn(f"{obj.name}: class {cls!r} is not in input/categories.json")
    me = obj.data
    if (faces is None) == (material is None):
        raise ValueError("pvtools.tag_faces: give exactly one of faces= or material=")
    if material is not None:
        slot = material if isinstance(material, int) else next(
            (i for i, s in enumerate(obj.material_slots) if s.material and s.material.name == material), None)
        if slot is None:
            raise ValueError(f"pvtools.tag_faces: no material slot {material!r} on {obj.name}")
        faces = [p.index for p in me.polygons if p.material_index == slot]
    table = json.loads(me.get("pv_face_tags", "[]"))
    entry = [cls, None if id is None else str(id)]
    if entry not in table:
        table.append(entry)
    k = table.index(entry) + 1
    attr = me.attributes.get("pv_tag")
    if attr is None or attr.domain != "FACE" or attr.data_type != "INT":
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new("pv_tag", "INT", "FACE")
    n = len(me.polygons)
    for f in faces:
        if not 0 <= int(f) < n:
            raise IndexError(f"pvtools.tag_faces: face {f} out of range on {obj.name}")
        attr.data[int(f)].value = k
    me["pv_face_tags"] = json.dumps(table)
    return obj


# ------------------------------------------------------------------ resolution (shared with the scorer)
def _own(idb, key="pv_class"):
    if idb is None or key not in idb.keys():
        return None
    if key == "pv_class":
        return (str(idb["pv_class"]), str(idb["pv_id"]) if "pv_id" in idb.keys() else None)
    return str(idb[key])


def _collection_parents(scene):
    parents = {}

    def walk(c):
        for ch in c.children:
            parents.setdefault(ch.name, c)
            walk(ch)
    walk(scene.collection)
    return parents


def _from_collections(obj, scene, key, parents):
    """Nearest tagged collection: breadth-first upward from every collection holding the object."""
    level = list(obj.users_collection)
    seen = set()
    while level:
        hits = sorted((c.name, _own(c, key)) for c in level if _own(c, key) is not None)
        if hits:
            if len({h[1] for h in hits}) > 1:
                return hits[0][1], "collection:" + hits[0][0] + " (ambiguous)"
            return hits[0][1], "collection:" + hits[0][0]
        seen.update(c.name for c in level)
        level = [parents[c.name] for c in level if c.name in parents and parents[c.name].name not in seen]
    return None, None


def resolve(obj, scene=None, key="pv_class", _parents=None):
    """(value, source) for an object: own > nearest tagged parent object > nearest tagged collection.

    key="pv_class" returns value=(class, id); key="pv_space" returns value="S01". Face tags are separate
    (see face_tags()).
    """
    import bpy
    scene = scene or bpy.context.scene
    v = _own(obj, key)
    if v is not None:
        return v, "own"
    p, depth = obj.parent, 1
    while p is not None:
        v = _own(p, key)
        if v is not None:
            return v, f"parent:{p.name}"
        p, depth = p.parent, depth + 1
    v, src = _from_collections(obj, scene, key, _parents if _parents is not None else _collection_parents(scene))
    if v is not None:
        return v, src
    return None, None


def face_tags(mesh):
    """(per-face int array or None, table [[class, id], ...]) for a Mesh datablock."""
    attr = mesh.attributes.get("pv_tag")
    table = json.loads(mesh.get("pv_face_tags", "[]")) if "pv_face_tags" in mesh.keys() else []
    if attr is None or attr.domain != "FACE" or not table:
        return None, table
    vals = [0] * len(attr.data)
    attr.data.foreach_get("value", vals)
    return vals, table


RENDERABLE = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD", "VOLUME", "GREASEPENCIL"}


def resolve_all(scene=None):
    """One row per renderable object in the scene (instances are resolved by the scorer's readback)."""
    import bpy
    scene = scene or bpy.context.scene
    parents = _collection_parents(scene)
    rows = []
    for obj in scene.objects:
        if obj.type not in RENDERABLE:
            continue
        lab, src = resolve(obj, scene, "pv_class", parents)
        sp, sp_src = resolve(obj, scene, "pv_space", parents)
        row = {"object": obj.name, "type": obj.type, "class": lab[0] if lab else None, "id": lab[1] if lab else None,
               "source": src, "space": sp, "space_source": sp_src, "hide_render": bool(obj.hide_render),
               "face_tags": None}
        if obj.type == "MESH":
            vals, table = face_tags(obj.data)
            if vals is not None:
                counts = {}
                for v in vals:
                    counts[v] = counts.get(v, 0) + 1
                row["face_tags"] = [{"class": table[k - 1][0], "id": table[k - 1][1], "faces": n}
                                    for k, n in sorted(counts.items()) if 0 < k <= len(table)]
                row["faces_inherit"] = counts.get(0, 0)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ self-check
def _world_bbox(obj, depsgraph):
    from mathutils import Vector
    ev = obj.evaluated_get(depsgraph)
    pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return lo, hi


def _in_view(scene, cam, lo, hi):
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                c = world_to_camera_view(scene, cam, Vector((x, y, z)))
                if c.z > 0 and 0 <= c.x <= 1 and 0 <= c.y <= 1:
                    return True
    return False


def pv_check(scene=None, sample_fps=1.0, verbose=True):
    """Report how the scene's tags resolve. Reveals nothing about scoring or reference data."""
    import bpy
    scene = scene or bpy.context.scene
    rows = resolve_all(scene)
    cats, ents = categories(), entities()
    by_id, unknown, untagged, size_warn, warns = {}, {}, [], [], []
    for r in rows:
        labs = [(r["class"], r["id"])] if r["class"] else []
        labs += [(f["class"], f["id"]) for f in (r["face_tags"] or [])]
        if not r["class"] and not (r["face_tags"] and r.get("faces_inherit", 1) == 0):
            if not r["hide_render"]:
                untagged.append(r["object"])
        for cls, eid in labs:
            if not valid_class(cls):
                unknown.setdefault(cls, []).append(r["object"])
            if eid:
                by_id.setdefault(eid, {"classes": set(), "objects": []})
                by_id[eid]["classes"].add(cls)
                by_id[eid]["objects"].append(r["object"])
    # entity extents at the first frame, and which sampled frames show them in the active camera
    f0, f1 = scene.frame_start, scene.frame_end
    fps = scene.render.fps / (scene.render.fps_base or 1)
    step = max(1, int(round(fps / sample_fps)))
    frames = list(range(f0, f1 + 1, step))
    if frames and frames[-1] != f1:
        frames.append(f1)
    keep = scene.frame_current
    report_ents = {}
    if by_id or ents:
        seen = {e: [] for e in by_id}
        for fr in frames:
            scene.frame_set(fr)
            dg = bpy.context.evaluated_depsgraph_get()
            cam = scene.camera
            for e, d in by_id.items():
                boxes = [_world_bbox(scene.objects[o], dg) for o in d["objects"]]
                lo = [min(b[0][i] for b in boxes) for i in range(3)]
                hi = [max(b[1][i] for b in boxes) for i in range(3)]
                if fr == frames[0]:
                    d["size"] = [round(hi[i] - lo[i], 3) for i in range(3)]
                    d["center"] = [round((hi[i] + lo[i]) / 2, 3) for i in range(3)]
                if cam is not None and _in_view(scene, cam, lo, hi):
                    seen[e].append(fr)
        scene.frame_set(keep)
        for e in sorted(set(by_id) | set(ents)):
            d = by_id.get(e)
            exp = ents.get(e, {}).get("class")
            row = {"listed": e in ents, "expected_class": exp,
                   "tagged_classes": sorted(d["classes"]) if d else [], "objects": len(d["objects"]) if d else 0}
            if d:
                row.update(size_m_at_start=d.get("size"), center_at_start=d.get("center"),
                           frames_in_camera=len(seen[e]), first_in_camera=seen[e][0] if seen[e] else None,
                           last_in_camera=seen[e][-1] if seen[e] else None)
                cls = sorted(d["classes"])[0]
                pri = cats.get(cls, {}).get("size_m")
                if pri and d.get("size"):
                    L = max(d["size"])
                    if L < 0.5 * pri[0] or L > 2 * pri[1]:
                        size_warn.append(f"{e} ({cls}) longest side {L:.2f} m vs typical {pri[0]}-{pri[1]} m")
                if exp and exp not in d["classes"]:
                    warns.append(f"{e} listed as {exp!r} but tagged {sorted(d['classes'])}")
                if len(d["classes"]) > 1:
                    warns.append(f"{e} carries several classes {sorted(d['classes'])}")
            elif e in ents:
                warns.append(f"{e} is listed but no object carries it")
            else:
                pass
            if not row["listed"] and ents:
                warns.append(f"{e} is not in input/entities.json")
            report_ents[e] = row
    sp_counts = {}
    for r in rows:
        sp_counts[r["space"] or "-"] = sp_counts.get(r["space"] or "-", 0) + 1
    cams = [o.name for o in scene.objects if o.type == "CAMERA"]
    binds = sorted((m.frame, m.camera.name) for m in scene.timeline_markers if m.camera)
    rep = {"version": VERSION, "frames": [f0, f1], "fps": fps, "renderable_objects": len(rows),
           "tagged_objects": sum(1 for r in rows if r["class"] or r["face_tags"]),
           "untagged_visible": untagged[:40], "untagged_visible_count": len(untagged),
           "unknown_classes": {k: v[:5] for k, v in unknown.items()},
           "classes_used": sorted({r["class"] for r in rows if r["class"]} |
                                  {f["class"] for r in rows for f in (r["face_tags"] or [])}),
           "entities": report_ents, "spaces": sp_counts, "cameras": cams, "camera_markers": binds[:60],
           "size_warnings": size_warn, "warnings": warns}
    if verbose:
        print("PV_CHECK " + json.dumps(rep, ensure_ascii=False), flush=True)
    return rep


__all__ = ["tag", "tag_faces", "tag_space", "pv_check", "resolve", "resolve_all", "face_tags",
           "categories", "entities", "spaces", "valid_class", "VERSION"]
