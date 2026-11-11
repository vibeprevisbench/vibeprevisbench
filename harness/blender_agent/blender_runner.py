"""Sandbox-only bpy executor; geometry and animation remain model-authored."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

import bpy


def configure_workbench(scene):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.render_aa = "8"


def main():
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    run = Path(job["run_dir"])
    workspace = run.parent.parent if run.parent.name in {"rounds", "queries"} else run.parent
    state = {"status": "running", "stages": [], "blender": bpy.app.version_string}
    if job.get("continue_render"):
        state.update(job.get("source_provenance", {}))
    # Preinstalled tagging module (workspace lib/), imported before model code and the dependency audit.
    os.environ["PV_INPUT_DIR"] = str(workspace / "input")
    sys.path.insert(0, str(workspace / "lib"))
    bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import pvtools
        state.update(pvtools_version=pvtools.VERSION,
                     pvtools_sha256=hashlib.sha256(Path(pvtools.__file__).read_bytes()).hexdigest())
    except Exception as exc:
        state["pvtools_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.dont_write_bytecode = bytecode

    def save():
        temporary = run / "stages.tmp"
        temporary.write_text(json.dumps(state))
        temporary.replace(run / "stages.json")

    def stage(name, action):
        state["active_stage"] = name
        save()
        began = time.monotonic()
        try:
            action()
        except BaseException:
            state["status"] = "failed"
            state["error"] = traceback.format_exc()
            state["stages"].append({"name": name, "sec": time.monotonic()-began, "ok": False})
            save()
            print(state["error"], flush=True)
            raise
        state["stages"].append({"name": name, "sec": time.monotonic()-began, "ok": True})
        save()

    namespace = {"__name__": "__main__", "__file__": job["code_file"], "__builtins__": __builtins__}
    if job.get("blend_file"):
        stage("open_blend", lambda: bpy.ops.wm.open_mainfile(filepath=job["blend_file"]))
    if not job.get("blend_file"):
        configure_workbench(bpy.context.scene)
    if not job.get("continue_render"):
        def execute():
            code = Path(job["code_file"]).read_bytes()
            state["source_sha256"] = hashlib.sha256(code).hexdigest()
            dependencies = {}
            generated = set()
            active = [True]
            def audit(event, args):
                if not active[0] or event != "open" or not isinstance(args[0], (str, bytes)):
                    return
                path = Path(os.fsdecode(args[0])).resolve()
                mode = args[1]
                if (path.suffix != ".py" or workspace not in path.parents or path == Path(job["code_file"])
                        or workspace / "lib" in path.parents):  # preinstalled pvtools is not a script dependency
                    return
                if isinstance(mode, str) and any(flag in mode for flag in "wx"):
                    generated.add(path)
                    return
                if isinstance(mode, str) and "a" in mode:
                    return
                if path in generated:
                    return
                active[0] = False
                try:
                    if path.is_file():
                        dependencies[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
                finally:
                    active[0] = True
            sys.addaudithook(audit)
            try:
                exec(compile(code, "query_code.py" if job.get("blend_file") else "model_code.py", "exec"), namespace)
            finally:
                active[0] = False
                state["script_dependencies"] = dependencies
                value = namespace.get("REPRESENTATION")
                if isinstance(value, str) and value.strip():
                    state["runtime_representation"] = value.strip() + "\n"
                save()
        stage("model_code", execute)
    if not job.get("blend_file") or job.get("continue_render"):
        scene = bpy.context.scene
        resolution, fps = job["res"], int(job["fps"])

        def configure():
            if callable(namespace.get("setup_render")):
                namespace["setup_render"](resolution, fps)
            if job["engine"] != "MODEL_CHOICE":
                raise ValueError("managed rendering requires model engine choice")
            if scene.render.engine not in {"BLENDER_WORKBENCH", "CYCLES"}:
                raise ValueError("choose BLENDER_WORKBENCH or CYCLES; Eevee is not enabled")
            if scene.render.engine == "BLENDER_WORKBENCH":
                configure_workbench(scene)
            scene.render.resolution_x, scene.render.resolution_y = resolution
            scene.render.resolution_percentage = 100
            scene.render.fps, scene.render.fps_base = fps, 1.0
            scene.render.image_settings.file_format = "PNG"
            scene.display.shading.color_type = "MATERIAL"
            gray = bpy.data.materials.new("_BA_DefaultGray")
            gray.diffuse_color = (0.8, 0.8, 0.8, 1)
            for obj in scene.objects:
                if getattr(obj.data, "materials", None) is not None and not len(obj.data.materials):
                    obj.data.materials.append(gray)
            if scene.render.engine == "CYCLES":
                for material in bpy.data.materials:
                    if not material.use_nodes:
                        color = tuple(material.diffuse_color)
                        material.use_nodes = True
                        shader = material.node_tree.nodes.get("Principled BSDF")
                        if shader:
                            shader.inputs["Base Color"].default_value = color
            if scene.camera is None:
                raise ValueError("model did not set scene.camera")
            if scene.frame_start != 1 or scene.frame_end < scene.frame_start:
                raise ValueError("expected a valid frame range starting at 1")
        stage("setup_render", configure)
        state.update(frame_range=[scene.frame_start,scene.frame_end], fps=fps,
                     engine=scene.render.engine, resolution=resolution,
                     render_aa=scene.display.render_aa if scene.render.engine == "BLENDER_WORKBENCH" else None,
                     render_samples=scene.cycles.samples if scene.render.engine == "CYCLES" else None,
                     render_device=scene.cycles.device if scene.render.engine == "CYCLES" else None)
        # Preserve a usable editable project before a slow/failed render.
        stage("save_blend", lambda: bpy.ops.wm.save_as_mainfile(filepath=str(run/"scene.blend"), compress=True, copy=True))
        frames = run / "frames"
        frames.mkdir(exist_ok=True)
        state.update(preserved_frames=0, newly_rendered_frames=0)
        def render():
            for frame in range(scene.frame_start, scene.frame_end+1):
                scene.frame_set(frame)
                scene.render.filepath = str(frames/f"f_{frame:04d}.png")
                if job.get("continue_render") and Path(scene.render.filepath).is_file():
                    state["preserved_frames"] += 1
                else:
                    bpy.ops.render.render(write_still=True)
                    state["newly_rendered_frames"] += 1
                state["last_rendered_frame"] = frame
                save()
        stage("render_all", render)
    state["status"] = "ok"
    state.pop("active_stage",None)
    save()
    print("BA_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        raise SystemExit(1)
