from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shlex
import re
import shutil
import signal
import subprocess
import tempfile
import time
import gzip
import io
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

from .common import write_json, digest

ROOT = Path(__file__).resolve().parent
OLD_RUNNER = ROOT / "blender_runner.py"


def host_ffmpeg(requested="ffmpeg", portable=False):
    candidates = []
    found = shutil.which(requested)
    if found:
        candidates.append(found)
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    for candidate in dict.fromkeys(candidates):
        try:
            version = subprocess.run([candidate, "-version"], capture_output=True, text=True, timeout=5)
            match = re.search(r"ffmpeg version (?:n)?(\d+)\.(\d+)", version.stdout)
            if version.returncode or not match or tuple(map(int, match.groups())) < (5, 1):
                continue
        except (OSError, subprocess.TimeoutExpired):
            continue
        if not portable:
            return candidate
        # Copying a dynamically linked host executable does not copy its libraries.
        try:
            result = subprocess.run(["ldd", candidate], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        linkage = result.stdout + result.stderr
        if "not a dynamic executable" in linkage or "statically linked" in linkage:
            return candidate
    return None


class Workspace:
    """Docker sandbox (docker_holder.DockerHolder), or explicit local execution."""
    def __init__(self, recorder, deadline, backend="docker", blender="blender", ffmpeg="ffmpeg",
                 rounds=8, fps=24, resolution=(640, 360), final_resolution=None):
        self.recorder, self.deadline = recorder, deadline
        self.backend, self.blender, self.ffmpeg = backend, blender, ffmpeg
        self.max_rounds, self.fps, self.resolution = rounds, fps, resolution
        self.final_resolution = final_resolution or resolution
        self.holder = None
        self.root = None
        self.rounds = 0
        self.tools_called = 0
        self._synced = {}

    def start(self):
        self.recorder.event("environment_start", backend=self.backend)
        if self.backend == "docker":
            from .docker_holder import DockerHolder
            self.holder = DockerHolder()
            self.holder.ensure()
            self.root = self.holder.root
            self.recorder.event("sandbox_created", sandbox_id=self.holder.sandbox_id)
            self._renew_sandbox_lease()
            self._keepalive_stop = threading.Event()
            def keepalive():
                while not self._keepalive_stop.wait(300):
                    try:
                        self._renew_sandbox_lease()
                    except Exception as exc:
                        self.recorder.event("sandbox_keepalive_error", error=str(exc))
            self._keepalive_thread = threading.Thread(target=keepalive, daemon=True)
            self._keepalive_thread.start()
            # Create the output directory before background-job redirection uses it.
            setup = self.holder.run("mkdir -p " + shlex.quote(self.root), timeout=min(30, self.deadline.remaining()))
            if setup.exit_code:
                raise RuntimeError("cannot initialize sandbox workspace: " + setup.stderr)
        else:
            self.root = tempfile.mkdtemp(prefix="blender_agent_")
        self.raw("mkdir -p " + shlex.quote(self.root))
        self.put("runner.py", OLD_RUNNER.read_bytes())
        self.put("launcher.py", (ROOT / "launcher.py").read_bytes())
        self.put("lib/pvtools.py", (ROOT / "pvtools.py").read_bytes())  # imported by runner.py before model code
        # Environment verification before exposing the workspace to the model.
        probe = self.raw("python3 -c " + shlex.quote(
            "import json,shutil; print(json.dumps({k:shutil.which(v) for k,v in "
            + repr({"blender": self.blender, "ffmpeg": self.ffmpeg, "python": "python3", "bash": "bash"}) + ".items()}))"))
        paths = json.loads(probe["stdout"].strip().splitlines()[-1])
        ffmpeg_check = self.raw(shlex.quote(paths["ffmpeg"]) + " -version") if paths.get("ffmpeg") else None
        ffversion = re.search(r"ffmpeg version (?:n)?(\d+)\.(\d+)", ffmpeg_check["stdout"]) if ffmpeg_check else None
        if ffmpeg_check is None or ffmpeg_check["exit_code"] or not ffversion or tuple(map(int, ffversion.groups())) < (5, 1):
            source = host_ffmpeg(self.ffmpeg, portable=bool(self.holder))
            if source:
                if self.holder:
                    compressed = gzip.compress(Path(source).read_bytes(), compresslevel=1)
                    self.put("bin/ffmpeg.gz", compressed)
                    self.raw("python3 -c " + shlex.quote(
                        "import gzip,pathlib; p=pathlib.Path(" + repr(self.root + "/bin/ffmpeg")
                        + "); p.write_bytes(gzip.decompress(p.with_suffix('.gz').read_bytes())); p.chmod(0o755)"))
                    paths["ffmpeg"] = self.root + "/bin/ffmpeg"
                else:
                    paths["ffmpeg"] = source
                self.recorder.event("ffmpeg_provisioned", source_sha256=digest(source), path=paths["ffmpeg"])
            else:
                raise RuntimeError("No working sandbox FFmpeg or portable host FFmpeg found; install imageio-ffmpeg==0.6.0 on the execution host (FFmpeg >=5.1 required).")
        if not all(paths.values()):
            raise RuntimeError(f"missing execution dependencies: {paths}")
        self.blender, self.ffmpeg = paths["blender"], paths["ffmpeg"]
        cfg = {"blender": self.blender, "ffmpeg": self.ffmpeg, "fps": self.fps,
               "resolution": self.resolution, "final_resolution": self.final_resolution, "max_rounds": self.max_rounds,
               "deadline_epoch": None if self.deadline.unlimited else time.time() + self.deadline.remaining()}
        cfg["render_engine"] = "MODEL_CHOICE"
        self.execution_config = cfg
        self.put("execution_config.json", json.dumps(cfg).encode())
        self.put("ledger.json", b'{"rounds":0,"queries":0,"last_source":null}')
        self.raw("mkdir -p " + shlex.quote(self.root + "/bin"))
        self.put("bin/blender", ("#!/bin/bash\nexec python3 -I " + shlex.quote(self.root + "/launcher.py") + ' "$@"\n').encode())
        self.raw("chmod +x " + shlex.quote(self.root + "/bin/blender"))
        versions = self.raw(shlex.quote(self.blender) + " --version && "
                            + shlex.quote(self.ffmpeg) + " -version && python3 --version")
        if versions["exit_code"]:
            raise RuntimeError("execution dependency check failed: " + versions["stderr"])
        frozen = ROOT / "frozen_version.json"
        if frozen.exists():
            version = json.loads(frozen.read_text())
            actual = versions["stdout"].splitlines()[0].removeprefix("Blender ")
            if actual != version["blender"]:
                raise RuntimeError("frozen Blender version mismatch: " + actual)
            write_json(self.recorder.root/"harness_version.json",version)
        else:
            write_json(self.recorder.root/"harness_version.json",{"release_id":"live"})
        smoke_video = self.root + "/.environment_smoke.mp4"
        smoke_frame = self.root + "/.environment_smoke.png"
        encode = self.raw(shlex.join([self.ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                    "color=c=gray:s=32x32:r=24", "-frames:v", "4", "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", smoke_video]))
        decode = self.raw(shlex.join([self.ffmpeg, "-y", "-hide_banner", "-i", smoke_video,
                                    "-frames:v", "1", smoke_frame])) if not encode["exit_code"] else None
        if encode["exit_code"] or decode is None or decode["exit_code"] or "Duration:" not in decode["stderr"]:
            raise RuntimeError("sandbox FFmpeg encode/decode check failed: "
                               + (encode["stderr"] if encode["exit_code"] else decode["stderr"]))
        smoke = {"h264_yuv420p_encode": True, "video_duration_probe": True, "image_decode": True}
        workbench_probe = self.root+'/.environment_workbench.png'
        script=("import bpy,json,importlib.util; "
                + "spec=importlib.util.spec_from_file_location('ba_runtime',"+repr(self.root+'/runner.py')+"); "
                + "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                + "s=bpy.context.scene; m.configure_workbench(s); "
                + "s.render.resolution_x=32; s.render.resolution_y=32; s.render.resolution_percentage=100; "
                + "s.render.image_settings.file_format='PNG'; s.render.filepath="+repr(workbench_probe)+"; "
                + "bpy.ops.render.render(write_still=True); "
                + "print('BA_WORKBENCH_PROBE '+json.dumps({'engine':s.render.engine,'render_aa':s.display.render_aa}))")
        rendered=self.raw(shlex.join([self.blender,'-b','--factory-startup','-t','4','--python-exit-code','1','--python-expr',script]))
        if rendered['exit_code'] or 'BA_WORKBENCH_PROBE ' not in rendered['stdout']:
            raise RuntimeError('sandbox Workbench smoke render failed: '+rendered['stderr'][-2000:]+rendered['stdout'][-2000:])
        details=json.loads(rendered['stdout'].split('BA_WORKBENCH_PROBE ',1)[1].splitlines()[0])
        smoke.update(workbench_single_frame=True,workbench_engine_id=details['engine'],workbench_render_aa=details['render_aa'])
        self.recorder.event("environment_validation", checks=smoke)
        self.raw(shlex.join(["rm", "-f", smoke_video, smoke_frame, workbench_probe]))
        self.recorder.event("environment", backend=self.backend, paths=paths,
                            workspace=self.root, sandbox_id=self.holder.sandbox_id if self.holder else None)
        write_json(self.recorder.root / "environment.json", {"backend": self.backend, "paths": paths,
                   "workspace": self.root, "versions": versions["stdout"], "validation": smoke,
                   "sandbox_lifetime_s": 43200 if self.holder else None,
                   "sandbox_renewal_interval_s": 300 if self.holder else None})
        return self

    def raw(self, command, rendering=False):
        timeout = None if rendering and self.deadline.unlimited else self.deadline.remaining()
        if timeout is not None and self.deadline.unlimited:
            timeout += 120  # Safety guard for ordinary operations only.
        if self.holder:
            # Long commands use background jobs + short polling (old harness pattern).
            stamp = str(time.time_ns())
            base = self.root + "/.cmd_" + stamp
            guard = "" if timeout is None else f"timeout -k 2 {max(1, int(timeout))} "
            body = f"{guard}bash -c {shlex.quote(command)} > {base}.out 2> {base}.err; echo $? > {base}.rc"
            launched = self.holder.run("setsid nohup bash -c " + shlex.quote(body)
                                      + f" </dev/null >/dev/null 2>&1 & echo $! > {base}.pid", timeout=30 if timeout is None else min(30, timeout))
            if launched.exit_code:
                raise RuntimeError("sandbox command launch failed: " + launched.stderr)
            checkpoint = time.monotonic()
            last_delivery_sync = checkpoint
            while True:
                remaining = self.deadline.remaining()
                poll = (f"if [ -f {base}.rc ]; then echo done; cat {base}.rc; "
                        f"elif [ -f {base}.pid ]; then p=$(cat {base}.pid); "
                        "s=$(ps -o stat= -p \"$p\"); "
                        "case \"$s\" in ''|Z*|X*) echo dead;; *) echo running;; esac; "
                        "else echo missing; fi")
                result = self.holder.run(poll, timeout=min(30, remaining))
                if result.exit_code:
                    raise RuntimeError("sandbox command polling failed: " + result.stderr)
                status = result.stdout.strip().splitlines()
                if status and status[0] in ("done", "dead"):
                    return {"exit_code": int(status[1]) if status[0] == "done" else 137,
                            "stdout": self.holder.read_bytes(base + ".out").decode(errors="replace"),
                            "stderr": self.holder.read_bytes(base + ".err").decode(errors="replace")
                                      + ("\nBackground worker exited without completion status." if status[0] == "dead" else "")}
                if not status or status[0] != "running":
                    raise RuntimeError("sandbox command has no valid worker status: " + result.stdout)
                if time.monotonic() - checkpoint >= 20:
                    self.recorder.event("command_wait", command=command, job=base,
                                        elapsed_s=time.monotonic()-checkpoint)
                    checkpoint = time.monotonic()
                if rendering and self.recorder.phase == "delivery" and time.monotonic()-last_delivery_sync >= 60:
                    last_delivery_sync = time.monotonic()
                    try:
                        self.sync()
                        self.recorder.event("delivery_checkpoint")
                    except Exception as exc:
                        self.recorder.event("delivery_checkpoint_error", error=str(exc))
                time.sleep(min(0.5, remaining))
        proc = subprocess.Popen(["bash", "-c", command], cwd=self.root,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise TimeoutError("command exceeded task deadline")
        return {"exit_code": proc.returncode, "stdout": out.decode(errors="replace"), "stderr": err.decode(errors="replace")}

    def path(self, relative):
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("artifact paths must be workspace-relative")
        return self.root + "/" + relative

    def put(self, name, data):
        path = self.path(name)
        if self.holder:
            self.holder.run("mkdir -p " + shlex.quote(str(Path(path).parent)), timeout=min(30, self.deadline.remaining()))
            if len(data) <= 256 * 1024:
                self.holder.write_bytes(path, data)
            else:
                # Existing harness's parallel 256KB chunk-transfer pattern.
                chunks = [(i, data[o:o + 256 * 1024]) for i, o in enumerate(range(0, len(data), 256 * 1024))]
                def upload(item):
                    self.deadline.remaining()
                    i, chunk = item
                    self.holder.write_bytes(f"{path}.part{i:06d}", chunk)
                with ThreadPoolExecutor(8) as pool:
                    list(pool.map(upload, chunks))
                joined = self.holder.run("cat " + shlex.quote(path) + ".part* > " + shlex.quote(path)
                             + " && rm " + shlex.quote(path) + ".part*", timeout=min(30, self.deadline.remaining()))
                if joined.exit_code:
                    raise RuntimeError("chunk assembly failed: " + joined.stderr)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data)

    def get(self, name):
        return self.holder.read_bytes(self.path(name)) if self.holder else Path(self.path(name)).read_bytes()

    @staticmethod
    def is_render_command(command):
        pattern = r"(?:^|[;&|\n])\s*(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*(?:/[^\s]+/)?blender(?:\s|$)([^\n]*)"
        return any(not re.search(r"--purpose\s+query\b|--version\b", match.group(1))
                   for match in re.finditer(pattern, command))

    def command(self, command):
        rendering = self.is_render_command(command)
        prefix = f"cd {shlex.quote(self.root)} && export PATH={shlex.quote(self.root + '/bin')}:$PATH && "
        self.recorder.event("command_start", command=command)
        began = time.monotonic()
        try:
            result = self.raw(prefix + command, rendering=rendering)
        except BaseException as exc:
            self.recorder.event("command_failed", command=command, elapsed_s=time.monotonic()-began,
                                error=str(exc), error_type=type(exc).__name__)
            raise
        self.recorder.event("command_end", command=command, elapsed_s=time.monotonic()-began,
                            exit_code=result["exit_code"])
        return result

    def sync(self):
        if not self.root:
            return
        began = time.monotonic()
        scan = """import os,json
root=ROOT_VALUE
items={}
for base,dirs,files in os.walk(root):
    dirs[:]=[d for d in dirs if d not in ('bin','__pycache__')]
    for name in files:
        path=os.path.join(base,name)
        rel=os.path.relpath(path,root)
        if name.startswith(('.cmd_','.sync_','.environment_')) or '.part' in name or rel=='execution_config.json' or os.path.islink(path):
            continue
        stat=os.stat(path)
        items[rel]=[stat.st_size,stat.st_mtime_ns]
print(json.dumps(items))
""".replace('ROOT_VALUE', repr(self.root))
        listing = self.raw("python3 -I -c " + shlex.quote(scan))
        if listing['exit_code']:
            raise RuntimeError("workspace manifest failed: " + listing['stderr'])
        current = json.loads(listing['stdout'])
        changed = [name for name, stat in current.items() if self._synced.get(name) != stat
                   or not (self.recorder.root/'workspace'/name).exists()]
        for name in set(self._synced)-set(current):
            (self.recorder.root/'workspace'/name).unlink(missing_ok=True)
            if name.startswith(('rounds/','final/')):
                (self.recorder.root/name).unlink(missing_ok=True)
        copied = 0
        if self.holder and changed:
            bundle = self.root+'/.sync_bundle.tar.gz'
            self.put('.sync_inputs.json',json.dumps(changed).encode())
            pack = "import tarfile,json; names=json.load(open(" + repr(self.root+'/.sync_inputs.json') + ")); root=" + repr(self.root) + "; bundle=" + repr(bundle) + "; "
            pack += "t=tarfile.open(bundle,'w:gz',compresslevel=1); [t.add(root+'/'+n,arcname=n,recursive=False) for n in names]; t.close()"
            packed = self.raw("python3 -I -c " + shlex.quote(pack))
            if packed['exit_code']:
                raise RuntimeError("workspace bundle failed: " + packed['stderr'])
            data = self.holder.read_bytes(bundle)
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                for member in archive:
                    self.deadline.remaining()
                    if not member.isfile() or member.name not in changed:
                        raise ValueError("invalid workspace archive member")
                    self.path(member.name)
                    target = self.recorder.root/'workspace'/member.name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_name(target.name+'.sync_tmp')
                    with archive.extractfile(member) as source, temporary.open('wb') as output:
                        shutil.copyfileobj(source,output)
                    temporary.replace(target)
                    copied += member.size
            self.raw("rm -f " + shlex.quote(bundle))
        elif not self.holder:
            for name in changed:
                self.deadline.remaining()
                target=self.recorder.root/'workspace'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(self.path(name),target)
                copied += current[name][0]
        for name in changed:
            if name.startswith(('rounds/','final/')):
                target=self.recorder.root/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(self.recorder.root/'workspace'/name,target)
        self._synced = current
        ledger = self.recorder.root/'workspace/ledger.json'
        if ledger.exists():
            self.rounds = json.loads(ledger.read_text()).get('rounds',0)
        self.recorder.event('workspace_sync', changed_files=len(changed), copied_bytes=copied,
                            elapsed_s=time.monotonic()-began)

    def tool(self, name, args):
        self.tools_called += 1
        began = time.monotonic()
        if name == "bash":
            result = self.command(args["command"])
        elif name == "python":
            self.put(f"scripts/tool_{self.tools_called:04d}.py", args["code"].encode())
            result = self.command(f"python3 scripts/tool_{self.tools_called:04d}.py")
        elif name == "blender":
            self.put(f"scripts/tool_{self.tools_called:04d}.py", args["code"].encode())
            purpose = args.get("purpose", "render")
            if purpose not in ("render", "query"):
                raise ValueError("blender purpose must be render or query")
            command = f"blender --purpose {purpose} --python scripts/tool_{self.tools_called:04d}.py"
            if args.get("blend_file"):
                command += " --blend-file " + shlex.quote(self.path(args["blend_file"]))
            result = self.command(command)
        elif name == "ffmpeg":
            arguments = args["args"]
            if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
                raise ValueError("ffmpeg args must be a list of strings")
            result = self.command(shlex.join([self.ffmpeg, *arguments]))
        else:
            raise ValueError(f"unknown tool: {name}")
        self.sync()
        result["elapsed_s"] = time.monotonic() - began
        result["rounds_used"] = self.rounds
        self.recorder.event("tool", name=name, arguments=args, result=result)
        return result

    def final(self, code, continue_from=None):
        self.execution_config["deadline_epoch"] = None if self.deadline.unlimited else time.time() + self.deadline.remaining()
        self.put("execution_config.json",json.dumps(self.execution_config).encode())
        self.put("final_submission.py", code.encode())
        # Reuse an already verified render only when source and render spec match.
        reuse = """import json,shutil,pathlib,hashlib
root=pathlib.Path(ROOT_VALUE)
ledger=json.loads((root/'ledger.json').read_text())
config=json.loads((root/'execution_config.json').read_text())
source=pathlib.Path(ledger['last_source']).parent if ledger.get('last_source') else None
reused=False
if source and (source/'execution.json').exists():
    info=json.loads((source/'execution.json').read_text())
    job=json.loads((source/'job.json').read_text())
    frames=info.get('frame_range') or []
    matching=(source/'executed.py').read_text()==(root/'final_submission.py').read_text()
    hashes=info.get('artifact_sha256') or {}
    intact=len(hashes)==3 and all((source/name).is_file() and hashlib.sha256((source/name).read_bytes()).hexdigest()==value for name,value in hashes.items())
    if info.get('status')=='ok' and intact and matching and job['res']==config['final_resolution'] and info['fps']==config['fps'] and len(frames)==2 and (source/'preview.mp4').is_file() and (source/'scene.blend').is_file() and len(list((source/'frames').glob('f_*.png')))==frames[1]-frames[0]+1:
        shutil.copytree(source,root/'final',symlinks=True)
        (root/'final/preview.mp4').rename(root/'final/video.mp4')
        (root/'final/solution.py').write_text((root/'final_submission.py').read_text())
        hashes['video.mp4']=hashes.pop('preview.mp4')
        info.update(purpose='final',reused_from=str(source),directory=str(root/'final'),source=str(root/'final/solution.py'))
        (root/'final/execution.json').write_text(json.dumps(info))
        reused=True
print(json.dumps({'reused':reused}))
""".replace('ROOT_VALUE',repr(self.root))
        probe=self.command('python3 -I -c '+shlex.quote(reuse))
        if probe['exit_code']:
            raise RuntimeError('final reuse check failed: '+probe['stderr'])
        if json.loads(probe['stdout'])['reused']:
            result=dict(probe,reused=True)
        else:
            command = "BA_FINAL_RENDER=1 blender --purpose final --python final_submission.py"
            if continue_from:
                command += " --continue-from " + shlex.quote(self.path(continue_from))
            result = self.command(command)
        if continue_from:
            self.raw("rm -rf " + shlex.quote(self.path(continue_from)))
        self.sync()
        self.recorder.event("final_render", result=result)
        return result

    def restore_workspace(self, source, excluded, prefix=""):
        """Restore archived generated files, keeping fresh managed execution files."""
        source = Path(source)
        archive_path = self.recorder.root / "resume_workspace.tar.gz"
        count = 0
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            for path in sorted(source.rglob("*")):
                name = path.relative_to(source)
                if name.parts[0] in excluded or name.parts[0].startswith("."):
                    continue
                if path.is_symlink():
                    raise ValueError("cannot restore workspace symlinks")
                if path.is_file():
                    archive.add(path, arcname=str(Path(prefix)/name), recursive=False)
                    count += 1
        self.recorder.event("resume_workspace_upload", files=count, bytes=archive_path.stat().st_size)
        self.put(".resume_workspace.tar.gz", archive_path.read_bytes())
        script = ("import tarfile,pathlib; root=pathlib.Path(" + repr(self.root) + "); "
                  "archive=root/'.resume_workspace.tar.gz'; "
                  "t=tarfile.open(archive); "
                  "assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() "
                  "and '..' not in pathlib.PurePosixPath(m.name).parts for m in t.getmembers()); "
                  "t.extractall(root,filter='data'); t.close(); archive.unlink()")
        result = self.raw("python3 -I -c " + shlex.quote(script))
        if result["exit_code"]:
            raise RuntimeError("workspace restore failed: " + result["stderr"])
        archive_path.unlink()
        self.recorder.event("resume_workspace_restored", files=count)

    def _renew_sandbox_lease(self):
        self.holder.ensure().set_timeout(43200, request_timeout=30)
        self.recorder.event("sandbox_lease_renewed", sandbox_id=self.holder.sandbox_id,
                            lifetime_s=43200, renewal_interval_s=300)

    def close(self):
        if hasattr(self, "_keepalive_stop"):
            self._keepalive_stop.set()
            self._keepalive_thread.join(timeout=35)
        if self.holder:
            self.holder.reclaim()
            self.holder = None
        elif self.root:
            shutil.rmtree(self.root)
