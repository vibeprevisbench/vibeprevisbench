"""Docker sandbox for the harness workspace: one container per task, the same interface the workspace uses for any
sandbox (ensure, run, read_bytes, write_bytes, reclaim, sandbox_id).

The image (harness/docker/Dockerfile, default tag vibeprevisbench-sandbox:5.2.2) holds Blender 5.2.2 and the frozen
ffmpeg 7.0.2 on PATH; the workspace is /work/blender_agent inside the container. Settings:
  VPB_SANDBOX_IMAGE    image tag (default vibeprevisbench-sandbox:5.2.2)
  VPB_SANDBOX_CPUS     CPU limit of replay containers (default 4, the paper's replay sandbox)
  VPB_SANDBOX_MEMORY   memory limit of replay containers (default 8g, the paper's replay sandbox)
  VPB_SANDBOX_NETWORK  docker network of replay containers (default none: the replay needs no network)
  VPB_DOCKER           docker executable (default docker)
The agent's container (harness/vpb_harness.py prepare) takes its limits from --cpus and --memory instead (default 8 CPUs
and 16g, the paper's agent sandbox) and its network from --network (default bridge: the agent needs a route to its
model API); see docs/harness.md.
"""
import os
import shlex
import subprocess
import uuid

DEFAULTS = {}          # keyword defaults set by the caller (harness/vpb_harness.py: name, network, keep, root)


class Result:
    def __init__(self, exit_code, stdout, stderr):
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr


class DockerHolder:
    def __init__(self, image=None, cpus=None, memory=None, name=None, network=None, keep=None, root=None):
        image, cpus, memory = image or DEFAULTS.get("image"), cpus or DEFAULTS.get("cpus"), memory or DEFAULTS.get("memory")
        name, network = name or DEFAULTS.get("name"), network or DEFAULTS.get("network")
        keep = DEFAULTS.get("keep", False) if keep is None else keep
        self.root = root or DEFAULTS.get("root") or "/work/blender_agent"
        self.image = image or os.environ.get("VPB_SANDBOX_IMAGE", "vibeprevisbench-sandbox:5.2.2")
        self.cpus = str(cpus or os.environ.get("VPB_SANDBOX_CPUS", "4"))
        self.memory = memory or os.environ.get("VPB_SANDBOX_MEMORY", "8g")
        self.network = network or os.environ.get("VPB_SANDBOX_NETWORK", "none")
        self.keep = keep
        self.docker = os.environ.get("VPB_DOCKER", "docker")
        self.name = name or "vpb-" + uuid.uuid4().hex[:12]
        self.sandbox_id = None
        self.create_log = {"image": self.image, "cpus": self.cpus, "memory": self.memory, "network": self.network}

    def _docker(self, args, timeout=None, data=None):
        p = subprocess.run([self.docker] + args, input=data, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def ensure(self):
        if self.sandbox_id is None:
            rc, out, err = self._docker(["run", "-d", "--name", self.name, "--cpus", self.cpus, "--memory", self.memory,
                                         "--network", self.network, "--shm-size", "1g", self.image, "sleep", "infinity"], timeout=300)
            if rc:
                raise RuntimeError("docker run failed: " + err.decode(errors="replace")[-2000:])
            self.sandbox_id = out.decode().strip()
        return self

    def set_timeout(self, *args, **kwargs):           # a local container has no lease to renew
        return None

    def run(self, command, timeout=None):
        self.ensure()
        try:
            rc, out, err = self._docker(["exec", self.sandbox_id, "bash", "-c", command],
                                        timeout=None if timeout is None else max(1, timeout) + 30)
        except subprocess.TimeoutExpired:
            return Result(124, "", "docker exec timed out")
        return Result(rc, out.decode(errors="replace"), err.decode(errors="replace"))

    def read_bytes(self, path):
        self.ensure()
        rc, out, err = self._docker(["exec", self.sandbox_id, "cat", path], timeout=600)
        if rc:
            raise RuntimeError("cannot read %s: %s" % (path, err.decode(errors="replace")[-500:]))
        return out

    def write_bytes(self, path, data):
        self.ensure()
        rc, _, err = self._docker(["exec", "-i", self.sandbox_id, "sh", "-c", "cat > " + shlex.quote(path)],
                                  timeout=600, data=data)
        if rc:
            raise RuntimeError("cannot write %s: %s" % (path, err.decode(errors="replace")[-500:]))

    def reclaim(self):
        if self.sandbox_id is not None and not self.keep:
            self._docker(["rm", "-f", self.sandbox_id], timeout=120)
        self.sandbox_id = None
