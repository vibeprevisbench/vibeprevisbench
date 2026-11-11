"""Render the deterministic evaluation orbit for photo tasks using Blender 5.2.2."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data"));p.add_argument("--run",type=Path,required=True)
    p.add_argument("--blender",required=True);p.add_argument("--tasks");p.add_argument("--threads",type=int,default=2)
    a=p.parse_args();want=set(a.tasks.split(",")) if a.tasks else None;failed=[]
    for t in json.loads((a.data/"tasks/index.json").read_text())["tasks"]:
        i=t["task_id"];scene=(a.run/i/"scene.blend").resolve()
        if t["kind"]!="image" or (want is not None and i not in want) or not scene.is_file():continue
        out=(a.run/i/"orbit").resolve()
        if (out/"orbit.mp4").exists():continue
        rc=subprocess.call([a.blender,"-b","--factory-startup","-t",str(a.threads),"--python-exit-code","1",str(scene),
                            "--python",str(Path(__file__).with_name("render_orbit.py")),"--","--out",str(out),"--threads",str(a.threads)])
        if rc or not (out/"orbit.mp4").is_file():failed.append(i)
    print(json.dumps({"failed":failed}));return bool(failed)
if __name__=="__main__":sys.exit(main())
