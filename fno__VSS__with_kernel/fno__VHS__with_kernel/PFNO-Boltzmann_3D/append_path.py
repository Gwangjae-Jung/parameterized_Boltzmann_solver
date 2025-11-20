import  sys
from    pathlib     import  Path

for p in map(str, reversed(Path().cwd().parents)):
    if p.rfind("deep_numerical")>=0:
        sys.path.append(p); break