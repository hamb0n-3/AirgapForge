import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def load_exploit_testset(limit=None):
    path = os.path.join(config.OFFENSIVE_DATA, "test.jsonl")
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            by_role = {m["role"]: m["content"] for m in msgs}
            out.append({"system": by_role.get("system", ""),
                        "user": by_role.get("user", ""),
                        "reference": by_role.get("assistant", "")})
            if limit and len(out) >= limit:
                break
    return out
