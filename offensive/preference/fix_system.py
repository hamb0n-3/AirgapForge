# fix_system.py  (in-place rewrite)
import json, sys
path = "data/valid.jsonl"
out  = "data/train.valid.jsonl"
with open(path, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
    for line in f:
        r = json.loads(line)
        if "system" not in r or not r["system"]:
            # peel a short system header out of the prompt; keep the rest in prompt
            sysmsg = "You are a security auditor. Identify vulnerabilities, explain exploitability and risk, then propose mitigations and a safe rewrite."
            r = {"system": sysmsg, "prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        w.write(json.dumps(r, ensure_ascii=False)+"\n")
print("wrote", out)

