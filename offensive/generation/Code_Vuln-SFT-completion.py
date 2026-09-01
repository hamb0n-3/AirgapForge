import json, os
from datasets import load_dataset

SYSTEM = (
    "You are a advanced red team operator. Identify vulnerabilities, explain exploitability "
    "and risk, then provide payload examples and step-by-step exploitation steps ."
)

def build_instruction(vulnerable_code):
    return (
        f"Task: (audit this vulnerability):\n"
        f"\n{vulnerable_code}\n"
    )

def build_response(vulnerability):
    return (
        f"VULNERABILITY: {vulnerability}\n"
        "- <impact/severity>\n\n"
        "EXPLOITATION STEPS:\n"
        "- <step-by-step instruction on how to exploit the bug>\n\n"
        "PAYLOADS:\n"
        "- <Example payloads>\n\n"
    )

def build_system():
    return f"{SYSTEM}"

def main():
    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
    os.makedirs("data", exist_ok=True)
    with open("data/train.jsonl", "w") as f:
        for ex in ds:
            instruction   = build_instruction(ex["rejected"])   # rejected == vulnerable
            response   = build_response(ex["vulnerability"])                      # chosen == fixed
            system = build_system()                      # or ex["rejected"]
            f.write(json.dumps({ "system": system, "instruction": instruction, "response": response}) + "\n")

    # (Optional) create a tiny valid set; here just copy the first 200 lines
    with open("data/train.jsonl") as fin, open("data/valid.jsonl", "w") as fout:
        for i, line in enumerate(fin):
            if i >= 200: break
            fout.write(line)

if __name__ == "__main__":
    main()
