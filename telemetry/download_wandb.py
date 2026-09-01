#!/usr/bin/env python3
import argparse
import json
import pathlib

import wandb


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project", default="/[SANITIZED]/wandb-project/", help='"entity/project" (entity optional if it is your default)')
    p.add_argument("-o", "--out", default="wandb_export")
    p.add_argument("--artifacts", action="store_true",
                   help="also download artifact contents (can be very large)")
    args = p.parse_args()

    api = wandb.Api()
    root = pathlib.Path(args.out)

    for run in api.runs(args.project):
        d = root / run.id
        d.mkdir(parents=True, exist_ok=True)
        print(f"→ {run.name} ({run.id})")

        # metadata: config, summary, tags, state, url
        dump(d / "run.json", {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "tags": run.tags,
            "notes": run.notes,
            "created_at": run.created_at,
            "config": dict(run.config),
            "summary": dict(run.summary),
        })

        # full, unsampled metric history
        with open(d / "history.jsonl", "w") as f:
            for row in run.scan_history():
                f.write(json.dumps(row, default=str) + "\n")

        # files: output.log (console logs), requirements.txt,
        # wandb-metadata.json, saved code, media, checkpoints saved via wandb.save
        for file in run.files():
            try:
                file.download(root=str(d / "files"), replace=True)
            except Exception as e:
                print(f"   ! skipped file {file.name}: {e}")

        # artifacts
        arts = list(run.logged_artifacts()) + list(run.used_artifacts())
        dump(d / "artifacts.json", [
            {"name": a.name, "type": a.type, "size": a.size, "state": a.state}
            for a in arts
        ])
        if args.artifacts:
            for a in arts:
                try:
                    a.download(root=str(d / "artifacts" / a.type / a.name.replace(":", "_")))
                except Exception as e:
                    print(f"   ! skipped artifact {a.name}: {e}")


if __name__ == "__main__":
    main()