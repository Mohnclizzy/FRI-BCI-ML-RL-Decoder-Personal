#!/usr/bin/env python3
import urllib.request
import json
import os
import sys
import subprocess

TOKEN = None
token_file = os.path.expanduser("~/.github_token")
if os.path.exists(token_file):
    with open(token_file, "r") as f:
        TOKEN = f.read().strip()
if not TOKEN:
    TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

REPO = "Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal"
BRANCH = "codex/robot-learning-design"

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "Antigravity-Agent"
}

def gh_api(endpoint, method="GET", data=None):
    url = f"https://api.github.com/repos/{REPO}/{endpoint}"
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def push_changes(commit_message):
    # Get modified/added files from git
    status_output = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if not status_output:
        print("No changes to push.")
        return

    files_to_add = []
    for line in status_output.splitlines():
        file_path = line[3:].strip()
        if os.path.isdir(file_path):
            for root, _, files in os.walk(file_path):
                for f in files:
                    files_to_add.append(os.path.join(root, f))
        elif os.path.exists(file_path):
            files_to_add.append(file_path)

    tree_entries = []
    for file_path in files_to_add:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        tree_entries.append({
            "path": file_path.replace("\\", "/"),
            "mode": "100644",
            "type": "blob",
            "content": content
        })

    if not tree_entries:
        print("No file modifications found.")
        return

    ref_data = gh_api(f"git/refs/heads/{BRANCH}")
    parent_sha = ref_data["object"]["sha"]
    commit_data = gh_api(f"git/commits/{parent_sha}")
    base_tree_sha = commit_data["tree"]["sha"]

    new_tree = gh_api("git/trees", method="POST", data={
        "base_tree": base_tree_sha,
        "tree": tree_entries
    })

    new_commit = gh_api("git/commits", method="POST", data={
        "message": commit_message,
        "tree": new_tree["sha"],
        "parents": [parent_sha]
    })

    gh_api(f"git/refs/heads/{BRANCH}", method="PATCH", data={"sha": new_commit["sha"]})
    print(f"Pushed to GitHub: {new_commit['sha']} - {commit_message}")
    subprocess.run(["git", "pull", "origin", BRANCH], check=True)

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "Auto update from agent"
    push_changes(msg)
