# Sharing this project across accounts

## Source of truth

Use [Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal), branch `codex/robot-learning-design`. GitHub is the source of record; a specific local folder is not required. The repository was public when checked on November 8, 2026. Read access requires no collaborator invitation; write access still requires authorization.

Code, reports, curated policies, decision streams, and summaries are versioned in Git. Bulk raw outputs and cached public data are available as GitHub Release assets. See the [repository layout and recovery guide](REPOSITORY_LAYOUT.md) for exact locations, checksums, exclusions, and restoration commands.

The original AL-DEV06 repository and the earlier non-Personal fork remain separate historical references. Publish ongoing work to the independent `-Personal` repository.

## Give the other account GitHub access

1. Sign in to GitHub as `Mohnclizzy`.
2. Open the independent repository URL above and go to **Settings → Collaborators** (the
   exact label may appear under **Access**).
3. Invite the GitHub username or email tied to the other account's GitHub login.
   Accept the invitation while signed into that GitHub account.
4. On the other computer or checkout, clone the independent repository and checkout
   `codex/robot-learning-design` to see the current project work.

ChatGPT account access and GitHub account access are separate. Inviting a
second ChatGPT login to a ChatGPT project does not automatically grant GitHub
access, and GitHub access does not transfer ChatGPT chat history. If desired,
also invite the second ChatGPT account to the ChatGPT project so it can use
project chats, uploaded files, and instructions.

## Share generated run artifacts

The current artifact snapshot is now available directly from the private GitHub repository as a release. See [ARTIFACT_HANDOFF.md](ARTIFACT_HANDOFF.md). OneDrive sharing below is an optional way to share subsequent local updates; it is no longer required to obtain this snapshot.

Use the GitHub Release assets linked in [the recovery guide](REPOSITORY_LAYOUT.md). The November 8 consolidation preserves the known local outputs, public data cache, and historical workspace snapshots. OneDrive is no longer required for accessing those project files.

## Recreate a working checkout

```powershell
git clone https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal.git
cd FRI-BCI-ML-RL-Decoder-Personal
git switch codex/robot-learning-design
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The EEG dataset is public and can be downloaded by the documented decoder
commands. Isaac Lab requires its separately pinned runtime described in
`simulation/README.md`; the ordinary Python virtual environment above does not
install Isaac Lab.
