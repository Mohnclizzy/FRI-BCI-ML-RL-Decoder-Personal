# Sharing this project across accounts

## Source of truth

The canonical working copy is the nested `FRI-BCI-ML-RL-Decoder` folder. Its
GitHub `origin` is Mario's independent private repository:
`https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`. It contains the
complete Git history and project branches but is not a GitHub fork. The
original repository is retained as the `upstream` remote:
`https://github.com/AL-DEV06/FRI-BCI-ML-RL-Decoder`. The active branch is
`codex/isaaclab-project-scaffold`, and new project commits should go to `origin`
(the independent repository). The parent `New project 6` folder is a local workspace
wrapper, not the GitHub repository.

An additional private fork named
`https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder` was created earlier in
the setup. GitHub private forks inherit upstream permission rules, so use the
independent `-Personal` repository for sharing and ongoing work.

The GitHub repository contains source, protocols, results summaries, and the
curated project history. It does not contain downloaded EEG data, model
checkpoints, detailed simulation telemetry, virtual environments, or generated
outputs; these are ignored by `.gitignore` to keep the repository manageable.

## Give the other account GitHub access

1. Sign in to GitHub as `Mohnclizzy`.
2. Open the independent private repository URL above and go to **Settings → Collaborators** (the
   exact label may appear under **Access**).
3. Invite the GitHub username or email tied to the other account's GitHub login.
   Accept the invitation while signed into that GitHub account.
4. On the other computer or checkout, clone the personal fork and checkout
   `codex/isaaclab-project-scaffold` to see the current project work.

ChatGPT account access and GitHub account access are separate. Inviting a
second ChatGPT login to a ChatGPT project does not automatically grant GitHub
access, and GitHub access does not transfer ChatGPT chat history. If desired,
also invite the second ChatGPT account to the ChatGPT project so it can use
project chats, uploaded files, and instructions.

## Share generated run artifacts

The detailed run artifacts remain in the local `outputs/` directory. This
checkout lives beneath the Windows OneDrive `Documents/ChatGPT/New project 6`
folder. To make those artifacts available to the other Microsoft account,
share the `FRI-BCI-ML-RL-Decoder` folder (or a narrower `outputs` subfolder)
from the intended OneDrive account with that account's Microsoft email. Confirm
that the other account can open the shared folder before relying on it. Do not
use OneDrive-synced duplicate checkouts as simultaneous working copies of the
same Git branch; use Git commits/pulls to coordinate source changes.

## Recreate a working checkout

```powershell
git clone https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal.git
cd FRI-BCI-ML-RL-Decoder
git switch codex/isaaclab-project-scaffold
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The EEG dataset is public and can be downloaded by the documented decoder
commands. Isaac Lab requires its separately pinned runtime described in
`simulation/README.md`; the ordinary Python virtual environment above does not
install Isaac Lab.
