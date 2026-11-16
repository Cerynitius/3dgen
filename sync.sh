#!/bin/bash
# Pull the working copies into this repo: ~/endless-labyrinth (git, merged as a subtree) and ~/endless-streets (plain
# directory, sources only; tile caches, generated assets and the FastFix3D engine stay out).
set -e
cd "$(dirname "$0")"
git subtree pull -q --prefix=endless-labyrinth ~/endless-labyrinth main -m "Sync endless-labyrinth"
rsync -a --delete --exclude cache/ --exclude assets_cache/ --exclude runs/ --exclude __pycache__/ --exclude '*.log' --exclude 'core.*' \
    ~/endless-streets/ endless-streets/
git status --short
