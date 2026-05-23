# Restructure Rollback Notes

Created before the source/data restructure on 2026-05-22.

Original top-level layout relevant to this change:

- `conv/src/`: conversation source files.
- `rec/src/`: recommendation source files.
- `conv/data/inspired/`: conversation-specific Inspired data.
- `conv/data/redial/redial/`: conversation-specific ReDial data.
- `data/inspired/`: previous root Inspired data used by recommendation and shared KG loaders.
- `data/redial/redial/`: previous root ReDial data used by recommendation and shared KG loaders.
- `conv/requirements.txt`: legacy conversation requirements file.
- `requirements.txt`: current root requirements file.

Target layout for this change:

- `src/common/`: source shared by conversation and recommendation.
- `src/conv/`: conversation-only source.
- `src/rec/`: recommendation-only source.
- `data/common/`: KG, embeddings, and other data shared by both tasks.
- `data/conv/`: conversation-only data moved out of `conv/data`.
- `data/rec/`: recommendation-only train/pretrain data split out of the old root `data`.

Manual rollback map:

- Move `src/conv/*` back to `conv/src/`.
- Move `src/rec/*` back to `rec/src/`.
- Move `src/common/evaluate_rec.py` back to both `conv/src/evaluate_rec.py` and `rec/src/evaluate_rec.py`.
- Move `src/common/path_utils.py` back to `conv/src/path_utils.py` and `rec/src/path_utils.py`, then restore the old per-task content from git if needed.
- Old copies that were removed from runtime are also kept in `restructure_backup_20260522/` for manual comparison.
- Legacy `conv/requirements.txt`, old Python caches, and `.DS_Store` files were moved under `restructure_backup_20260522/`.
- Move `data/conv/inspired` back to `conv/data/inspired`.
- Move `data/conv/redial` back to `conv/data/redial`.
- Move `data/common/inspired/*` and `data/rec/inspired/*` back into `data/inspired/`.
- Move `data/common/redial/redial/*` and `data/rec/redial/redial/*` back into `data/redial/redial/`.

If git is available and no user edits were made after this restructure, `git diff --name-status` shows the exact file moves/edits to reverse.
