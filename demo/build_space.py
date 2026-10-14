"""Assemble and upload the Hugging Face Space JaneRoseFerdinant/face-mask-demo.

The Space is a copy of this folder's app plus three example photos from the CC0 dataset:

    app.py             <- demo/app.py (downloads the weights from the model repository at start-up)
    README.md          <- demo/space/README.md (Space configuration + the page text)
    requirements.txt   <- demo/space/requirements.txt (pinned for Python 3.11 on Spaces)
    examples/*.png     <- data/images/ (the three photos named in demo/app.py)

Log in once with `hf auth login` (or set HF_TOKEN in the environment; never put a token on a command line),
then run from the repository root:

    python demo/build_space.py --dry-run     # only assemble the files and list them
    python demo/build_space.py               # create/update the Space and upload
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPACE_ID = "JaneRoseFerdinant/face-mask-demo"

sys.path.insert(0, str(HERE))


def assemble(target: Path) -> list[Path]:
    from app import EXAMPLE_NAMES  # the same three photos the app lists as examples

    shutil.copy2(HERE / "app.py", target / "app.py")
    shutil.copy2(HERE / "space" / "README.md", target / "README.md")
    shutil.copy2(HERE / "space" / "requirements.txt", target / "requirements.txt")
    (target / "examples").mkdir()
    for name in EXAMPLE_NAMES:
        shutil.copy2(REPO / "data" / "images" / name, target / "examples" / name)
    return sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="assemble and list the files, upload nothing")
    parser.add_argument("--space", default=SPACE_ID, help=f"Space id (default: {SPACE_ID})")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        files = assemble(target)
        print("Space contents:", *map(str, files), sep="\n  ")
        if args.dry_run:
            return
        from huggingface_hub import HfApi

        api = HfApi()  # uses the saved login
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
        api.upload_folder(folder_path=str(target), repo_id=args.space, repo_type="space", commit_message="Update Space")
        print(f"Uploaded: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
