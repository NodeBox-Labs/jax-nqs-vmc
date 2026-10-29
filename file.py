import os

BASE = r"C:\Users\nidha\Desktop\github reposistory\jax-nqs-vmc\.github"

files = {
    # Workflows
    "workflows/ci.yml": "# tests + lint + types (PR + push)\n",
    "workflows/codeql.yml": "# security static analysis (weekly + PR)\n",
    "workflows/docs.yml": "# build API docs (future MkDocs)\n",
    "workflows/nightly.yml": "# slow tests + Windows + GPU nightly\n",
    "workflows/release.yml": "# publish to PyPI on tag\n",
    "workflows/stale.yml": "# auto-close stale issues/PRs\n",
    "workflows/scorecard.yml": "# OpenSSF supply-chain score\n",
    "workflows/benchmark.yml": "# (Phase 3) perf benchmarks on PRs\n",

    # Issue templates
    "ISSUE_TEMPLATE/bug_report.yml": "# Bug report template\n",
    "ISSUE_TEMPLATE/feature_request.yml": "# Feature request template\n",
    "ISSUE_TEMPLATE/config.yml": "# Issue template config\n",

    # Root-level files
    "dependabot.yml": "# Dependabot configuration\n",
    "pull_request_template.md": "# Pull request template\n",
}

for rel_path, content in files.items():
    path = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):  # don't overwrite existing files
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {path}")
    else:
        print(f"Skipped (already exists): {path}")

print("\nDone. Structure created successfully.")