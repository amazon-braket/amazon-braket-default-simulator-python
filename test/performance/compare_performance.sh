#!/bin/bash

set -e

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
WORKING_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HAS_UNCOMMITTED_CHANGES="$(git status --porcelain)"

rm -rf "$REPOSITORY_ROOT/test/.benchmarks"
if [[ $HAS_UNCOMMITTED_CHANGES ]]; then git stash -u; fi
git checkout master
pytest --benchmark-autosave --benchmark-only performance
git stash -u && git checkout $WORKING_BRANCH
if [[ $HAS_UNCOMMITTED_CHANGES ]]; then git stash pop; fi
git stash pop
pytest --benchmark-autosave --benchmark-only --benchmark-compare --benchmark-compare-fail=mean:0.001 performance
rm -rf "$REPOSITORY_ROOT/test/.benchmarks"