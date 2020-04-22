#!/bin/bash

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
WORKING_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

rm -rf "$REPOSITORY_ROOT/test/.benchmarks"
git stash -u && git checkout master
pytest --benchmark-autosave --benchmark-only performance
git stash -u && git checkout $WORKING_BRANCH
git stash pop && git stash pop
pytest --benchmark-autosave --benchmark-only --benchmark-compare --benchmark-compare-fail=mean:1% performance
rm -rf "$REPOSITORY_ROOT/test/.benchmarks"