#!/bin/bash
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
WORKING_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git stash -u
git checkout master
rm -rf $REPOSITORY_ROOT/.benchmarks
pytest --benchmark-autosave --benchmark-only performance
git checkout $WORKING_BRANCH
git stash pop
#pytest --benchmark-autosave --benchmark-only --benchmark-compare --benchmark-compare-fail=mean:0.1 performance