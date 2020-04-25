# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

#!/bin/sh

set -e

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
WORKING_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HAS_UNCOMMITTED_CHANGES="$(git status --porcelain)"

rm -rf "$REPOSITORY_ROOT/test/.benchmarks"
if [ -n "$HAS_UNCOMMITTED_CHANGES" ]; then git stash -u; fi
git checkout master
pytest --benchmark-autosave --benchmark-only --benchmark-timer='time.process_time' --benchmark-warmup='on' --benchmark-warmup-iterations=100 performance
git stash -u && git checkout $WORKING_BRANCH
if [ -n "$HAS_UNCOMMITTED_CHANGES" ]; then git stash pop; fi
git stash pop
pytest --benchmark-autosave --benchmark-only --benchmark-timer='time.process_time' --benchmark-warmup='on' --benchmark-warmup-iterations=100 --benchmark-compare --benchmark-compare-fail=min:5% performance
rm -rf "$REPOSITORY_ROOT/test/.benchmarks"