# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
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

# Remove any old data from previous executions
rm -rf "$REPOSITORY_ROOT/test/.benchmarks"
# Stash uncommitted changes (if any)
if [ -n "$HAS_UNCOMMITTED_CHANGES" ]; then git stash -u; fi
git checkout main
# Execute performance tests against the latest commit on the main branch
pytest --benchmark-autosave --benchmark-only --benchmark-timer='time.process_time' --benchmark-warmup='on' --benchmark-warmup-iterations=100 performance
# Stash the performance test execution results and switch back to the working branch
git stash -u && git checkout $WORKING_BRANCH
# Retrieve the stashed performance test execution results for the main branch (to compare against)
git stash pop
# Retrieve the uncommitted changes which had been stashed (if any)
if [ -n "$HAS_UNCOMMITTED_CHANGES" ]; then git stash pop; fi
# Execute the performance tests and compare the results against the main branch.
# Fails if there >= 5% increase in the minimum execution time of any test
pytest --benchmark-autosave --benchmark-only --benchmark-timer='time.process_time' --benchmark-warmup='on' --benchmark-warmup-iterations=100 --benchmark-compare --benchmark-compare-fail=min:5% performance
# Cleanup benchmark metadata
rm -rf "$REPOSITORY_ROOT/test/.benchmarks"
