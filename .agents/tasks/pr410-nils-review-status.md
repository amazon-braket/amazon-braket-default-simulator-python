# PR 410 review status for Nils's comments

Verdict: 4 of 5 comments are fixed in code at head `cb7e203`. 1 is open (the test rewrite scope question). None of the five replies are visible to Nils yet, because all of them sit in an unsubmitted (PENDING) review.

Reviewer: Nils Quetschlich (`nilsquet`), one review submitted 2026-10-06 with 5 inline threads. No issue-level comments from Nils. No thread is marked resolved on GitHub.

## Pending replies

The author replies to Nils (comment ids 4200049229, 4200146053, 4200158066, 4200173458, 4200174251) all belong to pending review 5434124123 by `yitchen-tim`. GraphQL reports `state=PENDING` for each, and the REST endpoint returns 404 for them. Nils sees five threads with no response until that review is submitted.

## Per-comment status

| # | file:line | Nils's comment | Status | Evidence |
|---|---|---|---|---|
| 1 | circuit.py:142 (outdated) | [Rename `elements` in `add_measure` to something more explicit](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4197045742) | Addressed | Commit `9455208` renames it to `register_indices`. See `circuit.py:134` and the docstring at `circuit.py:142-145`. Pending reply: "changed to register_indices". |
| 2 | circuit.py:158 (outdated) | [`zip()` silently truncates when `target` and `elements` lengths differ. Validate, raise `ValueError`, add a test](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4197445591) | Addressed | Commit `ddb881f` adds the length check and `ValueError` at `circuit.py:158-162`. Test `test_add_measure_rejects_register_indices_length_mismatch` at `test_circuit.py:74-93` covers 2 vs 1, 1 vs 2 and 2 vs 0, and asserts nothing was recorded. |
| 3 | circuit.py:13 (anchor only, the comment is about the PR description) | [Overview describes the #410 + #411 stack. Move behavior details to #411 and describe #410 as groundwork](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4197463206) | Addressed | The PR body was edited 2026-10-06 19:28 UTC. It now opens with "This is groundwork only: measurements are not routed through declared registers yet". It no longer mentions `simulation_path.py`/`simulator.py`, and it states that measurements still go through `add_measure_by_index` into the anonymous register. Pending reply: "fixed". |
| 4 | test_program_context.py:181 (outdated) | [Avoid test classes](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4197483389) | Addressed | Commit `cb7e203` replaces `class TestRegisterDeclaration` with four module-level functions (`test_program_context.py:181-224`). The PR adds no other test classes to `test_circuit.py` or `test_program_context.py`. Pending reply: "fixed". |
| 5 | test_interpreter.py:2207 | [Are the broad test rewrites in `test_mcm.py`/`test_interpreter.py` needed for #410, or prep for #411's single-register restriction? If prep, move them to #411 and keep the original scalar/multi-register coverage in #410](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4197556994) | Not addressed (reply drafted, not posted, and only partly answers) | No code change. The pending reply says the rewrites correct the implicit measure-all behavior and that the author wants to avoid patching or preserving it. It does not say whether the `test_mcm.py` rewrites are needed for #410, and it does not respond to "move them to #411". Details below. |

## Evidence for comment 5

I ran the PR head's `test_mcm.py` and `test_interpreter.py` against the `main` source:

- All rewritten `test_mcm.py` tests pass on `main`. This matches the PR body ("The rewritten tests also pass on `main`. They prepare for #411"). So the `test_mcm.py` rewrites are not required by #410. They prepare for #411, which is the case Nils asked to move.
- 3 `test_interpreter.py` tests fail on `main`: `test_measure_qubit_twice_allowed` and two `test_measurement` cases. These are the `measured_qubits` ordering updates, and #410 does require them. Changing `bit[1] b;` to `bit[3] b;` also fixes an out-of-range program. The `test_interpreter.py` side of the question therefore has a valid answer: those changes belong in #410.
- Coverage loss in #410: the diff removes about 50 `bit b;`, 25 `bit result;` and the `b0`/`b1`/`b2` multi-register declarations from `test_mcm.py`, mostly replacing them with `bit[2] b;`. The author's earlier self-comment ([r4185750151](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4185750151)) says partial measurement is covered by `test_scalar_bit_mcm_reports_only_the_measured_qubit`. That test exists only in #411 (`test_mcm.py:2616` on `registers-2-route`), not in #410. If #410 merges alone, scalar-bit and multi-register MCM coverage is reduced for a period.

## Test run on PR head

`pytest test/unit_tests --cov` on `cb7e203` gave 1229 passed and 71 xfailed, with 100% total coverage (`circuit.py` and `program_context.py` both at 100%).

## Items needing action

1. Submit pending review 5434124123. Without it, Nils sees no response on any thread, including the four that are fixed.
2. Comment 5: before submitting, rework the reply so it answers Nils directly. Suggested points:
   - The `test_interpreter.py` ordering changes are required by #410 because `measured_qubits` now follows classical index order.
   - The `test_mcm.py` rewrites are not required by #410. They pass on `main` and prepare for #411.
   - Then either move the `test_mcm.py` rewrites (and the `bit[1]`/`bit[3]` style changes that exist only for #411) into #411 and restore the original scalar and multi-register cases in #410, or keep them in #410 and say why. If you keep them, also add `test_scalar_bit_mcm_reports_only_the_measured_qubit` (or an equivalent scalar-bit MCM test) to #410 so the coverage your earlier comment cites exists in this PR.
3. Optional follow-up on comment 1: `register_indices` is the name only in `add_measure`. `ClassicalRegister.bind(element, ...)`, `clear_measurement(register, element)` and the `measurement_slots` tuple `(register, element, qubit)` still say `element`. Nils's nit targeted `add_measure`, so this is not blocking, but aligning the names would avoid two terms for one concept.
4. Optional heads-up: #411 has a test class in `test_program_context.py` (around line 270 on `registers-2-route`). Nils will likely repeat the "avoid test classes" nit there.

## Other reviewers

No other reviewers have commented. The only other threads are three unresolved self-comments by the author (`yitchen-tim`) that explain changes ([r4185750151](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4185750151), [r4185778435](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4185778435), [r4187572227](https://github.com/amazon-braket/amazon-braket-default-simulator-python/pull/410#discussion_r4187572227)). They need no action. The Codecov bot comment refers to the first commit `861817b` and is stale.
