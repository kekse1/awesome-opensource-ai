# PR Review Rules

## When to Run

On every agent loop cycle, scan all open pull requests.

## Skip Conditions (do not process if any apply)

- PR already has `agent:reviewed` label → skip entirely
- PR is a draft → skip (wait until it's marked ready)
- PR was opened by an agent → skip (avoid self-review loops)
- PR is from a bot (dependabot, renovate, etc.) → label `needs-human`, skip
- **PR has merge conflicts** (`mergeable = false`) → comment asking author to rebase, apply `needs-info` label, do NOT approve, skip remaining review steps entirely

## Review Steps

For each PR not skipped:

### 1. Understand what the PR does

- Read the PR title and description
- Read the diff carefully - what entries are being added, removed, or changed?

### 2. Structural checks

- Run `python3 tools/validate_awesome.py --skip-remote` against the PR branch before approving or requesting changes
- If the validator reports any errors, request changes and quote the relevant failures clearly
- Does the PR follow the format in `CONTRIBUTING.md`?
- Is the entry placed in the correct section and category?
- Does it use the correct badge format (GitHub stars badge, etc.)?
- Is the link valid? (Check the URL resolves to the right repo)
- Is the description concise and factual (not marketing language)?
- Does it include the project name in bold + link format?

### 3. Content checks (for additions)

**Open source verification:**
- Is the license OSI-approved? Check the repo's LICENSE file directly.
- Is the full code/model publicly available, or is it "open-ish" (API-only, partial weights)? If the latter → request changes, explain the standard.

**Activity check:**
- When was the last commit? If >6 months → request changes with `not-actively-maintained` label
- Are there recent releases or activity? Stars alone are not enough.
- If GitHub auth is available in the runner, prefer full validation with `python3 tools/validate_awesome.py` so star and last-push checks are enforced by script, not only by manual review

**Duplicate check:**
- Search the current README (not just the PR diff) for the project name and GitHub URL
- If duplicate → comment clearly, apply `duplicate` label, request closure

**Category fit:**
- Does it belong in the section it was placed in?
- If it could fit better elsewhere, suggest the right section

**Quality bar:**
- Is this project genuinely notable? Real adoption, useful to the community?
- Avoid listing every possible project - the list should stay curated and high-signal

### Version Replacement Check (for updates/new versions)

When a PR adds a newer version of an already-listed project:

1. **Search README** for existing entries from same org/repo family (e.g., "Qwen", "Gemma", "PyTorch")
2. **Determine relationship:** Is the new version a direct successor or a different variant?
3. **Apply "Current Best" principle:**
   - **Direct successor** (same architecture, just newer) → PR should also *remove* the old version entry
   - **Coexisting warranted** → Only if both serve different use cases or both widely deployed (LTS, major version differences)
   - **Minor bump** (v1.2 → v1.3) → Request changes, not worth a list update

**Action:** If PR adds without removing the superseded entry, request changes with reference to [CONTRIBUTING.md Curation Philosophy](../CONTRIBUTING.md#curation-philosophy-current-best).

### 4. For removals

- Is the reason stated? If not, ask.
- Verify the claim: is the project actually dead/closed-source/abandoned?
- If valid → approve with a note confirming your verification

### 5. For corrections

- Is the correction accurate?
- Does it maintain correct formatting?

### 6. Leave your review comment

Be specific:
- List each issue found with a clear explanation
- Include validator failures when applicable instead of paraphrasing them loosely
- If approving: say exactly why it meets the bar
- If requesting changes: give actionable, friendly guidance

**Do not:** leave vague comments like "looks good" or "needs work" without detail.

### 7. Apply labels and set review status

- Always apply `agent:reviewed`
- Apply `agent:approved` if the PR meets all criteria
- Apply `agent:changes-requested` if changes are needed
- Apply relevant issue labels (`duplicate`, `not-open-source`, etc.) as needed
- Apply `needs-human` for anything that requires maintainer judgment (borderline cases, disputes, etc.)

## Important: Agent approval ≠ merge

An `agent:approved` label means the PR passed automated review. **Only the maintainer merges.** The agent never merges PRs directly.

## Resolution Phase (NEW: Closed-Loop Verification)

After completing review steps above, you MUST take one of these actions. **No PRs should be left pending with just comments.**

### Path 1: Verify & Merge (When ALL criteria confirmed via API)

Before merging, verify these facts directly via GitHub API (do NOT trust PR descriptions):

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Stars | `gh api repos/{owner}/{repo}` | ≥1000 per API |
| Activity | `gh api repos/{owner}/{repo}` | pushed_at within 6 months |
| License | `gh api repos/{owner}/{repo}/contents/LICENSE` | OSI-approved per CONTRIBUTING.md |
| Duplicate | Search README for `{owner}/{repo}` | Not already listed |
| Format | `python3 tools/validate_awesome.py --skip-remote` | No errors |

If ALL checks pass:
1. Label `agent:verified`
2. Run: `gh pr merge --squash --delete-branch`
3. Comment: "Verified via API. Merged."

### Path 2: Fix & Merge (Minor issues only)

If ONLY these issues (no factual errors):
- Wrong category placement
- Badge format incorrect
- Description too long/marketing-heavy
- Missing validator fixes

Then:
1. Push fix commit to PR branch (or use `gh pr edit` for description changes)
2. Re-run validator
3. If clean: `gh pr merge --squash --delete-branch`
4. Comment what was fixed

### Path 3: Close (Hallucination or criteria not met)

If ANY of these found:
- Project doesn't exist (404 from GitHub API)
- Stars < 1000 per API (not what PR claimed)
- License doesn't meet criteria per API
- Last commit > 6 months per API
- Duplicate already exists in README
- Unfixable structural issues

Then:
1. Label `agent:hallucination` or `agent:rejected`
2. Comment with SPECIFIC factual mismatch: "Closing: API shows 847 stars (claimed 1200). License is GPL-2.0 (not OSI-approved per our criteria)."
3. Run: `gh pr close --comment "[reason]. Research loop will retry this category."`

### Path 4: Escalate (Rare - API failure or ambiguity)

Only if:
- GitHub API returns errors/rate limit
- Ambiguous case requiring maintainer judgment
- License unclear even after checking

Then:
- Label `needs-human`
- Comment explaining the ambiguity
- **Do NOT leave pending** - if API unavailable, close with note to retry

## Edge Cases (Updated)

- **PR has merge conflicts:** Close it. Comment: "Closing due to conflicts. Research loop will recreate fresh."
- **Partial hallucination** (3 projects, 1 fake): Remove fake project via commit, merge remaining with comment.
- **API rate limit:** Close with `needs-human` label and note to retry tomorrow.
- **Research PR updates after review started:** Re-verify from scratch.
