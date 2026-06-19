<!-- markdownlint-disable-file MD041 -->
<!-- TITLE CONVENTION (per bf-docs/AGENTS.md → "PR title convention"):
       [Type][Area] short subject in present-tense imperative

     Examples:
       [Feat][CI] add bf-pr-review workflow
       [Fix][bf-tools] handle empty .bf-paths gracefully
       [Docs][ADR-0004] clarify channel→tag mapping
       [bf-patch][CODEOWNERS] route bf-docs/ to core owners

     `[bf-patch]` is the canonical type for commits editing upstream
     files (per ADR-0003); pair it with the touched-file area. -->

## Summary

<!-- One paragraph: what does this change and why? Link related ADRs if relevant. -->

## Test plan

<!-- How did you verify? Examples:
       - CI lint + cpu-smoke passed
       - Pulled :dev-pr-<N>-<sha> image and ran `--model <model>` smoke
       - Deployed to staging engine deployment; benchmark sweep within tolerance
       - N/A (docs-only change) -->

## Classification

<!-- `bf-classification-lint` auto-labels this PR as `bf-addition` or `bf-patches` (per ADR-0003).
     If the lint flags it as mixed: split into two PRs.

     For `[bf-patch]` commits (patches to upstream files), confirm each commit has:
       - Subject prefix `[bf-patch]`
       - `Upstream-status:` trailer set to one of: candidate | submitted-#NNNN |
         pending-upstream-#NNNN | rejected | bf-only
       - `Rationale:` trailer if status is `bf-only` or `rejected`
       - `Upstream-original-sha:` trailer if status is `pending-upstream-#NNNN` -->

## References

<!-- Related ADRs (e.g., bf-docs/adr/0003-…), tracked issues, upstream PRs. -->
