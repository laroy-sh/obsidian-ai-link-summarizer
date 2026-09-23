# Project notes

## Release authorization

Direct pushes and version tags to `main` are authorized for releases (patch bump → preflight → commit → push → tag → deploy to vault), even though they bypass the required status checks on `main`. Merging PRs with `gh pr merge --admin` is authorized once the diff has been reviewed.
