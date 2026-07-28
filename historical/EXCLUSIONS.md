# Historical archive exclusions

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

The current archive is governed by the “full public release, authorized exceptions” rule. The following V25.8 directories are not copied verbatim because they carry an explicit `LICENSE-PROPRIETARY.md`, duplicate a separately frozen checkout, or mix unrelated worktree state:

| Source path | Disposition | Reason |
| --- | --- | --- |
| `C:/dev/mrg-v25/components/` | excluded | Component trees contain explicit proprietary license notices. |
| `C:/dev/mrg-v25/workspace/` | excluded | Workspace contains an explicit proprietary license notice. |
| `C:/dev/mrg-v25/worktrees/` | excluded | Contains duplicated V26 worktrees and mixed historical checkout state. |

The archive also excludes compiled caches, supplier SDK/binary artifacts, model weights, and any file identified by the release scan as containing a credential. These exclusions are not a statement about technical value; they are authorization and redistribution boundaries.

V26 is copied from its Git-tracked checkout rather than from the duplicate V25 worktree. The commit identifier and copy summary are maintained in the release manifest.
