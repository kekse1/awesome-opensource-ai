# Labels

Canonical GitHub labels used by agents in this repository.

## Agent Labels

| Label | Color | Description |
|-------|-------|-------------|
| `agent:reviewed` | `#0075ca` | Item was reviewed by an agent - will not be re-reviewed automatically |
| `agent:commented` | `#cfd3d7` | Agent left a comment on this item |
| `agent:approved` | `#0e8a16` | Agent approved this PR (still needs maintainer merge) |
| `agent:changes-requested` | `#e4e669` | Agent requested changes on this PR |
| `agent:suggested` | `#d4edda` | Agent suggested this entry via research loop |
| `agent:verified` | `#0e8a16` | PR verified via GitHub API, facts confirmed |
| `agent:hallucination` | `#b60205` | Research PR contained fake/non-existent project |
| `agent:rejected` | `#b60205` | PR rejected - criteria not met per API verification |

## Status Labels

| Label | Color | Description |
|-------|-------|-------------|
| `needs-human` | `#e11d48` | Needs maintainer attention - agent could not resolve |
| `stale` | `#ededed` | No activity for an extended period |
| `duplicate` | `#cfd3d7` | Already exists in the list |
| `not-open-source` | `#b60205` | Project does not meet open-source criteria |
| `not-actively-maintained` | `#e4e669` | Last commit >6 months ago or project abandoned |
| `needs-info` | `#fbca04` | Waiting on submitter for more information |

## Setup

Agents should ensure all labels above exist in the repo before executing loops.
If a label is missing, create it with the specified color and description via the GitHub API before proceeding.
