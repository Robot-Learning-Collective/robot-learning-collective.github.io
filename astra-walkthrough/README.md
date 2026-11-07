# Astra episode walkthrough

Standalone, dark-mode research supplement for the BEHAVIOR 2026 blog. Open
`/astra-walkthrough/` after serving the repository with `python -m http.server 8000`.
No framework, dependencies or build step. Jekyll copies this directory as static
files; the HTML deliberately has no front matter and keeps its own layout.

## Scope and source

This is one `hanging_pictures` rollout, instance 310, rollout 0, episode
`f316a723-8f24-4ea5-a99e-7d32eeca6784`. It is an editorial case study, not a
general-purpose episode viewer or a performance comparison.

The source is the already-public walkthrough at
https://keertimanney.github.io/behavior/ (source commit
`5b38be0efd9e6ad62eddabfc66c83047a7fa0a43`). This package carries its prepared
trace and active page assets; it does not export any additional private portal
data. The discarded design-research memo and older viewer variants are omitted.

`full-data.js` retains 246 activity records, 2,790 evaluator states and references
to 108 observation images. Original IDs and payloads are inspectable. Evaluator
states are displayed separately from information supplied to the agent. Model
summaries are returned text, not a full account of internal reasoning. The last
robot call has an unknown terminal acknowledgement despite evaluator success.

## Structure

- `index.html`, `case.css`: compact title, agent loop, video and narrative,
  then timing and alignment graphics, trace explorer and methods disclosure.
- `agent-loop.js`: four source-linked steps in the orientation correction.
- `case.js`: selected segment, source records, video and URL state.
- `diagrams.js`, `story.js`, `refinements.js`: clock charts, alignment attempts,
  evaluator lanes, trace filtering and observation comparison.
- `media.js`: media host configuration. Images and video are not stored in git.

## Media hosting

Preview currently uses the existing public prototype's media host. To move to
the team's public bucket, change `MEDIA_BASE` in `media.js`, preserving
`media/episode.mp4` and `media/observations/<id>.jpg`. Serve video as `video/mp4`
with range requests supported. Do not use private portal URLs or signed URLs
that expire. The video is checked against its recorded duration before enabling
synchronized playback. A media error leaves the trace readable.

This dependency is intentional for the PR preview; team bucket migration is a
publication follow-up. The walkthrough is not yet linked from the blog homepage.
The BEHAVIOR 2026 article can link to `/astra-walkthrough/` or embed it in an
iframe with a descriptive title and an accompanying full-page link.

## Editorial checks

Open the four segments, select a source, follow its fragment URL, scrub the
video, inspect the clock chart with arrow keys, drag the comparison slider,
filter the trace and open Data and methods. Check at desktop and mobile widths.
Run `node --check` on each JavaScript file after edits.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the proposed Anjana handoff and the
six additional episodes to review. The portal integration is a separate PR.
