# Project page — "vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models"

Anonymous project page for the ICRA 2027 submission. Static, no build step:

    python3 -m http.server -d . 8000

## Layout

    index.html                  the whole page (inline CSS, ~20 lines of JS)
                                order: highlights, abstract, video, explorer, method, robots
    assets/explorer.js          the model x backend explorer
    assets/vla.cpp.mp4          hero overview video (+ -poster.jpg)
    assets/*.png, ur10e.jpg     figures rasterised from the paper's PDFs
    assets/videos/*.mp4         four ALOHA rollouts and two SimplerEnv clips

Total, about 20 MB. Everything is tracked in git.

## Theme

Follows [nerfies.github.io](https://nerfies.github.io/) by way of the sibling `vla.simd` page:
Bulma's neutrals, Noto Sans / Google Sans throughout, dark rounded pill buttons, a 960px column
and 10px radii.

    Dark        #363636   headings, buttons
    Text        #4a4a4a   body
    Muted       #7a7a7a   captions
    Border      #dbdbdb   rules, card edges
    Panel       #f5f5f5   stat tiles, callouts
    Link        #3273dc   links, and every selection affordance

Two extra colours, and the split between them is the whole convention:

    Engine      #3D5A80   `vla.cpp` — slate, always monospace so it never reads as a link
    Ours        #0F766E   the runtime's own measured results

Teal means *we measured this*: the success bar, the `.ours` columns in the explorer, the coverage
cells, the `vla.cpp BF16` tag on the robot cards. Baselines — eager PyTorch, `torch.compile`, the
Ryzen host that the SYCL numbers are measured against — stay neutral. Blue means *you can click
this*. Light only; `color-scheme: light` stops browsers auto-inverting it.

## Model x backend explorer

`assets/explorer.js` holds every number the page states about models and hardware, transcribed
from four places in the paper:

    Table I      eleven-model LIBERO-Object evaluation, and the CUDA / Metal / OpenVINO profiles
    Table III    seven-model device study (RTX 3060, AGX Orin, Orin Nano)
    Table IV     server-side latency against eager and compiled PyTorch, RTX 3090
    Sec. V-B     the backend-coverage paragraph: i9-14900HX CPU requests, Arc A380 through SYCL

Eight devices, eleven models, and deliberately incomplete coverage — 58 of the 88 pairs carry a
number. **Nothing is interpolated.** A pair the paper does not report renders as "not measured",
and the two Orin Nano checkpoints that do not load render as "out of memory". The coverage matrix
above the table exists to make that shape visible rather than hide it: the three experiments were
run on different model subsets with different episode counts and execution horizons, so the
columns are not directly comparable and the page says so.

Per-device column sets live in `COLS`, the latency key that drives the bar and the sort order in
`LAT`, and the data in `MODELS`. To add a device, append to `DEVICES`, give it a `COLS` entry and
a `LAT` key, and add its measurements to the models that have them.

Two things the app is careful about:

- **Success and latency come from different experiments.** The left pane shows the 100-episode
  LIBERO-Object rate with its Wilson interval; the `of 200` column is the separate device study.
  They are labelled separately and never mixed into one number.
- **Reference rates are not matched comparisons.** The reference tick is a published rate except
  for pi0 and the three GR00T checkpoints, where it is the paper's own PyTorch reproduction. The
  pane says which.

## Videos

Seven in total, all tracked.

`assets/vla.cpp.mp4` is the hero overview, placed after the link buttons the way nerfies places
its teaser. It is 1:50 **with an audio track**, so it does not autoplay — browsers block autoplay
with sound. It gets controls and a poster frame (`assets/vla.cpp-poster.jpg`, `ffmpeg -ss 30`).

The four ALOHA clips are the external overview camera, GR00T-N1.6 in BF16 on an AGX Orin,
re-encoded at 3x speed and 640px wide with `setpts=PTS/3`. Two successes and two failures: 35 of
the 40 trials succeed, and the page shows the minority case rather than cutting it.

    aloha-task1-pass-3x.mp4     task 1, setup 1, trial 2   PASS 0m47s
    aloha-task2-pass-3x.mp4     task 2, setup 1, trial 1   PASS 0m24s
    aloha-task1-fail-3x.mp4     task 1, setup 5, trial 2   FAIL timeout/retry
    aloha-task2-fail-3x.mp4     task 2, setup 5, trial 4   FAIL complete

The originals are in `vla-cpp.github.io/src/assets/aloha-videos/vlacpp/<task>/<setup>/<trial>/`,
240 clips over three cameras and 1.1 GB, which is why only four are here. That tree also has a
`pytorch/` sibling, but `aloha-evidence.js` carries no rows for it and the paper reports no
PyTorch ALOHA numbers, so there is no side-by-side to make.

The two SimplerEnv clips are 256x256 and about two seconds, silent, autoplaying.

    simplerenv-carrot-plate.mp4      WidowX, put carrot on plate    (paper: 80% for GR00T-N1.6)
    simplerenv-eggplant-basket.mp4   WidowX, put eggplant in basket (no rate in the paper)

**These were `libero-videos/episode-00000{3,6}` on the old page, labelled "LIBERO rollout".**
They are not LIBERO: both show a WidowX arm in the SimplerEnv / Bridge scenes, one at a wooden
table with a carrot and plate, one at a sink with an eggplant and a yellow basket. They were
renamed here and captioned for what they are. Nothing on the page claims a policy for them,
because the asset carries no policy label.

A placeholder card upgrades itself when its file appears, so more footage needs no HTML edit —
drop the `.mp4` in and give the card the right filename.

## Anonymity

No author names, affiliations, institution, repository link or contact address;
`<meta name="robots" content="noindex">` keeps it out of search indexes. The figure PNGs are
rasterised, so they carry no PDF metadata. Every video was re-muxed with `-map_metadata -1`; the
hero video arrived carrying a `Made with Remotion 4.0.525` comment, which is gone.

Two things in the hero video are worth a look before this goes anywhere public, because neither
matches the paper:

- it closes on **"Twelve Architectures"**, where the paper says eleven models throughout;
- it claims to be **"the first native flow-matching and diffusion action-head server on
  llama.cpp"**, a priority claim the paper does not make.

Re-render or trim the video if either should go.

To de-anonymise, the places to edit are the two `.btn.soon` buttons in `<nav class="links">` and
the footer.

## Adding the paper PDF

Put the compiled, anonymised PDF at `assets/paper.pdf`, then replace

    <a class="btn soon" href="#"><span class="ic">▣</span> Paper</a>

with

    <a class="btn" href="assets/paper.pdf"><span class="ic">▣</span> Paper</a>

The Code button works the same way — point it at an `anonymous.4open.science` mirror, or at the
real repository once the submission is no longer blind.

## Regenerating a figure

    pdftoppm -png -r 300 -singlefile <name>.pdf assets/<name>

from `overleaf/vla.cpp/ICRA/`. `vla-cpp-diagram.png`, `roofline.png`, `replan-sweep.png` and
`ur10e.jpg` came across already rasterised from the previous page.
