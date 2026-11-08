# Project page — "vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models"

Project page for vla.cpp, linking the arXiv paper and the code. Static, no build step:

    python3 -m http.server -d . 8000

## Layout

    index.html                  the whole page (inline CSS, ~20 lines of JS)
                                order: teaser, backend chips, abstract, video, explorer,
                                architecture, simulation, robots, analysis
    assets/explorer.js          the model x backend explorer
    assets/roofline.js          the roofline, drawn in the page from counted FLOPs and bytes
    assets/horizon.js           the execution-horizon sweep, drawn in the page from the paper's table
    assets/teaser.svg           the teaser: the paper's Fig. 1, updated to the current project
    figures/                    the teaser's TikZ source and build script
    figures/roofline/           how the roofline's FLOPs and bytes were counted
    assets/vla.cpp.mp4          hero overview video (+ -poster.jpg)
    assets/ur10e.jpg            the UR10e figure, rasterised from the paper's PDF
    assets/arch/                the architecture's logos and thumbnails (icons are a sprite in index.html)
    assets/videos/*.mp4         four ALOHA rollouts, two LIBERO-Object and two SimplerEnv clips

Total, about 20 MB. Everything is tracked in git.

## Theme

Follows [nerfies.github.io](https://nerfies.github.io/) by way of the sibling `vla.simd` page:
Bulma's neutrals, Noto Sans / Google Sans throughout, dark rounded pill buttons, a 960px column
and 10px radii.

    Dark        #363636   headings, buttons
    Text        #4a4a4a   body
    Muted       #7a7a7a   captions
    Border      #dbdbdb   rules, card edges
    Panel       #f5f5f5   callouts, explorer matrix
    Link        #3273dc   links, and every selection affordance

Two extra colours, and the split between them is the whole convention:

    Engine      #3D5A80   `vla.cpp` — slate, always monospace so it never reads as a link
    Ours        #0F766E   the runtime's own measured results

Teal means *we measured this*: the success bar, the `.ours` columns in the explorer, the coverage
cells, the `vla.cpp BF16` tag on the robot cards. Baselines — eager PyTorch, `torch.compile`, the
Ryzen host that the SYCL numbers are measured against — stay neutral. Blue means *you can click
this*. Light only; `color-scheme: light` stops browsers auto-inverting it.

## Paper vs. current project

The page describes the project as it is now, not the submitted paper; the `.notice` under the
header says so. Where the project has newer information, the page uses it:

- thirteen models and seven backends (ROCm and Hexagon added) in the meta description, hero,
  abstract, teaser panel (a) and the architecture caption;
- teaser panel (b) plots the latest 200-episode LIBERO-Object run (`docs/benchmark/libero.md`)
  against the paper's reference rates, which puts SmolVLA, GR00T-N1.6 and π0 outside the ±4 pp
  band (the paper had only N1.6 outside). Octo-Small and TurboVLA have no reference rate;
- the explorer (below);
- the roofline (below) is redrawn from the explorer's latencies and a fresh FLOP and byte count;
- the architecture is redrawn in the page (see *Architecture* below) and shows all seven backends;
- the robot section has a third embodiment the paper does not: π0.5 and GR00T-N1.7 on an SO-101
  arm, served from an Intel AI board (subsection C);
- the simulation section has a third benchmark the paper does not: a RoboCasa clip.

Everything else has nothing newer and still comes from the paper: teaser panels (c) and (d), the
abstract's BitVLA results (200/200 on the Orin Nano, 1.3 GiB, 4.0–4.6×), the SmolVLA case study,
SimplerEnv, and the UR10e and ALOHA subsections.

## Roofline

The roofline sits in the Analysis section at the end of the page, as subsection B after the
execution horizon, so the two cost discussions read together. `assets/roofline.js` draws one panel
per device, RTX 3060 and AGX Orin, as inline SVG so the labels use the page's fonts. It plots the ten models whose
multiplies run in BF16 (see below). Points are teal like every other measured number; labels
carry only the model name, and hovering or tabbing to a point shows its numbers. Labels are
placed in the browser, after measuring them, so they avoid points, each other and the roof; in
crowded spots a thin leader joins label and point.

Achieved throughput is one `predict()`'s FLOPs over its p50 latency at the defaults, copied from
the explorer. The ceilings are the paper's: measured BF16 GEMM rates of 25.6 and 21.3 TFLOP/s,
nominal bandwidths of 360 and 204.8 GB/s.

**The paper's FLOP and byte estimates do not carry over.** Its figure implies 2.7–2.8 TFLOP per
call for each GR00T model; at today's latencies that would put them above the compute ceiling.
So the work is counted instead, from the runtime itself:

- `ggml-count.patch` adds a counter to ggml's `ggml_backend_graph_compute_async` (llama.cpp
  `b11223`, the runtime's pinned tag). FLOPs are 2·M·N·K for every `MUL_MAT`, plus QKᵀ and PV for
  every `FLASH_ATTN_EXT`. Bytes are the operands of those ops that no op in the graph produced:
  weights at their resident dtype, counted at every use (so an action head's weights count once
  per solver step), plus cached keys and values. Activations are not counted, so intensities are
  upper bounds, by the same convention as the paper's (a BF16 layer over N tokens scores N).
- `stub.py` writes a sparse stand-in for a checkpoint from its header alone, so no weights need
  downloading. `count.sh` runs `vla-bench` on the CPU backend at each model's benchmark views,
  input size and prompt length, once with one call and once with two; `tally.py` takes the
  difference. The CPU backend keeps every op inside ggml, where the counter sees it.

**Only BF16 models are plotted.** The roof is a BF16 GEMM rate, which is the ceiling only for
multiplies that run in BF16 on the tensor cores: the ten models with BF16 resident weights.
Three are left off because a BF16 roof is not their ceiling:

- **BitVLA** multiplies int8 activations by unpacked ternary weights on the int8 tensor cores
  (`mma.sync.m16n8k32` and `wmma`, in `src/kernels/bitvla/`). Against the BF16 roof it lands at
  107% on the RTX 3060 and 99% on AGX Orin. That is not a counting error, but it reads as one;
  the int8 tensor-core rate is a separate, higher ceiling, so BitVLA stays off this plot.
- **Octo-Small** and **TurboVLA** keep their F32 weights as shipped and multiply in F32, whose
  ceiling is lower than the roof drawn.

Two models are counted by hand under the same rules:

- **OpenVLA-OFT**: its 14 GB of BF16 weights do not fit the 15 GB host that ran the counter. Its
  sequence through `vla-bench` is 330 tokens (16 prompt, 256 image, 1 proprio, 56 action, 1
  stop). With a 7B LM and the dual vision tower, that gives 4737 GFLOP and 14.81 GB. As a check,
  counting VLA-Adapter by hand the same way, with the same vision towers, comes within 1% of the
  counter's FLOPs and matches its bytes exactly.
- **BitVLA** (not plotted): its packed ternary weights run only in its own CUDA kernels, outside
  ggml. Its layout is OFT's with the BitNet LM, 330 tokens. That gives 1626 GFLOP: the LM 1409,
  the vision tower 211, the projector 5 and the head 1. The bytes are the 0.92 GB it holds
  resident: 2-bit weights plus an F32 head and projector.

| Model | GFLOP | GB | FLOP/byte | RTX 3060 | AGX Orin |
|---|--:|--:|--:|--:|--:|
| GR00T N1.7 | 621.9 | 9.823 | 63 | 33% | 51% |
| GR00T N1.6 | 703.1 | 9.462 | 74 | 35% | 45% |
| VLA-JEPA | 528.8 | 4.568 | 116 | 36% | 41% |
| GR00T N1.5 | 863.2 | 6.013 | 144 | 39% | 41% |
| Evo-1 | 1492.8 | 7.981 | 187 | 25% | 23% |
| SmolVLA | 650.1 | 2.912 | 223 | 20% | 18% |
| π0 | 2821.2 | 11.763 | 240 | 48% | 49% |
| π0.5 | 2810.9 | 11.636 | 242 | 48% | 49% |
| VLA-Adapter | 660.3 | 2.506 | 263 | 34% | 39% |
| OpenVLA-OFT | 4737.2 | 14.807 | 320 | — | 75% |
| *Not plotted* | | | | | |
| Octo-Small (F32) | 18.8 | 0.231 | 82 | | |
| TurboVLA (F32) | 102.1 | 0.429 | 238 | | |
| BitVLA (int8) | 1625.7 | 0.922 | 1763 | | |

The percentages are achieved throughput as a share of the BF16 roof at that intensity. The GR00T
models land at 63–144 FLOP/byte, not the paper's 285, because their action heads re-read 1–2 GB
of weights at every one of four solver steps. On AGX Orin, N1.6 and N1.7 sit left of the ridge.

The counted source is the runtime repository's head on 2026-10-29, a few weeks newer than the
commits the latencies were measured at. A graph change in between would shift a point; that was not
checked.

## Model x backend explorer

`assets/explorer.js` holds the runtime's latest benchmark round, not the paper's tables. The two
disagree cell for cell: the round postdates the submission, adds Octo-Small and TurboVLA, and
times `predict()` in process with `vla-bench` (synthetic inputs, 3 warmups + 20 timed calls,
best of three processes) where the paper measures requests through the server. The header's
notice and the line under the section heading cover the mismatch.

It is copied from the runtime repository's docs, never retyped:

    docs/benchmark/<device>.md   latency and memory at the defaults ("Latency and memory") and
                                 the fastest flag set ("Fastest configuration"), 22 device and
                                 backend rows; the reasons a pair has no number ("Not run")
    docs/benchmark/libero.md     LIBERO-Object successes out of 200 per model (RTX 3090,
                                 defaults), the replay counts, BitVLA's and GR00T N1.7's four suites
    docs/backend/{rocm,ov}.md,   fixed-input checks: max |delta| of the action chunk against a
    hexagon{,-windows}.md        CPU reference, per model, for the RX 9060 XT, the three OpenVINO
                                 targets and the two Hexagon NPUs

Fourteen models on 22 rows: 260 of the 308 pairs carry a number. **Nothing is interpolated.**
Every other pair carries the report's reason, as one of four kinds: no path on that backend
(BitVLA off CUDA, Octo-Small on OpenVINO), out of memory, not attempted because the weights
exceed the memory, or not run (ACT is measured only on the Dragonwing IQ-10, and has no LIBERO
success rate).

The matrix is the navigation: devices down, models across, each measured cell shaded by its p50
at the defaults on one teal ramp, six log-spaced steps from under 30 ms to over 3 s, darker is
faster. p50 is the headline everywhere (cells, sort, bar, verdict) because min catches boost
power on the power-limited CPUs. The RTX 5090 row is the exception: it is the PR author's
numbers, min only, shown in italics.

In the source, `R` and `T` (defaults and fastest flags) are generated from the reports' tables;
`MODELS`, `DEVICES`, `WHY`/`SKIP` (missing pairs), `CHECK` (fixed-input checks) and `PAIR`
(per-pair notes) are written by hand from the same docs. Before publishing a regenerated `R`,
check its `min` column against the summary table in `docs/benchmark/README.md`. In this round
all 308 cells agree.

Things the app is careful about:

- **Success and latency come from different runs.** Success is measured once, on the RTX 3090
  at the defaults, through the server and the simulator; the pane says so and never pairs it
  with another device's latency as if it were measured there.
- **Fast flags change the numerics.** The tuned column and the "fastest flags" block note that
  the success rate does not carry over to them. On the Jetson AGX Orin the defaults run the
  bit-exact CUDA path, which is why most flags are slower there.
- **A latency is not a correctness check.** Where the backend notes measured one, the pane
  shows the fixed-input difference and flags anything above the 2.9e-3 bar those notes use;
  the Snapdragon X Hexagon checks predate the latency pin, and the pane says that too.

The explorer's data names no one: no commit hashes, PR numbers, branch names or Hugging Face
organisation.

## Videos

Twelve in total, all tracked.

`assets/vla.cpp.mp4` is the hero overview, placed after the link buttons the way nerfies places
its teaser. It is 1:53, 1920x1080, **with a music track** (no narration), so it does not autoplay —
browsers block autoplay with sound. It is the Remotion render remuxed as-is with
`-map_metadata -1`, not re-encoded. It gets controls and a poster frame
(`assets/vla.cpp-poster.jpg`, the title card at `ffmpeg -ss 4`).

The four ALOHA clips are the external overview camera, GR00T-N1.6 in BF16 on an AGX Orin,
re-encoded at 3x speed and 640px wide with `setpts=PTS/3`. Two successes and two failures: 35 of
the 40 trials succeed, and the page shows the minority case rather than cutting it.

    aloha-task1-pass-3x.mp4     task 1, setup 1, trial 2   PASS 0m47s
    aloha-task2-pass-3x.mp4     task 2, setup 1, trial 1   PASS 0m24s
    aloha-task1-fail-3x.mp4     task 1, setup 5, trial 2   FAIL timeout/retry
    aloha-task2-fail-3x.mp4     task 2, setup 5, trial 4   FAIL complete

The originals, 240 clips over three cameras and 1.1 GB, stay outside this repo, which is why
only four are here. They also include PyTorch runs of the same tasks, but no evidence rows come
with those and the paper reports no PyTorch ALOHA numbers, so there is no side-by-side to make.

The two SO-101 clips are π0.5 and GR00T-N1.7 on an Intel AI board (Core Ultra X7 358H), one
landscape recording each (654x368) of the same task: the arm opens the drawer, puts the red block
in and pushes the drawer shut. The card's wording is read off the footage, not a task string. Both
play at 1x: scaled to 640x360, stripped of audio and metadata (`-an -map_metadata -1`), and encoded
with x264 at CRF 28. They replace earlier SO-101 clips: three 4x portrait ones, then a single
drawer clip that was labelled π0.5 but was the GR00T-N1.7 recording. The page claims no success
rate or backend for them.

    so101-drawer-pi05-1x.mp4        π0.5         0:57   4.9 MB -> 0.92 MB
    so101-drawer-gr00t-n17-1x.mp4   GR00T-N1.7   0:49   4.3 MB -> 0.85 MB

The raw recordings, with audio where they had it, are kept outside this repo.

The *In simulation* section, between Architecture and the robots, shows five square clips in a row
(two per row on phones, the fifth centred), each tagged with its benchmark.

The two LIBERO-Object clips are the runtime's own episodes from a July 2026 CI check on the RTX
3090, one episode per task (`outputs/ci/rtx3090/<model>/<model>/libero_object/task_<n>/` in the
runtime's release tree). Both are successes per their `summary.txt`. They were cropped to the
scene camera (the left half of the 512x256 two-camera frame), re-encoded with `-map_metadata -1`,
and are about five seconds, silent, autoplaying. They are not from the 200-episode run behind the
explorer's success rates, and the page does not say they are.

    libero-object-gr00t-n17-alphabet-soup.mp4   GR00T-N1.7, task 0, alphabet soup into the basket
    libero-object-bitvla-butter.mp4             BitVLA, task 6, butter into the basket

The same tree has one episode for every LIBERO-Object task for ten models (and all four suites for
BitVLA and GR00T-N1.7). Of those, π0 failed tasks 3, 4 and 7, and GR00T-N1.5, GR00T-N1.6 and
SmolVLA one task each; every other clip is a success.

The two SimplerEnv clips are 256x256 and about two seconds, silent, autoplaying.

    simplerenv-carrot-plate.mp4      WidowX, put carrot on plate    (paper: 80% for GR00T-N1.6)
    simplerenv-eggplant-basket.mp4   WidowX, put eggplant in basket (no rate in the paper)

**These were `libero-videos/episode-00000{3,6}` on the old page, labelled "LIBERO rollout".**
They are not LIBERO: both show a WidowX arm in the SimplerEnv / Bridge scenes, one at a wooden
table with a carrot and plate, one at a sink with an eggplant and a yellow basket. They were
renamed here and captioned for what they are. Nothing on the page claims a policy for them,
because the asset carries no policy label.

The RoboCasa clip is SmolVLA closing a fridge door, 10 s at 20 fps. The source frame is three
256x256 views side by side; the page keeps the left, third-person one (`crop=256:256:0:0`),
re-encoded with `-an -map_metadata -1`. It is shown as a rollout only, with no success rate.

    robocasa-smolvla-close-fridge.mp4   RoboCasa, close the fridge

A placeholder card upgrades itself when its file appears, so more footage needs no HTML edit —
drop the `.mp4` in and give the card the right filename.

## Review period

This page names the authors (in the footer's BibTeX) and links the arXiv paper and the code. While
the paper is under double-anonymous review, nothing submitted for review may link here, and this
page must not link to any anonymous copy of itself. `<meta name="robots" content="noindex">` keeps
it out of search results until the review ends. The figure PNGs are rasterised, so they carry no
PDF metadata. Every video was re-muxed with `-map_metadata -1`.

The hero video's current cut (November 2026) matches the page: thirteen models, seven backends,
the three robots and RoboCasa, and no priority claim. An earlier cut closed on "Twelve
Architectures" and called itself the first flow-matching and diffusion action-head server on
llama.cpp; both are gone.

## Paper, code and BibTeX

The Paper button opens the arXiv abstract (2606.08094) and Code opens
`github.com/VinRobotics/vla.cpp`, the repository the arXiv entry names. Each button leads with its
mark from the icon sprite. The footer's BibTeX is arXiv's own entry with a shorter key
(`nguyen2026vlacpp`); switch it to `@inproceedings` once the paper is published. Its Copy button
needs a secure context (https or localhost) and hides itself without one.

## Architecture

The architecture is HTML and CSS in `index.html`, not an image, so it reflows: five columns on wide
screens, read left to right as bundle, wire, runtime, wire, client, and one stacked column in the
same order with vertical arrows below 760px. It keeps the paper's diagram's content (client
fields, Protobuf over ZeroMQ, the runtime's four features, llama.cpp's ggml, the GGUF bundle) but
shows all seven backends instead of CUDA and CPU, adds RoboCasa and the SO-101 to the client's
targets, and lists only `model.gguf` in the bundle. The client's six thumbnails sit two by two
(simulators first, then robots); on phones they run three by two.

Its pictures and marks, all local, nothing loaded from elsewhere:

- **Icon sprite** at the top of `<body>`, used as `<svg class="ic"><use href="#..."/></svg>`.
  Brand marks (NVIDIA, AMD, Qualcomm, Intel, Apple, Python, C++, Hugging Face, GitHub, arXiv) are
  Simple Icons 16.36.0, CC0; the marks remain their owners' trademarks and stand only for the
  hardware, software or site named. The hardware and software marks' viewBoxes are cropped to each
  mark, so wordmarks (AMD, Intel) render wide;
  `.lg-*` sets each width from its aspect ratio. Interface icons (eye, sliders, chart, message,
  plug, database, repeat, binary, cpu, file, arrows) are Tabler Icons 3.50.0, MIT.
- `assets/arch/llama-cpp.svg`: llama.cpp's icon (`media/llama1-icon-transparent.svg`, MIT), with
  the editor metadata stripped.
- `assets/arch/vla-cpp-logo.png`: the project's wordmark from the previous page, trimmed to 250x48.
- `assets/arch/{libero,simplerenv,robocasa,aloha,ur10e,so101}.jpg`: 168x126 thumbnails cut from
  the page's own clips and the UR10e figure. RoboCasa is frame 30 of its clip, SO-101 the GR00T-N1.7
  drawer clip at 0:20; both saved without metadata.

The backend chips above the abstract use the same sprite, a logo before each backend, and drop the
vendor's name where the logo is the name (AMD, Intel).

## Regenerating a figure

    pdftoppm -png -r 300 -singlefile <name>.pdf assets/<name>

from `overleaf/vla.cpp/ICRA/`. `ur10e.jpg` came across already rasterised from the previous page.

The roofline and the execution-horizon sweep are no longer images; both are drawn in the page in
its own colours and fonts (see *Roofline* above). `assets/horizon.js` holds the sweep's numbers,
copied from the paper's supplementary table: successes out of 50 with their Wilson intervals,
request latency and S/L for each S. Success is teal and throughput the neutral second series. The
S = H setting is shaded like the roofline's memory-bound region, and hovering shows each
setting's numbers. It keeps the paper's two y-axes.

The teaser is TikZ, not a PDF. `figures/teaser.tex` is the paper's `teaser.tex` turned into a
standalone document (IEEEtran, so the fonts match the paper, with the page sized to the figure)
and updated as described under *Paper vs. current project*; the header comment lists each
panel's source. Rebuild it with

    figures/build.sh

which needs `tectonic`, poppler's `pdftocairo` and `python3`. It writes `assets/teaser.svg` with
coordinates rounded to three decimals, about a sixth smaller.

The caption is retyped in `index.html`. The page shows the one SVG as four crops so the panels
stack below 760px: each `.tz-p` carries `--x` and `--w`, its left edge and width in SVG units (the
file is 505.47 wide). The splits sit in the blank gaps between panels, measured on Chrome's
render of the SVG, which is about 1 unit right of poppler's; re-measure them if the figure's
layout changes.
