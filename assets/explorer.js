/* Model x backend explorer — every number here is transcribed from the paper.
   Sources: Table I (eleven-model LIBERO-Object evaluation and CUDA/Metal/OpenVINO profiles),
   Table III (seven-model device study), Table IV (server-side latency against PyTorch on an
   RTX 3090), and the backend-coverage paragraph (i9 CPU requests, Arc A380 through SYCL).
   Nothing is interpolated: a pair the paper does not measure renders as "not measured". */
(function () {
  'use strict';

  var DEVICES = [
    { id:'rtx3060', name:'RTX 3060', sub:'CUDA · 12 GB', tag:'cuda',
      note:'The cross-platform profile device. P90 is request latency and Vision the vision-stage ' +
           'latency; GPU is device allocation and Host the process RSS. L/S and the 200-episode ' +
           'counts come from the separate device study, which uses its own execution horizons.' },

    { id:'rtx3090', name:'RTX 3090', sub:'CUDA · against PyTorch', tag:'cuda',
      note:'Server-side latency, mean of 200 timed calls after warmup, with transport and image ' +
           'serialisation excluded. Compile is torch.compile in default mode. Speedup divides the ' +
           'faster of the two PyTorch configurations by ours.' },

    { id:'m4', name:'Apple M4', sub:'Metal · 24 GB', tag:'metal',
      note:'Separate M4 trials give 374 ms and 9/10 successes for SmolVLA, 1164 ms and 8/10 for ' +
           'π₀, and 539 ms and 10/10 for GR00T-N1.7. Running the vision and language ' +
           'towers in 32-bit cuts fixed-input action discrepancy from 2.0×10⁻³ to ' +
           '8.3×10⁻⁵, for 33 ms and 672 MiB.' },

    { id:'arc', name:'Intel Arc iGPU', sub:'OpenVINO', tag:'ov',
      note:'An Intel Arc integrated GPU through the OpenVINO backend. RSS is the process resident ' +
           'set size.' },

    { id:'a380', name:'Intel Arc A380', sub:'SYCL · discrete', tag:'sycl',
      note:'Request latency through SYCL against the same request served by the card’s Ryzen 5 ' +
           'host. Maximum absolute action difference is 2.9×10⁻³ against CPU outputs ' +
           'with magnitude up to 0.99. Three models were measured this way.' },

    { id:'agx', name:'Jetson AGX Orin', sub:'CUDA · embedded', tag:'edge',
      note:'Part of the seven-model device study. Cells give client inference time per executed ' +
           'action, including preprocessing and transport, and successes out of 200 episodes.' },

    { id:'nano', name:'Jetson Orin Nano', sub:'CUDA · 8 GB', tag:'edge',
      note:'Five of the seven models in the device study load on the 8 GB board. CPU and GPU share ' +
           'memory here, so process RSS alone omits some accelerator allocations.' },

    { id:'i9', name:'Core i9-14900HX', sub:'CPU · 8 threads', tag:'cpu',
      note:'Pure-CPU checks on eight threads, measured for six models. Peak RSS spans 1.4–6.1 GiB ' +
           'across them; the paper does not break it down per model.' }
  ];

  /* Column sets. get() returns null when the paper does not report that cell. */
  var COLS = {
    rtx3060: [
      { h:'P90', u:'ms', k:'p90', f:1 }, { h:'Vision', u:'ms', k:'vis', f:1 },
      { h:'GPU', u:'MiB', k:'gpu', f:0 }, { h:'Host', u:'MiB', k:'host', f:0 },
      { h:'L/S', u:'ms', k:'ls', f:2 }, { h:'of 200', k:'ok', f:0 }
    ],
    rtx3090: [
      { h:'Ours', u:'ms', k:'ours', f:1, ours:true }, { h:'Eager', u:'ms', k:'eager', f:1 },
      { h:'Compile', u:'ms', k:'comp', f:1 }, { h:'Speedup', k:'sp', f:2, x:true, ours:true }
    ],
    m4:  [ { h:'P90', u:'ms', k:'p90', f:1 }, { h:'Vision', u:'ms', k:'vis', f:1 },
           { h:'RSS', u:'MiB', k:'rss', f:0 } ],
    arc: [ { h:'P90', u:'ms', k:'p90', f:1 }, { h:'Vision', u:'ms', k:'vis', f:1 },
           { h:'RSS', u:'MiB', k:'rss', f:0 } ],
    a380:[ { h:'Ryzen host', u:'ms', k:'host', f:0 }, { h:'SYCL', u:'ms', k:'sycl', f:0, ours:true },
           { h:'Speedup', k:'sp', f:2, x:true, ours:true } ],
    agx: [ { h:'S', k:'s200', f:0 }, { h:'L/S', u:'ms', k:'ls', f:2 }, { h:'of 200', k:'ok', f:0 } ],
    nano:[ { h:'S', k:'s200', f:0 }, { h:'L/S', u:'ms', k:'ls', f:2 }, { h:'of 200', k:'ok', f:0 },
           { h:'RSS', u:'MiB', k:'rss', f:0 } ],
    i9:  [ { h:'Request', u:'ms', k:'req', f:0 } ]
  };

  /* Latency key per device, used for the bar and the "sort by latency" order. */
  var LAT = { rtx3060:'p90', rtx3090:'ours', m4:'p90', arc:'p90', a380:'sycl',
              agx:'ls', nano:'ls', i9:'req' };

  var MODELS = [
    { id:'smolvla', name:'SmolVLA', head:'iterative', input:512, S:1,
      succ:96, ci:[90.2, 98.4], ref:96.0,
      note:'The positional-index case study. Its vision encoder maps patch coordinates to discrete ' +
           'indices; computing them at a different precision moves a value across an index boundary ' +
           'and selects another embedding. Matching the half-precision reference takes the ' +
           'position-embedding relative error from 0.32 to 0.003 and LIBERO-Object task 0 from 2/10 ' +
           'to 9/10 episodes. The residual action difference reaches 1.97 after denormalisation, ' +
           'concentrated on the gripper channel — enough to reverse the gripper command.',
      d:{ rtx3060:{ p90:98.9, vis:47.7, gpu:1322, host:735, ls:28.16, ok:181, s200:4 },
          rtx3090:{ ours:55.8, eager:174.1, comp:65.9, sp:1.18 },
          m4:{ p90:330.6, vis:186.3, rss:1380 },
          arc:{ p90:391.6, vis:44.8, rss:1069 },
          a380:{ host:1920, sycl:630, sp:1920 / 630 },
          agx:{ ls:65.41, ok:181, s200:4 },
          nano:{ ls:141.81, ok:176, s200:4, rss:2031 },
          i9:{ req:2141 } } },

    { id:'evo1', name:'Evo-1', head:'iterative', input:448, S:8,
      succ:96, ci:[90.2, 98.4], ref:97.7,
      note:'Uses uniform initial noise rather than the Gaussian sample of the other flow-matching ' +
           'heads. The slowest of the six models measured on CPU.',
      d:{ rtx3060:{ p90:180.0, vis:56.6, gpu:1574, host:822, ls:63.60, ok:189, s200:8 },
          rtx3090:{ ours:142.3, eager:161.6, comp:142.5, sp:1.00 },
          m4:{ p90:775.7, vis:328.0, rss:1620 },
          arc:{ p90:585.7, vis:183.6, rss:1200 },
          a380:{ host:7695, sycl:1176, sp:7695 / 1176 },
          agx:{ ls:131.01, ok:191, s200:8 },
          nano:{ ls:458.84, ok:195, s200:8, rss:2135 },
          i9:{ req:8966 } } },

    { id:'pi0', name:'π₀', head:'iterative', input:224, S:50,
      succ:82, ci:[73.3, 88.3], ref:79.0, repro:true,
      note:'The one model whose success sits well below the rest, at 82% against a 79% PyTorch ' +
           'reproduction. Each layer caches prefix keys and values for joint attention with the ' +
           'changing suffix. The device study runs it at S = 32 on the RTX 3060 and AGX Orin, and ' +
           'S = 50 on the Orin Nano.',
      d:{ rtx3060:{ p90:203.6, vis:40.2, gpu:5480, host:833, ls:9.74, ok:175, s200:32 },
          rtx3090:{ ours:97.9, eager:99.4, comp:99.7, sp:1.02 },
          m4:{ p90:1123.5, vis:189.8, rss:5530 },
          arc:{ p90:779.6, vis:83.9, rss:5060 },
          agx:{ ls:27.90, ok:171, s200:32 },
          nano:{ ls:39.10, ok:161, s200:50, rss:6068, remote:true },
          i9:{ req:7888 } } },

    { id:'pi05', name:'π₀.₅', head:'iterative', input:224, S:10,
      succ:96, ci:[90.2, 98.4], ref:98.2,
      note:'Reference rate is the OpenPI figure reported in LeRobot’s documentation. Profiled on ' +
           'all three platforms, but not part of the device study or the PyTorch comparison.',
      d:{ rtx3060:{ p90:254.1, vis:45.9, gpu:5938, host:729 },
          m4:{ p90:1198.2, vis:198.1, rss:5987 },
          arc:{ p90:568.1, vis:37.7, rss:5490 } } },

    { id:'n15', name:'GR00T-N1.5', head:'iterative', input:224, S:16,
      succ:95, ci:[88.8, 97.8], ref:95.0, repro:true,
      note:'Caches projections of backbone features in a separate action transformer. One of the ' +
           'three models where the runtime is meaningfully faster than the better PyTorch ' +
           'configuration on the RTX 3090.',
      d:{ rtx3060:{ p90:95.2, vis:23.2, gpu:4830, host:1436, ls:14.17, ok:192, s200:16 },
          rtx3090:{ ours:75.4, eager:100.5, comp:94.7, sp:1.26 },
          m4:{ p90:423.2, vis:98.9, rss:6841 },
          arc:{ p90:249.1, vis:28.3, rss:5550 },
          agx:{ ls:28.78, ok:195, s200:16 },
          nano:{ ls:84.76, ok:192, s200:16, rss:5975, remote:true },
          i9:{ req:4888 } } },

    { id:'n16', name:'GR00T-N1.6', head:'iterative', input:224, S:16,
      succ:88, ci:[80.2, 93.0], ref:99.0, repro:true,
      note:'The unresolved case: 88% against 99% for its own PyTorch reproduction, a gap the paper ' +
           'states it cannot account for. It is otherwise well behaved — interpolated position ' +
           'embeddings give 10/10 successes in both tested precision configurations with an action ' +
           'difference near 0.003, and on SimplerEnv’s WidowX tasks at 252-pixel input it reaches ' +
           '80% on “put carrot on plate” and 70% on “put spoon on towel.” It is also ' +
           'the ALOHA policy, at roughly 470 ms per chunk on AGX Orin.',
      d:{ rtx3060:{ p90:91.3, vis:23.8, gpu:5918, host:1441, ls:10.29, ok:173, s200:16 },
          rtx3090:{ ours:61.2, eager:76.0, comp:61.1, sp:1.00 },
          m4:{ p90:389.9, vis:99.8, rss:7144 },
          arc:{ p90:167.3, vis:37.7, rss:4480 },
          agx:{ ls:26.70, ok:180, s200:16 },
          nano:{ oom:true },
          i9:{ req:3341 } } },

    { id:'n17', name:'GR00T-N1.7', head:'iterative', input:256, S:16,
      succ:99, ci:[94.6, 99.8], ref:99.0, repro:true,
      note:'The most-studied model in the paper. Its solver sweep on the RTX 3060 keeps success ' +
           'between 95 and 99 out of 100 from T = 1 to T = 16 while the backbone-plus-head time ' +
           'rises from 19.6 to 107.5 ms, so the extra steps buy little here. A matched comparison on ' +
           'an RTX PRO Blackwell 1000 favours TensorRT: 70.5 ms against 130.2 ms, a 1.85× ' +
           'speedup, at 98/100 against 96/100 successes.',
      d:{ rtx3060:{ p90:93.8, vis:21.7, gpu:6288, host:1417, ls:10.26, ok:196, s200:16 },
          rtx3090:{ ours:59.9, eager:66.1, comp:60.1, sp:1.00, fallback:true },
          m4:{ p90:385.0, vis:87.3, rss:7442 },
          arc:{ p90:299.8, vis:56.8, rss:5890 },
          agx:{ ls:26.84, ok:197, s200:16 },
          nano:{ oom:true },
          i9:{ req:2931 } } },

    { id:'jepa', name:'VLA-JEPA', head:null, input:224, S:7,
      succ:100, ci:[96.3, 100], ref:99.6,
      note:'One of the two models that clear every LIBERO-Object episode. Profiled on all three ' +
           'platforms; it is not in the device study or the PyTorch comparison.',
      d:{ rtx3060:{ p90:64.2, vis:22.3, gpu:3986, host:627 },
          m4:{ p90:258.2, vis:87.3, rss:4034 },
          arc:{ p90:151.2, vis:55.3, rss:3690 } } },

    { id:'adapter', name:'VLA-Adapter', head:'regression', input:224, S:8,
      succ:96, ci:[90.2, 98.4], ref:99.2,
      note:'A regression head: the chunk is predicted directly rather than refined over solver ' +
           'steps. One of the three models measured through SYCL on the Arc A380.',
      d:{ rtx3060:{ p90:82.1, vis:44.1, gpu:2864, host:1063 },
          m4:{ p90:332.4, vis:189.1, rss:3697 },
          arc:{ p90:239.3, vis:121.5, rss:2590 },
          a380:{ host:2994, sycl:517, sp:2994 / 517 } } },

    { id:'oft', name:'OpenVLA-OFT', head:'regression', input:224, S:8,
      succ:97, ci:[91.5, 99.0], ref:98.4,
      note:'By far the heaviest footprint in the set — 15.1 GiB RSS on Metal and 13.7 GiB through ' +
           'OpenVINO, roughly ten times BitVLA. No CUDA profile was recorded.',
      d:{ m4:{ p90:1719.9, vis:194.0, rss:15457 },
          arc:{ p90:482.7, vis:125.2, rss:14040 } } },

    { id:'bitvla', name:'BitVLA', head:'regression', input:224, S:8,
      succ:100, ci:[96.3, 100], ref:99.6,
      note:'Ternary weights, four two-bit codes per byte. It clears every LIBERO-Object episode at ' +
           '100 and again at 200 per device on all three CUDA boards, including the 8 GB Orin Nano, ' +
           'inside 2199 MiB. Across all four LIBERO suites it scores 94.5, 100, 92.5 and 83.5. The ' +
           'IMMA tensor-core kernel runs its client inference 4.0–4.6× faster than the ' +
           'dp4a CUDA-core baseline, and it takes the largest margin over PyTorch in the set, ' +
           '1.84× against a compiled reference that is itself 3.4× faster than eager.',
      d:{ rtx3060:{ p90:62.0, vis:12.0, gpu:1312, host:1072, ls:37.85, ok:200, s200:8 },
          rtx3090:{ ours:51.5, eager:319.7, comp:94.6, sp:1.84 },
          agx:{ ls:101.11, ok:200, s200:8 },
          nano:{ ls:355.65, ok:200, s200:8, rss:2199 } } }
  ];

  var dev  = 'rtx3060';
  var mod  = 'bitvla';
  var sort = 'lat';
  var el   = {};

  /* ---------------- helpers ---------------- */

  function model(id) { for (var i = 0; i < MODELS.length; i++) if (MODELS[i].id === id) return MODELS[i]; return null; }
  function device(id) { for (var i = 0; i < DEVICES.length; i++) if (DEVICES[i].id === id) return DEVICES[i]; return null; }
  function cell(m, d) { return (m.d && m.d[d]) || null; }

  /* A pair counts as measured only if it carries at least one reported number. */
  function measured(m, d) {
    var c = cell(m, d);
    if (!c || c.oom) return false;
    var cols = COLS[d];
    for (var i = 0; i < cols.length; i++) if (c[cols[i].k] != null) return true;
    return false;
  }

  function num(v, f, x) {
    if (v == null) return '<span class="xp-dim">—</span>';
    return v.toFixed(f) + (x ? '×' : '');
  }

  function lat(m, d) { var c = cell(m, d); return c ? c[LAT[d]] : null; }

  function rows(d) {
    var yes = [], no = [];
    for (var i = 0; i < MODELS.length; i++) (measured(MODELS[i], d) ? yes : no).push(MODELS[i]);
    yes.sort(function (a, b) {
      if (sort === 'succ') return b.succ - a.succ || lat(a, d) - lat(b, d);
      return lat(a, d) - lat(b, d);
    });
    return { yes: yes, no: no };
  }

  /* ---------------- panes ---------------- */

  function renderDevices() {
    var out = '';
    for (var i = 0; i < DEVICES.length; i++) {
      var p = DEVICES[i], n = 0;
      for (var j = 0; j < MODELS.length; j++) if (measured(MODELS[j], p.id)) n++;
      out += '<button type="button" class="xp-target' + (p.id === dev ? ' on' : '') +
             '" data-dev="' + p.id + '"><b>' + p.name + '</b><span>' + p.sub + '</span>' +
             '<i class="xp-b ' + p.tag + '">' + p.tag + '</i>' +
             '<i class="xp-b n">' + n + ' of 11</i></button>';
    }
    el.devices.innerHTML = out;
  }

  function renderMatrix() {
    var out = '<div class="xp-mgrid"><div></div>';
    for (var i = 0; i < DEVICES.length; i++)
      out += '<div class="xp-mh' + (DEVICES[i].id === dev ? ' on' : '') + '">' + DEVICES[i].name + '</div>';
    for (var j = 0; j < MODELS.length; j++) {
      var m = MODELS[j];
      out += '<div class="xp-mr' + (m.id === mod ? ' on' : '') + '">' + m.name + '</div>';
      for (var k = 0; k < DEVICES.length; k++) {
        var d = DEVICES[k].id, c = cell(m, d);
        var cls = c && c.oom ? 'oom' : measured(m, d) ? 'on' : 'off';
        var sel = (m.id === mod && d === dev) ? ' sel' : '';
        var t = cls === 'oom' ? 'out of memory' : cls === 'on' ? 'measured' : 'not measured';
        out += '<button type="button" class="xp-c ' + cls + sel + '" data-dev="' + d +
               '" data-mod="' + m.id + '" title="' + m.name + ' · ' + DEVICES[k].name + ' — ' + t +
               '"><span class="xp-sr">' + t + '</span></button>';
      }
    }
    out += '</div><ul class="xp-leg"><li><i class="on"></i>measured</li>' +
           '<li><i class="oom"></i>out of memory</li><li><i class="off"></i>not measured</li></ul>' +
           '<p class="xp-cap">Coverage is uneven by design: the eleven-model profiles, the seven-model ' +
           'device study and the PyTorch comparison are three separate experiments. Click any cell.</p>';
    el.matrix.innerHTML = out;
  }

  function renderModel() {
    var m = model(mod), c = cell(m, dev);
    var head = m.head ? m.head + ' head' : 'head type not stated';
    var out = '<div class="xp-mname">' + m.name + '</div>' +
              '<div class="xp-mhead">' + head + ' · ' + m.input + ' px input · S = ' + m.S + '</div>';

    /* LIBERO-Object, 100 episodes: success with its Wilson interval and the reference rate. */
    var lo = m.ci[0], hi = m.ci[1];
    out += '<div class="xp-sec">LIBERO-Object · 100 episodes</div>' +
           '<div class="xp-scale"><div class="xp-track">' +
           '<span class="xp-fill" style="width:' + m.succ + '%"></span>' +
           '<span class="xp-ref" style="left:calc(' + m.ref + '% - 1px)" title="reference ' + m.ref + '%"></span>' +
           '</div><div class="xp-whisk"><span style="left:' + lo + '%;right:' + (100 - hi) + '%"></span></div>' +
           '<div class="xp-ticks"><span>0</span><span>50</span><span>100%</span></div>' +
           '<p class="xp-slg">bar: success \u00b7 caps: 95% interval \u00b7 rule: reference rate</p></div>' +
           '<table class="xp-kv"><tbody>' +
           '<tr><td>success</td><td><b>' + m.succ + '%</b> <span class="xp-dim">[' + lo + ', ' + hi + ']</span></td></tr>' +
           '<tr><td>reference</td><td>' + m.ref.toFixed(1) + '%' +
           (m.repro ? ' <span class="xp-dim">PyTorch reproduction</span>' : ' <span class="xp-dim">published</span>') +
           '</td></tr></tbody></table>';

    /* This model on the selected device. */
    var p = device(dev);
    out += '<div class="xp-sec">on ' + p.name + '</div>';
    if (c && c.oom) {
      out += '<p class="xp-oom">Out of memory. The checkpoint does not load in the 8 GB shared pool.</p>';
    } else if (!measured(m, dev)) {
      out += '<p class="xp-nm">Not measured. The paper reports no number for this pair.</p>';
    } else {
      out += '<table class="xp-kv"><tbody>';
      var cols = COLS[dev];
      for (var i = 0; i < cols.length; i++) {
        var v = c[cols[i].k];
        if (v == null) continue;
        out += '<tr><td>' + cols[i].h + (cols[i].u ? ' <span class="xp-dim">(' + cols[i].u + ')</span>' : '') +
               '</td><td class="xp-num' + (cols[i].ours ? ' ours' : '') + '">' + num(v, cols[i].f, cols[i].x) + '</td></tr>';
      }
      out += '</tbody></table>';
      if (c.remote) out += '<p class="xp-fn">Evaluated against a remote simulator.</p>';
      if (c.fallback) out += '<p class="xp-fn">The compiled baseline allows eager fallback through ' +
                             'suppress_errors; the timing includes any fallback it took.</p>';
    }
    out += '<p class="xp-note">' + m.note + '</p>';
    el.model.innerHTML = out;
  }

  function renderTable() {
    var p = device(dev), cols = COLS[dev], r = rows(dev), i, j;
    var max = 0;
    for (i = 0; i < r.yes.length; i++) max = Math.max(max, lat(r.yes[i], dev) || 0);

    var out = '<table><thead><tr><th>Model</th>';
    for (i = 0; i < cols.length; i++)
      out += '<th>' + cols[i].h + (cols[i].u ? '<span class="xp-u"> ' + cols[i].u + '</span>' : '') + '</th>';
    out += '<th class="xp-barh">' + (dev === 'rtx3090' ? 'ours' : 'latency') + '</th></tr></thead><tbody>';

    for (i = 0; i < r.yes.length; i++) {
      var m = r.yes[i], c = cell(m, dev);
      out += '<tr data-mod="' + m.id + '" tabindex="0"' + (m.id === mod ? ' class="xp-sel"' : '') + '><th>' + m.name + '</th>';
      for (j = 0; j < cols.length; j++)
        out += '<td class="xp-num' + (cols[j].ours ? ' ours' : '') + '">' + num(c[cols[j].k], cols[j].f, cols[j].x) + '</td>';
      var w = max ? Math.max(2, Math.round(100 * lat(m, dev) / max)) : 0;
      out += '<td class="xp-bar"><span style="width:' + w + '%"></span></td></tr>';
    }

    for (i = 0; i < r.no.length; i++) {
      var n = r.no[i], nc = cell(n, dev);
      out += '<tr class="xp-none" data-mod="' + n.id + '" tabindex="0"><th>' + n.name + '</th>' +
             '<td colspan="' + (cols.length + 1) + '">' +
             (nc && nc.oom ? 'out of memory on this device' : 'not measured on this device') + '</td></tr>';
    }

    out += '</tbody></table><p class="xp-fn">' + p.note + '</p>';
    el.table.innerHTML = out;
  }

  function renderVerdict() {
    var m = model(mod), p = device(dev), c = cell(m, dev), out;

    if (c && c.oom) {
      out = '<b>' + m.name + ' does not fit ' + p.name + '.</b>' +
            '<span class="xp-vsub">out of memory · 8 GiB shared between CPU and GPU</span>';
    } else if (!measured(m, dev)) {
      out = '<b>' + m.name + ' on ' + p.name + ': not measured.</b>' +
            '<span class="xp-vsub">no number reported for this pair</span>';
    } else if (dev === 'rtx3090') {
      var best = Math.min(c.eager, c.comp);
      out = '<b>' + m.name + ' runs in ' + c.ours.toFixed(1) + ' ms on ' + p.name + '.</b>' +
            '<span class="xp-vsub">' + c.sp.toFixed(2) + '× the better PyTorch configuration at ' +
            best.toFixed(1) + ' ms · eager ' + c.eager.toFixed(1) + ' · compiled ' + c.comp.toFixed(1) + '</span>';
    } else if (dev === 'a380') {
      out = '<b>' + m.name + ' answers in ' + c.sycl + ' ms through SYCL.</b>' +
            '<span class="xp-vsub">' + c.sp.toFixed(2) + '× the ' + c.host + ' ms taken by the card’s Ryzen 5 host</span>';
    } else if (dev === 'i9') {
      out = '<b>' + m.name + ' answers in ' + c.req + ' ms on eight CPU threads.</b>' +
            '<span class="xp-vsub">CPU only · too slow for most control settings, but it runs</span>';
    } else if (c.ls != null) {
      var hz = 1000 / c.ls;
      out = '<b>' + m.name + ' on ' + p.name + ': ' + c.ok + ' of 200 episodes.</b>' +
            '<span class="xp-vsub">' + c.ls.toFixed(2) + ' ms per executed action at S = ' + c.s200 +
            ' · ' + hz.toFixed(1) + ' actions per second' + (c.rss ? ' · ' + c.rss + ' MiB peak RSS' : '') +
            (c.p90 != null ? ' · P90 ' + c.p90.toFixed(1) + ' ms' : '') + '</span>';
    } else {
      out = '<b>' + m.name + ' on ' + p.name + ': P90 ' + c.p90.toFixed(1) + ' ms.</b>' +
            '<span class="xp-vsub">vision stage ' + c.vis.toFixed(1) + ' ms · ' +
            (c.gpu != null ? c.gpu + ' MiB on device, ' + c.host + ' MiB host RSS' : c.rss + ' MiB RSS') + '</span>';
    }

    if (dev === 'rtx3060' && c && !c.oom && c.ls == null && measured(m, dev))
      out += '<span class="xp-vnote">Profiled here, but not part of the seven-model device study, so ' +
             'it has no per-action timing or 200-episode count on this board.</span>';

    el.verdict.innerHTML = out;
  }

  function render() {
    renderDevices(); renderMatrix(); renderModel(); renderTable(); renderVerdict();
    var sb = document.querySelectorAll('#xp-app [data-sort]');
    for (var i = 0; i < sb.length; i++)
      sb[i].setAttribute('aria-pressed', sb[i].dataset.sort === sort ? 'true' : 'false');
  }

  /* ---------------- wiring ---------------- */

  function init() {
    var root = document.getElementById('xp-app');
    if (!root) return;
    ['devices', 'matrix', 'model', 'table', 'verdict'].forEach(function (k) {
      el[k] = root.querySelector('[data-pane="' + k + '"]');
    });

    root.addEventListener('click', function (e) {
      var c = e.target.closest('.xp-c');
      if (c) { dev = c.dataset.dev; mod = c.dataset.mod; return render(); }
      var t = e.target.closest('[data-dev]');
      if (t) { dev = t.dataset.dev; return render(); }
      var row = e.target.closest('tr[data-mod]');
      if (row) { mod = row.dataset.mod; return render(); }
      var sb = e.target.closest('[data-sort]');
      if (sb) { sort = sb.dataset.sort; return render(); }
    });

    root.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      var row = e.target.closest('tr[data-mod]');
      if (row) { e.preventDefault(); mod = row.dataset.mod; render(); }
    });

    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
