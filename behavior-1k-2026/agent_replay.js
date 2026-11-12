// Astra rollout viewer: the video with narrated captions that stop at key moments, a phase
// timeline, the command behind the current frame, and where the real time went.
// Replay data comes from the public bucket manifest; narration comes from astra_story.js.
(() => {
  const MANIFEST = 'https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/agent-rollouts/full/manifest.json?v=20261111-full';
  const $ = id => document.getElementById(id);
  const make = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const SVG = 'http://www.w3.org/2000/svg';
  const svg = (tag, attrs = {}, text) => {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  const clock = (s, round = Math.floor) => {
    s = Math.max(0, round(s));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  const span = s => {
    s = Math.round(s);
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60), r = s % 60;
    return r ? `${m} min ${r} s` : `${m} min`;
  };
  const displayTask = name => name.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());

  const video = $('ar-video');
  const strip = $('ar-strip');
  const caption = $('ar-caption');
  const chart = $('ar-chart');
  const tip = $('ar-tooltip');

  // ---- Turning a replay file into an episode -----------------------------
  function prepare(entry, data) {
    const fps = data.metadata.video_fps;
    const last = data.metadata.steps - 1;
    const duration = data.agent.duration_seconds;
    const spans = data.agent.spans;

    // Wall time at which each video frame first appeared (state step n renders frame n - 1).
    const frameWall = [];
    for (const s of data.agent.states) {
      const f = Math.max(0, (s.step || 1) - 1);
      if (f <= last && frameWall[f] === undefined) frameWall[f] = s.wall_seconds;
    }
    frameWall[0] ??= 0;
    for (let f = 1; f <= last; f++) frameWall[f] ??= frameWall[f - 1];

    // The intent text Astra attached to each command: s.act(..., intent='…') or s.act('…', ...).
    const intentOf = robot => {
      const tool = spans.filter(t => t.kind === 'tool' && t.start_seconds <= robot.start_seconds + .01
        && robot.start_seconds - t.start_seconds < 1 && /\bs\.act\(/.test(t.arguments?.command || '')).at(-1);
      const command = tool?.arguments?.command || '';
      const m = /intent=(['"])(.*?)\1/.exec(command) || /\bs\.act\(\s*(['"])(.*?)\1/.exec(command);
      return m ? m[2] : '';
    };
    const robotCalls = spans.filter(s => s.kind === 'robot').sort((a, b) => a.start_seconds - b.start_seconds);
    let prevEnd = 0;
    const moves = robotCalls.filter(r => /act/i.test(r.label)).map(robot => {
      const move = {
        from: prevEnd, start: robot.start_seconds, startFrame: robot.video_frame || 0,
        reason: robot.result?.act_result?.reason || '',
        intent: intentOf(robot),
        summaries: spans.filter(s => s.kind === 'model' && s.summary && s.start_seconds >= prevEnd && s.start_seconds < robot.start_seconds),
      };
      prevEnd = robot.end_seconds ?? duration;
      return move;
    });

    // Narration: hand-written phases and captions when available, otherwise one unnamed phase.
    const story = window.ASTRA_STORIES?.[entry.slug];
    const phases = (story?.phases || [[0, 'Rollout', 'Rollout']]).map(([frame, name, short], k) => ({ n: k + 1, frame, name, short }));
    const beats = (story?.beats || []).map(([frame, text], k) => ({ k, frame, text }));
    const starts = new Set(moves.map(m => m.startFrame));
    for (const f of [...phases.map(p => p.frame), ...beats.map(b => b.frame)])
      if (f && !starts.has(f)) console.warn(`${entry.slug}: frame ${f} is not the start of a robot command`);
    phases.forEach((p, k) => {
      const next = phases[k + 1];
      p.startFrame = p.frame;
      p.endFrame = next ? next.frame : last;
      p.moves = moves.filter(m => m.startFrame >= p.startFrame && (!next || m.startFrame < next.frame));
      const first = moves.find(m => m.startFrame >= p.frame);
      p.wallStart = k === 0 ? 0 : first.from;
      p.stalled = p.moves.filter(m => m.reason === 'stalled').length;
      p.timeouts = p.moves.filter(m => m.reason === 'timeout').length;
    });
    phases.forEach((p, k) => { p.wallEnd = phases[k + 1]?.wallStart ?? duration; });

    const moving = robotCalls.reduce((sum, r) => sum + ((r.end_seconds ?? duration) - r.start_seconds), 0);
    return { entry, data, fps, last, duration, frameWall, robotCalls, moves, phases, beats, stillShare: 1 - moving / duration };
  }

  // ---- Player state ------------------------------------------------------
  let ep = null, segs = [];
  let shownPhase = -1, shownBeat = -1, shownMove = -1, holdTimer = null, dragging = false, resume = false;
  const frameNow = () => ep ? Math.min(ep.last, Math.max(0, Math.floor(video.currentTime * ep.fps + 1e-4))) : 0;
  const seekFrame = f => { video.currentTime = Math.min(ep.last, Math.max(0, f)) / ep.fps + .001; };
  const phaseAt = f => ep.phases.findLast(p => f >= p.startFrame) || ep.phases[0];
  const beatAt = f => ep.beats.findLast(b => f >= b.frame) || ep.beats[0];
  const moveAt = f => { let found = 0; ep.moves.forEach((m, i) => { if (m.startFrame <= f) found = i; }); return found; };

  function stopHold() {
    if (holdTimer) clearTimeout(holdTimer);
    holdTimer = null;
    caption.classList.remove('is-holding');
  }
  // When playback reaches a new caption, hold the frame long enough to read it.
  function hold(b) {
    const ms = Math.min(9000, 2000 + b.text.length * 40);
    holdTimer = -1; // marks the hold before pause() fires its event
    video.pause();
    seekFrame(b.frame);
    const bar = $('ar-hold');
    bar.style.transition = 'none';
    bar.style.width = '0';
    void bar.offsetWidth;
    bar.style.transition = `width ${ms}ms linear`;
    bar.style.width = '100%';
    caption.classList.add('is-holding');
    holdTimer = setTimeout(() => { stopHold(); video.play().catch(() => {}); }, ms);
  }
  function playFrom(frame) {
    stopHold();
    seekFrame(frame);
    video.play().catch(() => {});
  }
  function goTo(frame) {
    stopHold();
    video.pause();
    seekFrame(frame);
    update();
  }

  // ---- Timeline ----------------------------------------------------------
  function buildStrip() {
    segs = ep.phases.map(p => {
      const b = make('button', 'ar-seg');
      b.type = 'button';
      b.style.flex = `${Math.max(1, p.endFrame - p.startFrame)} 1 0`;
      b.append(make('b', null, p.n), make('span', null, p.short));
      b.setAttribute('aria-label', `Phase ${p.n}: ${p.name}`);
      for (const m of p.moves) {
        if (m.startFrame === p.startFrame) continue;
        const tick = make('i');
        tick.style.left = `${(m.startFrame - p.startFrame) / Math.max(1, p.endFrame - p.startFrame) * 100}%`;
        b.append(tick);
      }
      b.addEventListener('click', e => { if (e.detail === 0) playFrom(p.startFrame); });
      return b;
    });
    strip.replaceChildren(...segs, make('span', 'ar-playhead'));
    fitStrip();
  }
  const fitStrip = () => segs.forEach(b => {
    b.classList.remove('is-tight');
    if (b.scrollWidth > b.clientWidth + 1) b.classList.add('is-tight');
  });
  const stripFrame = e => {
    const r = strip.getBoundingClientRect();
    return Math.round(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * ep.last);
  };
  strip.addEventListener('pointerdown', e => {
    if (!ep) return;
    resume = !video.paused || !!holdTimer;
    stopHold();
    dragging = true;
    video.pause();
    strip.setPointerCapture(e.pointerId);
    seekFrame(stripFrame(e));
    update();
  });
  strip.addEventListener('pointermove', e => { if (dragging) { seekFrame(stripFrame(e)); update(); } });
  const endDrag = () => { if (!dragging) return; dragging = false; if (resume) video.play().catch(() => {}); };
  strip.addEventListener('pointerup', endDrag);
  strip.addEventListener('pointercancel', endDrag);

  // ---- Controls ----------------------------------------------------------
  const toggle = () => {
    if (!ep) return;
    if (holdTimer) { stopHold(); update(); return; }
    if (video.paused) {
      if (video.ended || frameNow() >= ep.last - 1) seekFrame(0);
      video.play().catch(() => {});
    } else video.pause();
  };
  $('ar-play').addEventListener('click', toggle);
  $('ar-bigplay').addEventListener('click', toggle);
  video.addEventListener('click', toggle);
  $('ar-prev').addEventListener('click', () => goTo(ep.moves[Math.max(0, shownMove - 1)].startFrame));
  $('ar-next').addEventListener('click', () => goTo(ep.moves[Math.min(ep.moves.length - 1, shownMove + 1)].startFrame));
  document.addEventListener('keydown', e => {
    if (!ep?.beats.length || e.target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const r = $('agent-replay').getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    e.preventDefault();
    const back = frameNow() > ep.beats[shownBeat].frame + 15 ? shownBeat : shownBeat - 1;
    const k = Math.max(0, Math.min(ep.beats.length - 1, e.key === 'ArrowLeft' ? back : shownBeat + 1));
    goTo(ep.beats[k].frame);
  });

  // ---- The command behind the current frame ------------------------------
  const OUTCOME = { reached: ['✓ reached', ''], stalled: ['‖ stalled', 'is-stalled'], timeout: ['⧗ timed out', 'is-timeout'] };
  function showCommand(i) {
    const m = ep.moves[i];
    const thinks = $('ar-thinks');
    thinks.replaceChildren();
    for (const s of m.summaries) {
      // Only the title shows; the verbatim body is often garbled, so it stays one click away.
      const text = s.summary.trim();
      const head = /^\*\*(.+?)\*\*\s*/.exec(text);
      const d = make('details', 'ar-think');
      d.append(make('summary', null, head ? head[1] : 'Reasoning summary'));
      const body = make('div');
      for (const para of text.slice(head ? head[0].length : 0).split(/\n\s*\n/)) if (para.trim()) body.append(make('p', null, para.trim()));
      d.append(body);
      thinks.append(d);
    }
    $('ar-command').textContent = m.intent || '(no stated intent)';
    const last = i === ep.moves.length - 1;
    const [label, cls] = OUTCOME[m.reason] || (last && ep.data.metadata.success ? ['✓ goal met', ''] : [m.reason || 'no result', '']);
    $('ar-meta').replaceChildren(`Astra’s command ${i + 1} of ${ep.moves.length} · thought ${span(m.start - m.from)} · `, make('span', cls, label));
    $('ar-prev').disabled = i === 0;
    $('ar-next').disabled = last;
  }

  // ---- Sync everything to the current frame -------------------------------
  function update() {
    if (!ep) return;
    const f = frameNow();
    strip.querySelector('.ar-playhead').style.left = `${f / ep.last * 100}%`;
    $('ar-time').textContent = `${clock(video.currentTime)} / ${clock((ep.last + 1) / ep.fps, Math.round)}`;
    $('ar-wall').textContent = clock(ep.frameWall[f]);

    const p = phaseAt(f);
    if (p.n !== shownPhase) {
      shownPhase = p.n;
      segs.forEach((b, k) => { b.classList.toggle('is-active', k === p.n - 1); b.setAttribute('aria-current', String(k === p.n - 1)); });
      chart.querySelectorAll('.ar-phase').forEach((g, k) => g.classList.toggle('is-active', k === p.n - 1));
    }
    const moving = !video.paused && !video.ended;
    if (ep.beats.length) {
      const beat = beatAt(f);
      if (beat.k !== shownBeat) {
        shownBeat = beat.k;
        $('ar-caption-phase').textContent = `${phaseAt(beat.frame).n} · ${phaseAt(beat.frame).name}`;
        $('ar-caption-text').textContent = beat.text;
        // Stop only when playback reaches a caption, not after a jump into the middle of one.
        if (moving && !dragging && f - beat.frame < 20) hold(beat);
      }
    }
    const mi = moveAt(f);
    if (mi !== shownMove) { shownMove = mi; showCommand(mi); }

    const playing = moving || !!holdTimer;
    $('ar-play').textContent = playing ? '❚❚' : '▶';
    $('ar-play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
    $('ar-screen').classList.toggle('is-playing', playing);
  }
  const tick = () => { update(); if (!video.paused) requestAnimationFrame(tick); };
  video.addEventListener('play', () => requestAnimationFrame(tick));
  for (const ev of ['pause', 'seeked', 'loadedmetadata', 'ended']) video.addEventListener(ev, update);
  video.addEventListener('error', () => { if (video.getAttribute('src')) status('The rollout video could not be loaded.'); });

  // ---- Where the real time went -----------------------------------------
  const describe = p => `<b>${p.n} · ${p.name}</b>${span((p.endFrame - p.startFrame) / ep.fps)} of video · ${span(p.wallEnd - p.wallStart)} of real time<br>${p.moves.length} commands · ${p.stalled} stalled · ${p.timeouts} timed out`;
  function drawChart() {
    const W = chart.clientWidth;
    if (!W || !ep) return;
    const narrow = W < 520;
    const L = narrow ? 0 : 84, R = W - 2;
    const topY = narrow ? 20 : 4, barH = 24, botY = topY + barH + 54, H = botY + barH + (narrow ? 42 : 24);
    const xv = f => L + f / ep.last * (R - L);
    const xw = t => L + t / ep.duration * (R - L);
    const videoTotal = clock((ep.last + 1) / ep.fps, Math.round), wallTotal = clock(ep.duration, Math.round);
    const root = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Video time against real time for each phase: ${videoTotal} of video took ${wallTotal} of real time.` });
    if (narrow) {
      root.append(svg('text', { x: 0, y: 13, class: 'ar-row' }, `Video · ${videoTotal}`));
      root.append(svg('text', { x: 0, y: H - 3, class: 'ar-row' }, `Real time · ${wallTotal}`));
    } else {
      root.append(svg('text', { x: 0, y: topY + 11, class: 'ar-row' }, 'Video'));
      root.append(svg('text', { x: 0, y: topY + 25, class: 'ar-sub' }, videoTotal));
      root.append(svg('text', { x: 0, y: botY + 11, class: 'ar-row' }, 'Real time'));
      root.append(svg('text', { x: 0, y: botY + 25, class: 'ar-sub' }, wallTotal));
    }
    const defs = svg('defs');
    const labels = [];
    root.append(defs);
    ep.phases.forEach((p, k) => {
      const lastPhase = k === ep.phases.length - 1;
      const g = svg('g', { class: 'ar-phase', tabindex: 0, role: 'button', 'aria-label': `Play phase ${p.n}, ${p.name}` });
      const a0 = xv(p.startFrame) + (k ? 1 : 0), a1 = xv(p.endFrame) - (lastPhase ? 0 : 1);
      const b0 = xw(p.wallStart) + (k ? 1 : 0), b1 = xw(p.wallEnd) - (lastPhase ? 0 : 1);
      g.append(svg('path', { class: 'ar-band', d: `M${a0},${topY + barH} L${a1},${topY + barH} L${b1},${botY} L${b0},${botY} Z` }));
      g.append(svg('rect', { class: 'ar-moving', x: a0, y: topY, width: Math.max(1, a1 - a0), height: barH, rx: 4 }));
      if (a1 - a0 > 16) g.append(svg('text', { x: (a0 + a1) / 2, y: topY + 16, 'text-anchor': 'middle', class: 'ar-num ar-on-moving' }, p.n));
      // Real-time bar: still background with robot-motion slivers, clipped to rounded ends.
      const id = `ar-clip-${k}`;
      const cp = svg('clipPath', { id });
      cp.append(svg('rect', { x: b0, y: botY, width: Math.max(1, b1 - b0), height: barH, rx: 4 }));
      defs.append(cp);
      const wall = svg('g', { 'clip-path': `url(#${id})` });
      wall.append(svg('rect', { class: 'ar-still', x: b0, y: botY, width: Math.max(1, b1 - b0), height: barH }));
      for (const r of ep.robotCalls) {
        const t0 = Math.max(r.start_seconds, p.wallStart), t1 = Math.min(r.end_seconds ?? ep.duration, p.wallEnd);
        if (t1 > t0) wall.append(svg('rect', { class: 'ar-moving', x: xw(t0), y: botY, width: Math.max(1, xw(t1) - xw(t0)), height: barH }));
      }
      g.append(wall);
      const label = svg('text', { y: botY + barH + 15 }, `${p.n} ${p.short}`);
      labels.push({ label, mid: (b0 + b1) / 2, w: textWidth(label.textContent) });
      g.append(label);
      // Hover/click area follows this phase's own bars and band, so slanted neighbours never overlap.
      g.append(svg('path', { class: 'ar-hit', d: `M${a0},${topY} H${a1} V${topY + barH} L${b1},${botY} V${botY + barH} H${b0} V${botY} L${a0},${topY + barH} Z` }));
      const show = e => {
        tip.innerHTML = describe(p);
        tip.hidden = false;
        const at = e.clientX != null ? { x: e.clientX, y: e.clientY } : (() => { const r = g.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top }; })();
        const w = tip.offsetWidth;
        tip.style.left = `${Math.min(innerWidth - w - 8, Math.max(8, at.x - w / 2))}px`;
        tip.style.top = `${at.y - tip.offsetHeight - 12}px`;
      };
      const hide = () => { tip.hidden = true; };
      const go = () => { hide(); $('agent-replay').scrollIntoView({ behavior: 'smooth', block: 'start' }); playFrom(p.startFrame); };
      g.addEventListener('pointermove', show);
      g.addEventListener('focus', show);
      g.addEventListener('pointerleave', hide);
      g.addEventListener('blur', hide);
      g.addEventListener('click', go);
      g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      g.classList.toggle('is-active', p.n === shownPhase);
      root.append(g);
    });
    // Phase names under the real-time bar may spill past a narrow band; any that would still
    // collide drop to a second row with a short leader line, or fall back to the number alone.
    const extra = placeLabels(labels, L, R, botY + barH);
    if (extra) {
      root.setAttribute('height', H + extra);
      root.setAttribute('viewBox', `0 0 ${W} ${H + extra}`);
      if (narrow) root.querySelector('.ar-row:last-of-type')?.setAttribute('y', H + extra - 3);
    }
    chart.replaceChildren(root);
  }

  const measure = document.createElement('canvas').getContext('2d');
  const textWidth = text => {
    measure.font = `${parseFloat(getComputedStyle(document.documentElement).fontSize) * 0.72}px "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif`;
    return measure.measureText(text).width;
  };
  // Lay labels out left to right without overlaps, keeping each one over or touching its band.
  function fitRow(items, L, R) {
    const GAP = 10;
    let prev = L - GAP;
    for (const it of items) { it.x = Math.max(it.mid - it.w / 2, prev + GAP, L); prev = it.x + it.w; }
    let next = R + GAP;
    for (const it of [...items].reverse()) { it.x = Math.min(it.x, next - GAP - it.w); next = it.x; }
    return items.filter((it, k) => it.x < L - 0.5 || (k && it.x < items[k - 1].x + items[k - 1].w + GAP - 0.5)
      || Math.abs(it.x + it.w / 2 - it.mid) > it.w / 2 + 4);
  }
  function placeLabels(labels, L, R, barBottom) {
    let row1 = labels, row2 = [];
    for (let bad = fitRow(row1, L, R); bad.length; bad = fitRow(row1, L, R)) {
      const worst = bad.reduce((a, b) => (a.w > b.w ? a : b));
      row1 = row1.filter(it => it !== worst);
      row2.push(worst);
    }
    row2.sort((a, b) => a.mid - b.mid);
    for (let bad = fitRow(row2, L, R); bad.length; bad = fitRow(row2, L, R)) {
      const worst = bad[0];
      row2 = row2.filter(it => it !== worst);
      worst.label.textContent = worst.label.textContent.split(' ')[0];
      worst.w = textWidth(worst.label.textContent);
      row1 = [...row1, worst].sort((a, b) => a.mid - b.mid);
      fitRow(row1, L, R);
    }
    for (const it of row1) it.label.setAttribute('x', it.x);
    for (const it of row2) {
      const y = +it.label.getAttribute('y') + 16;
      it.label.setAttribute('x', it.x);
      it.label.setAttribute('y', y);
      it.label.before(svg('line', { class: 'ar-leader', x1: it.mid, x2: it.mid, y1: barBottom + 2, y2: y - 11 }));
    }
    return row2.length ? 16 : 0;
  }

  // ---- Loading episodes ----------------------------------------------------
  const cache = new Map();
  const status = text => { $('ar-status').textContent = text; $('ar-status').hidden = !text; };
  let request = 0;
  async function choose(entry, button) {
    const mine = ++request;
    document.querySelectorAll('#agent-episode button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    stopHold();
    video.pause();
    status('Loading rollout…');
    try {
      if (!cache.has(entry.slug)) {
        const response = await fetch(entry.data);
        if (!response.ok) throw new Error('Rollout data could not be loaded');
        const data = await response.json();
        if (!data.agent?.spans?.length || !data.agent?.states?.length || !data.metadata?.video_fps || !data.video)
          throw new Error('Rollout data is incomplete');
        cache.set(entry.slug, prepare(entry, data));
      }
    } catch (error) {
      if (mine === request) status(error.message || 'Rollout could not be loaded.');
      return;
    }
    if (mine !== request) return; // a later tab click won
    ep = cache.get(entry.slug);
    status('');
    shownPhase = shownBeat = shownMove = -1;
    const md = ep.data.metadata, q = md.metrics?.q_score?.final;
    video.src = ep.data.video;
    video.setAttribute('aria-label', `${displayTask(md.task)}, instance ${md.instance_id}: wrist cameras on the left, head camera on the right`);
    $('ar-result').textContent = [
      `Instance ${md.instance_id}`,
      md.success === true ? 'task succeeded' : md.success === false ? 'task incomplete' : null,
      q != null ? `Q ${Number(q).toFixed(2)}` : null,
      `${clock((ep.last + 1) / ep.fps, Math.round)} of video took ${Math.round(ep.duration / 60)} min of real time`,
      `robot still ${Math.round(ep.stillShare * 100)}% of it`,
    ].filter(Boolean).join(' · ');
    caption.hidden = !ep.beats.length;
    buildStrip();
    drawChart();
    update();
  }

  new ResizeObserver(() => { if (ep) { fitStrip(); drawChart(); } }).observe(chart);
  fetch(MANIFEST)
    .then(response => { if (!response.ok) throw new Error('Episode list could not be loaded'); return response.json(); })
    .then(manifest => {
      if (!manifest.episodes?.length) throw new Error('No agent rollouts are available');
      const buttons = manifest.episodes.map(entry => {
        const button = make('button', null, displayTask(entry.task));
        button.type = 'button';
        button.setAttribute('aria-pressed', 'false');
        button.addEventListener('click', () => choose(entry, button));
        return button;
      });
      $('agent-episode').replaceChildren(...buttons);
      return choose(manifest.episodes[0], buttons[0]);
    })
    .catch(error => status(error.message || 'Episodes could not be loaded.'));
})();
