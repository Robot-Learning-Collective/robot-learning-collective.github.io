(() => {
  const $ = id => document.getElementById(id);
  const video = $('agent-video');
  const detail = $('agent-step-detail');
  const chooser = $('agent-episode');
  const progress = $('agent-progress');
  let episode = null;
  let selected = 0;
  let clipEnd = null;
  let progressSegments = [];
  let modelTotals = [];

  const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const seconds = value => `${Math.max(0, value).toFixed(1)} s`;
  const elapsed = value => {
    const time = Math.max(0, value);
    return time < 60 ? seconds(time) : `${Math.floor(time / 60)}m ${(time % 60).toFixed(1)}s`;
  };
  const tokens = value => Math.round(value).toLocaleString('en-US');
  const dollars = value => `$${value.toFixed(2)}`;
  const kindName = {model: 'Model', tool: 'Tool', robot: 'Robot', context: 'Context'};
  const displayTask = name => name.replaceAll('_', ' ').replace(/^./, first => first.toUpperCase());

  function buildProgress() {
    let totalTokens = 0;
    let totalCost = 0;
    let completeUsage = true;
    modelTotals = episode.steps.map(step => {
      if (step.kind === 'model') {
        if (Number.isFinite(step.tokens) && Number.isFinite(step.cost_usd)) {
          totalTokens += step.tokens;
          totalCost += step.cost_usd;
        } else {
          completeUsage = false;
        }
      }
      return completeUsage ? {tokens: totalTokens, cost: totalCost} : null;
    });
    progressSegments = episode.steps.map(step => el('span', null, `agent-progress-segment ${step.kind}`));
    progressSegments.forEach(segment => segment.setAttribute('aria-hidden', 'true'));
    progress.style.gridTemplateColumns = `repeat(${progressSegments.length}, minmax(0, 1fr))`;
    progress.replaceChildren(...progressSegments);
    progress.setAttribute('aria-valuemax', String(progressSegments.length));
  }

  function select(index) {
    if (!episode || index < 0 || index >= episode.steps.length) return;
    selected = index;
    const step = episode.steps[index];
    video.pause();
    video.playbackRate = 2;
    clipEnd = step.kind === 'robot' && step.end_frame > step.frame
      ? step.end_frame / episode.fps : null;
    video.currentTime = step.frame / episode.fps;
    if (clipEnd != null) video.play().catch(() => {});

    $('agent-step-count').textContent = `Step ${index + 1} of ${episode.steps.length}`;
    progressSegments.forEach((segment, position) => {
      segment.classList.toggle('is-current', position === index);
    });
    progress.setAttribute('aria-valuenow', String(index + 1));
    progress.setAttribute('aria-valuetext', `Step ${index + 1} of ${episode.steps.length}: ${kindName[step.kind] || step.kind}`);
    $('agent-previous').disabled = index === 0;
    $('agent-next').disabled = index === episode.steps.length - 1;
    $('agent-video-note').textContent = clipEnd != null ? 'Robot motion · 2×' : 'Simulator waiting for command';

    detail.replaceChildren();
    const top = el('div', null, 'agent-detail-top');
    top.append(el('span', kindName[step.kind] || step.kind, `agent-kind ${step.kind}`));
    const simSeconds = Math.max(0, (step.end_frame ?? step.frame) - step.frame) / episode.fps;
    const totalSimSeconds = (step.end_frame ?? step.frame) / episode.fps;
    top.append(el('span', `Wall ${seconds(step.end - step.start)} (${elapsed(step.end)} total) · Sim ${seconds(simSeconds)} (${elapsed(totalSimSeconds)} total)`, 'agent-detail-time'));
    detail.append(top);
    if (step.kind !== 'model') detail.append(el('h4', step.title));
    if (step.kind === 'model' && Number.isFinite(step.tokens) && Number.isFinite(step.cost_usd)) {
      const total = modelTotals[index];
      const usage = el('p', `Tokens +${tokens(step.tokens)}${total ? ` (${tokens(total.tokens)} total)` : ''} · Est. API cost +${dollars(step.cost_usd)}${total ? ` (${dollars(total.cost)} total)` : ''}`, 'agent-usage');
      usage.title = 'Token counts include cached input tokens; costs are estimates recorded in the evaluation trace.';
      detail.append(usage);
    }
    if (step.intent) {
      const description = el('p', null, 'agent-intent');
      description.append(el('strong', 'Action description: '), document.createTextNode(step.intent));
      detail.append(description);
    }
    if (step.summary) {
      const summary = step.summary.length > 160 ? `${step.summary.slice(0, 160).trimEnd()}…` : step.summary;
      detail.append(el('p', summary, 'agent-summary'));
      if (step.summary.length > 160) {
        const more = el('details');
        more.append(el('summary', 'Full returned summary'), el('p', step.summary));
        detail.append(more);
      }
    }
    if (step.detail) detail.append(el('p', step.detail, 'agent-muted'));
    if (step.result) detail.append(el('p', `Tool result: ${step.result}`, 'agent-controller'));
    if (step.outcome || step.ticks != null) {
      const status = [step.outcome && `Controller: ${step.outcome}`, step.ticks != null && `${step.ticks} simulator steps`].filter(Boolean).join(' · ');
      detail.append(el('p', status, 'agent-controller'));
    }
    if (step.status && step.status !== 'complete') detail.append(el('p', `Recorded call status: ${step.status}`, 'agent-controller'));
    if (step.code) {
      const code = el('details', null, step.kind === 'robot' ? 'agent-command' : 'agent-tool-call');
      if (step.kind === 'robot') {
        const heading = el('summary');
        heading.append(el('span', 'Robot command · selected arguments', 'agent-command-label'));
        heading.append(el('code', step.code, 'agent-command-preview'));
        code.append(heading, el('pre', step.code));
      } else {
        code.append(el('summary', 'Tool call'), el('pre', step.code));
      }
      detail.append(code);
    }

  }

  async function load(entry) {
    const response = await fetch(entry.data);
    if (!response.ok) throw new Error('Episode data could not be loaded');
    const data = await response.json();
    if (data.version !== 1 || !Array.isArray(data.steps) || !data.steps.length || !data.fps || !data.video)
      throw new Error('Episode data is incomplete');
    episode = data;
    const task = displayTask(data.task);
    video.src = data.video;
    video.setAttribute('aria-label', `${task}, instance ${data.instance}, agent evaluation video`);
    $('agent-rollout-title').textContent = 'Inside an Astra rollout';
    $('agent-episode-result').textContent = `${task} · instance ${data.instance} · ${data.result?.success === true ? 'task succeeded' : data.result?.success === false ? 'task incomplete' : 'outcome unavailable'}${data.result?.q != null ? ` · Q ${Number(data.result.q).toFixed(2)}` : ''}`;
    buildProgress();
    select(0);
  }

  video.addEventListener('timeupdate', () => {
    if (clipEnd != null && video.currentTime >= clipEnd - 0.04) {
      video.pause();
      clipEnd = null;
      $('agent-video-note').textContent = 'Robot motion complete';
    }
  });
  $('agent-previous').addEventListener('click', () => select(selected - 1));
  $('agent-next').addEventListener('click', () => select(selected + 1));
  progress.addEventListener('click', event => {
    if (!episode) return;
    const segment = event.target.closest('.agent-progress-segment');
    const rect = progress.getBoundingClientRect();
    const index = segment ? progressSegments.indexOf(segment) : Math.floor((event.clientX - rect.left) / rect.width * progressSegments.length);
    select(Math.max(0, Math.min(progressSegments.length - 1, index)));
  });
  progress.addEventListener('keydown', event => {
    if (!episode) return;
    if (event.key === 'Home') { event.preventDefault(); select(0); }
    if (event.key === 'End') { event.preventDefault(); select(episode.steps.length - 1); }
  });
  chooser.addEventListener('change', () => load(JSON.parse(chooser.value)).catch(showError));
  $('agent-replay').addEventListener('keydown', event => {
    if (event.target === video || event.target === chooser) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); select(selected - 1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); select(selected + 1); }
  });
  function showError(error) {
    detail.replaceChildren(el('p', error.message || 'Episode could not be loaded.'));
    $('agent-step-count').textContent = 'Unavailable';
  }
  fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/agent-rollouts/manifest.json')
    .then(response => { if (!response.ok) throw new Error('Episode list could not be loaded'); return response.json(); })
    .then(manifest => {
      if (!manifest.episodes?.length) throw new Error('No agent episodes are available');
      for (const entry of manifest.episodes) {
        const option = el('option', `${displayTask(entry.task)} · ${entry.instance}`);
        option.value = JSON.stringify(entry);
        chooser.append(option);
      }
      chooser.hidden = manifest.episodes.length < 2;
      return load(manifest.episodes[0]);
    })
    .catch(showError);
})();
