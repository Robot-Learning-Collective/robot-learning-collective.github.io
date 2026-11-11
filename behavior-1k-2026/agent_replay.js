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
  const kindName = {model: 'Model', tool: 'Tool', robot: 'Robot', context: 'Context'};
  const displayTask = name => name.replaceAll('_', ' ').replace(/^./, first => first.toUpperCase());
  const jsonDetails = (label, value) => {
    const box = el('details');
    box.append(el('summary', label), el('pre', JSON.stringify(value, null, 2)));
    return box;
  };

  function frameAtWall(seconds) {
    const states = episode.agent.states;
    let left = 0, right = states.length;
    while (left < right) {
      const middle = (left + right) >> 1;
      if (states[middle].wall_seconds <= seconds) left = middle + 1;
      else right = middle;
    }
    return Math.max(0, (states[Math.max(0, left - 1)]?.step || 1) - 1);
  }

  function stateSummary(frame) {
    const state = episode.agent.states[Math.min(episode.agent.states.length - 1, frame + 1)];
    const box = el('div', null, 'agent-state-summary');
    const hand = side => state?.held?.[side]?.label || 'empty';
    box.append(el('span', `Left hand: ${hand('left')} · Right hand: ${hand('right')}`));
    if (state?.goal) {
      const satisfied = state.goal.satisfied || [];
      const unsatisfied = state.goal.unsatisfied || [];
      box.append(el('span', `Goal conditions: ${satisfied.length}/${satisfied.length + unsatisfied.length}`));
      if (episode.metadata.goal_conditions?.length) {
        const list = el('ul');
        episode.metadata.goal_conditions.forEach(goal => list.append(el('li', `${satisfied.includes(goal.index) ? '✓' : '○'} ${goal.text}`)));
        const more = el('details');
        more.append(el('summary', 'Task progress'), list);
        box.append(more);
      }
    }
    box.append(el('small', 'Evaluator annotations; the agent may not know these facts.'));
    return box;
  }

  function buildProgress() {
    progressSegments = episode.steps.map(step => el('span', null, `agent-progress-segment ${step.kind}`));
    progressSegments.forEach((segment, index) => {
      segment.setAttribute('aria-hidden', 'true');
      segment.title = `${elapsed(episode.steps[index].start_seconds)} · ${episode.steps[index].label}`;
    });
    progress.style.gridTemplateColumns = `repeat(${progressSegments.length}, minmax(0, 1fr))`;
    progress.replaceChildren(...progressSegments);
    progress.setAttribute('aria-valuemax', String(progressSegments.length));
    $('agent-index').max = String(progressSegments.length);
  }

  function select(index) {
    if (!episode || index < 0 || index >= episode.steps.length) return;
    selected = index;
    const step = episode.steps[index];
    const frame = Math.min(episode.metadata.steps - 1, Math.max(0, step.video_frame || 0));
    video.pause();
    video.playbackRate = 2;
    clipEnd = step.kind === 'robot' && step.end_seconds != null
      ? Math.max(frame, frameAtWall(step.end_seconds)) / episode.metadata.video_fps : null;
    video.currentTime = frame / episode.metadata.video_fps;
    if (clipEnd != null && clipEnd > video.currentTime + 0.05) video.play().catch(() => {});

    $('agent-step-count').textContent = `Activity ${index + 1} of ${episode.steps.length}`;
    progressSegments.forEach((segment, position) => {
      segment.classList.toggle('is-current', position === index);
    });
    progress.setAttribute('aria-valuenow', String(index + 1));
    progress.setAttribute('aria-valuetext', `${kindName[step.kind] || step.kind}: ${step.label}`);
    $('agent-index').value = String(index + 1);
    $('agent-previous').disabled = index === 0;
    $('agent-next').disabled = index === episode.steps.length - 1;
    $('agent-video-note').textContent = clipEnd != null ? 'Robot motion · 2×' : 'Simulator waiting for command';

    detail.replaceChildren();
    const top = el('div', null, 'agent-detail-top');
    top.append(el('span', kindName[step.kind] || step.kind, `agent-kind ${step.kind}`));
    top.append(el('span', `${elapsed(step.start_seconds)} wall · ${elapsed((step.end_seconds ?? episode.agent.duration_seconds) - step.start_seconds)} activity · ${elapsed(frame / episode.metadata.video_fps)} sim`, 'agent-detail-time'));
    detail.append(top, el('h4', step.label || 'Recorded activity'), stateSummary(frame));
    if (step.summary) detail.append(el('p', step.summary, 'agent-summary-text'));
    else if (step.kind === 'model') detail.append(el('p', 'No readable reasoning summary was returned for this call.', 'agent-muted'));
    if (step.text) detail.append(el('p', step.text));
    if (step.status && step.status !== 'complete') detail.append(el('p', `Status: ${step.status}`, 'agent-controller'));
    if (step.images?.length) {
      const gallery = el('div', null, 'agent-observations');
      step.images.forEach(url => {
        const image = el('img');
        image.src = url;
        image.alt = 'Observation delivered to the agent';
        image.loading = 'lazy';
        gallery.append(image);
      });
      detail.append(gallery);
    }
    if (step.arguments != null) detail.append(jsonDetails('Tool arguments', step.arguments));
    if (step.result != null) detail.append(jsonDetails('Tool result', step.result));
    if (step.usage != null) detail.append(jsonDetails('Token usage and estimated cost', step.usage));
  }

  async function load(entry) {
    detail.replaceChildren(el('p', 'Loading rollout…'));
    const response = await fetch(entry.data);
    if (!response.ok) throw new Error('Rollout data could not be loaded');
    const data = await response.json();
    if (data.version !== 2 || !data.agent?.spans?.length || !data.agent?.states?.length || !data.metadata?.video_fps || !data.video)
      throw new Error('Full rollout data is incomplete');
    episode = {...data, steps: data.agent.spans};
    const task = displayTask(data.metadata.task);
    video.src = data.video;
    video.setAttribute('aria-label', `${task}, instance ${data.metadata.instance_id}, agent evaluation video`);
    const q = data.metadata.metrics?.q_score?.final;
    const result = data.metadata.success === true ? 'task succeeded' : data.metadata.success === false ? 'task incomplete' : 'outcome unavailable';
    $('agent-episode-result').textContent = `${task} · instance ${data.metadata.instance_id} · ${result}${q != null ? ` · Q ${Number(q).toFixed(2)}` : ''}`;
    const models = data.agent.spans.filter(span => span.kind === 'model');
    $('agent-availability').textContent = `${data.agent.spans.length} recorded activities · ${models.filter(span => span.summary).length} of ${models.length} model calls returned a readable summary`;
    $('agent-next-summary').disabled = !models.some(span => span.summary);
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
  video.addEventListener('play', () => {
    if (clipEnd == null) $('agent-video-note').textContent = 'Episode playback';
  });
  video.addEventListener('seeking', () => {
    if (clipEnd == null) $('agent-video-note').textContent = 'Episode playback';
  });
  $('agent-previous').addEventListener('click', () => select(selected - 1));
  $('agent-next').addEventListener('click', () => select(selected + 1));
  $('agent-next-summary').addEventListener('click', () => {
    if (!episode) return;
    const next = episode.steps.findIndex((step, index) => index > selected && step.kind === 'model' && step.summary);
    const first = episode.steps.findIndex(step => step.kind === 'model' && step.summary);
    select(next < 0 ? first : next);
  });
  $('agent-index').addEventListener('input', event => select(Number(event.target.value) - 1));
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
  fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/agent-rollouts/full/manifest.json?v=20261111-full')
    .then(response => { if (!response.ok) throw new Error('Episode list could not be loaded'); return response.json(); })
    .then(manifest => {
      if (!manifest.episodes?.length) throw new Error('No agent rollouts are available');
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
