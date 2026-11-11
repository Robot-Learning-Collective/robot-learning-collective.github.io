const tabs = document.getElementById('example-tabs');
const video = document.getElementById('example-video');
const instruction = document.getElementById('example-instruction');

fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/training-examples/manifest.json')
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((examples) => {
    function select(index) {
      const example = examples[index];
      video.pause();
      video.src = example.src;
      video.load();
      video.play().catch(() => {});
      instruction.textContent = `Instruction: ${example.instruction}`;
      tabs.querySelectorAll('button').forEach((button, buttonIndex) => {
        button.setAttribute('aria-selected', String(buttonIndex === index));
      });
    }

    examples.forEach((example, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.role = 'tab';
      button.textContent = example.label;
      button.setAttribute('aria-selected', 'false');
      button.addEventListener('click', () => select(index));
      tabs.appendChild(button);
    });
    if (examples.length) select(0);
  })
  .catch(() => {
    tabs.textContent = 'Training examples are unavailable.';
  });

const taskBars = document.getElementById('task-bars');
const selectedTask = document.getElementById('selected-task');
const episodeFrame = document.getElementById('episode-frame');
const taskTooltip = document.getElementById('task-tooltip');
const EXHIBIT = 'https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/';
const getJson = path => fetch(EXHIBIT + path).then(response => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
});

function displayTaskName(name) {
  const words = name.replaceAll('_', ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function scoreBadge(text, className) {
  const badge = document.createElement('span');
  badge.className = className;
  badge.textContent = text;
  return badge;
}

function showTaskTooltip(bar, task) {
  const lines = [`<strong>${displayTaskName(task.name)}</strong>`, `<span class="tip-vla">VLA Q ${task.q.toFixed(2)}</span>`];
  if (task.astra) {
    lines.push(`<span class="tip-astra">Astra Q ${task.astra.q.toFixed(2)} · ${task.astra.episodes} episode${task.astra.episodes === 1 ? '' : 's'}</span>`);
    lines.push('<em>Played by Astra in our submission</em>');
  }
  taskTooltip.innerHTML = lines.join('');
  taskTooltip.hidden = false;
  const rect = bar.getBoundingClientRect();
  const width = taskTooltip.offsetWidth;
  const height = taskTooltip.offsetHeight;
  const left = Math.max(8, Math.min(rect.left + rect.width / 2 - width / 2, innerWidth - width - 8));
  const top = rect.top + bar.offsetHeight * (1 - task.final) - height - 10;
  taskTooltip.style.left = `${left}px`;
  taskTooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTaskTooltip() {
  taskTooltip.hidden = true;
}

Promise.all([getJson('episodes/manifest.json'), getJson('data/tasks_2025.json'), getJson('data/astra_q_by_task.json')])
  .then(([manifest, priorTasks, astraData]) => {
    const prior = new Set(priorTasks);
    const byTask = new Map();
    for (const episode of manifest.episodes) {
      if (!byTask.has(episode.task)) byTask.set(episode.task, []);
      byTask.get(episode.task).push(episode);
    }
    const tasks = [...byTask].map(([name, episodes]) => {
      episodes.sort((a, b) => a.instance - b.instance);
      const q = episodes.reduce((sum, episode) => sum + Number(episode.q), 0) / episodes.length;
      const routed = astraData.astra_by_task[name];
      const astra = routed ? { q: routed.q_sum / routed.episodes, episodes: routed.episodes } : null;
      return { name, q, astra, final: astra ? astra.q : q, firstEpisode: episodes[0].id, prior: prior.has(name) };
    }).sort((a, b) => b.final - a.final || b.q - a.q || a.name.localeCompare(b.name));
    if (!tasks.length) throw new Error('No task results are available');

    let activeBar = null;
    for (const task of tasks) {
      const bar = document.createElement('button');
      bar.type = 'button';
      bar.className = `task-bar${task.prior ? '' : ' new-task'}${task.astra ? ' astra-task' : ''}`;
      if (task.astra) {
        const gain = document.createElement('span');
        gain.className = 'task-bar-gain';
        gain.style.height = `${Math.max(0, task.astra.q - task.q) * 100}%`;
        bar.append(gain);
      }
      const fill = document.createElement('span');
      fill.className = 'task-bar-fill';
      fill.style.height = `${Math.max(task.astra ? 0 : 1, task.q * 100)}%`;
      bar.append(fill);
      bar.setAttribute('aria-pressed', 'false');
      const name = displayTaskName(task.name);
      bar.setAttribute('aria-label', `${name}, VLA Q ${task.q.toFixed(2)}${task.astra ? `, Astra Q ${task.astra.q.toFixed(2)}` : ''}`);
      bar.addEventListener('pointerenter', () => showTaskTooltip(bar, task));
      bar.addEventListener('pointerleave', hideTaskTooltip);
      bar.addEventListener('focus', () => showTaskTooltip(bar, task));
      bar.addEventListener('blur', hideTaskTooltip);
      bar.addEventListener('click', () => {
        if (activeBar) activeBar.setAttribute('aria-pressed', 'false');
        activeBar = bar;
        bar.setAttribute('aria-pressed', 'true');
        const taskName = scoreBadge(name, 'selected-task-name');
        const badges = [scoreBadge(`VLA Q ${task.q.toFixed(2)}`, 'selected-task-score')];
        if (task.astra) {
          badges.push(scoreBadge(`Astra Q ${task.astra.q.toFixed(2)}`, 'selected-task-score astra'));
          badges.push(scoreBadge('the rollout below is the VLA’s', 'selected-task-note'));
        }
        selectedTask.replaceChildren(taskName, ...badges);
        episodeFrame.title = `${name} episode viewer`;
        episodeFrame.src = `./episode.html?embed=1&episode=${encodeURIComponent(task.firstEpisode)}`;
      });
      taskBars.append(bar);
    }
    taskBars.firstElementChild.click();
  })
  .catch(() => {
    selectedTask.textContent = 'Task results are unavailable.';
  });
window.addEventListener('scroll', hideTaskTooltip, { passive: true });

window.addEventListener('message', event => {
  if (event.source !== episodeFrame.contentWindow || event.origin !== location.origin) return;
  if (event.data?.type !== 'episode-viewer-height') return;
  const height = Number(event.data.height);
  if (Number.isFinite(height) && height > 0) episodeFrame.style.height = `${height}px`;
});

// Explanations for the clickable parts of the two diagrams. origin: base = π0.5 as released,
// 2025 = our 2025 changes, 2026 = new this year, ours = the Astra agent.
const methodNotes = {
  backbone: { origin: 'base', title: 'π0.5 backbone', detail: 'PaliGemma (SigLIP plus Gemma 2B) as the VLM, as released in openpi. The highlighted parts are what we changed on top of it.' },
  vision: { origin: 'base', title: 'Vision encoder', detail: 'SigLIP encodes the head and both wrist cameras at 224×224, 256 tokens each. New in 2026: a learned projection per camera after the shared encoder.' },
  expert: { origin: 'base', title: 'Action expert', detail: 'A smaller 300M Gemma that turns noise into actions by flow matching, conditioned on the flow time, as in π0.5.' },
  task: { origin: '2025', title: 'Task embeddings instead of text', detail: 'Each task has a learned embedding and the prompt is never tokenized. In 2026 each new task started from a weighted average of the embeddings of 1–5 similar 2025 tasks.' },
  stage_head: { origin: '2025', title: 'Stage head (“System 2”)', detail: 'The model predicts which stage of the task it is in, from the images and the task alone. At inference the stage is tracked by voting over the last 120 steps.' },
  kv: { origin: '2025', title: 'KV mixing', detail: 'Each action-expert layer reads a learnable mixture of the keys and values of all VLM layers, not just the matching layer.' },
  fast: { origin: '2025', title: 'FAST auxiliary loss', detail: 'During training only, the VLM also predicts discrete FAST tokens of the target actions. In 2026 we capped them at 68 tokens, saving about 9% of the step time.' },
  stage_tokens: { origin: '2026', title: 'Stages from skill annotations', detail: 'In 2025 a stage was an equal time slice of the episode. Now 413 stages (2–9 per task) follow the real steps of each task, taken from the demonstrations’ skill annotations.' },
  state: { origin: '2026', title: 'Per-joint state tokens', detail: 'π0.5 gives every joint the same token for the same bin. We add a learned embedding per joint and a sine/cosine code of the unclipped value, plus one continuous token for the whole state.' },
  actions: { origin: '2026', title: 'Actions that match teleoperation', detail: '21 dimensions instead of 32: the torso is controlled by translate and tilt, as the teleoperation rig moves it, and the padding is gone. The model predicts 30 actions at a time.' },
  conditioning: { origin: '2026', title: 'Direct task and stage input', detail: 'Two zero-initialized tokens carry the task and stage embeddings straight into the action expert, instead of only through the VLM.' },
  flow: { origin: '2026', title: 'Lower-variance flow matching', detail: 'Since 2025, one VLM pass serves several expert samples. New in 2026: stratified flow times and antithetic noise cut the noise from flow sampling by more than half, and a simple noise-versus-compute model picks K = 10.' },
  registers: { origin: '2026', title: 'Registers', detail: 'Eight learnable tokens per camera inside SigLIP. A few image cells were soaking up attention because of their position embeddings; moving them into registers made the attention maps readable.' },
  scratch: { origin: '2026', title: 'Scratch tokens', detail: 'Learnable tokens with no input of their own, 8 in the VLM and 4 in the action expert, give attention somewhere to go.' },
  sinks: { origin: '2026', title: 'Attention sinks', detail: 'Every attention layer of both transformers has a learnable sink, so attention with nowhere useful to go can park there and still return a learned value.' },
  practice_agent: { origin: 'ours', title: 'GPT-6 Astra in practice', detail: 'Astra runs in the Pi coding agent with one simulator. It gets the task instruction, the goal definition, the human demonstrations and lessons from earlier practice sessions.' },
  practice_loop: { origin: 'ours', title: 'Learning by practice', detail: 'Astra lists the hard parts of the task, then practices one at a time: it saves the scene, tries, compares its motion with the human demonstrations, and writes a recipe when a part works. Finally it runs the whole task from a fresh reset.' },
  practice_tools: { origin: 'ours', title: 'Practice-only tools', detail: 'While practicing, Astra can reset the scene, save and restore states, and check which goal conditions hold. None of this exists during evaluation.' },
  practice_outputs: { origin: 'ours', title: 'What practice produces', detail: 'A practice conversation of 29–395 MB per task, plus notes on what works and what fails, and short recipes for sub-skills. No weights are trained.' },
  frozen: { origin: 'ours', title: 'Frozen context', detail: 'At the end of practice the session is saved and never changed again. Every evaluation episode starts from a fresh copy, so nothing carries over between episodes.' },
  episode_agent: { origin: 'ours', title: 'Astra in evaluation', detail: 'Astra writes short programs that look at the scene, measure positions from depth, and send one goal at a time. While it thinks, the simulation waits.' },
  observation: { origin: 'ours', title: 'Official observations only', detail: 'The same input our VLA gets: the head and wrist camera images with depth, and the robot state. Nothing reports contact or force.' },
  controller: { origin: 'ours', title: 'Goal-based control', detail: 'Astra sends one goal at a time, such as a gripper pose or a base motion. The controller takes one step toward it at every 30 Hz simulator step and reports when the goal is reached or the motion stalls.' },
  routing: { origin: 'ours', title: 'Task routing', detail: 'The task decides who plays an episode: Astra on 14 tasks where the VLA struggled and the agent did better in our tests or the VLA scored about zero, the VLA everywhere else.' },
  fallback: { origin: 'ours', title: 'Runtime fallback', detail: 'API errors, repeated query timeouts, or cost and wall-time limits hand the rest of the episode to the VLA. A motion that misses its goal does not: Astra just plans again.' },
};
const originLabels = { base: 'From π0.5', 2025: 'Our 2025 change', 2026: 'New in 2026', ours: '' };
const methodPopover = document.getElementById('method-popover');
const methodPopoverTitle = document.getElementById('method-popover-title');
const methodPopoverDetail = document.getElementById('method-popover-detail');
const methodPopoverTag = document.getElementById('method-popover-tag');
const methodPopoverClose = document.getElementById('method-popover-close');
let activeMethodTopic = null;
let methodPopoverPinned = false;

function positionMethodPopover() {
  if (!activeMethodTopic || matchMedia('(max-width: 42rem)').matches) return;
  const rect = activeMethodTopic.getBoundingClientRect();
  const width = methodPopover.offsetWidth;
  const height = methodPopover.offsetHeight;
  methodPopover.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - width - 12))}px`;
  const below = rect.bottom + 10;
  methodPopover.style.top = `${below + height <= innerHeight - 12 ? below : Math.max(12, rect.top - height - 10)}px`;
}

function hideMethodPopover() {
  if (activeMethodTopic) activeMethodTopic.setAttribute('aria-expanded', 'false');
  activeMethodTopic = null;
  methodPopoverPinned = false;
  methodPopover.classList.remove('is-pinned');
  methodPopover.hidden = true;
}

function showMethodPopover(topic, pinned = false) {
  if (activeMethodTopic && activeMethodTopic !== topic) {
    activeMethodTopic.setAttribute('aria-expanded', 'false');
  }
  activeMethodTopic = topic;
  methodPopoverPinned = pinned;
  methodPopover.classList.toggle('is-pinned', pinned);
  topic.setAttribute('aria-expanded', 'true');
  const note = methodNotes[topic.dataset.methodTopic];
  methodPopoverTitle.textContent = note.title;
  methodPopoverDetail.textContent = note.detail;
  methodPopoverTag.textContent = originLabels[note.origin] || '';
  methodPopoverTag.dataset.origin = note.origin;
  methodPopoverTag.hidden = !methodPopoverTag.textContent;
  methodPopover.hidden = false;
  positionMethodPopover();
}

function wireMethodTopic(topic) {
  const note = methodNotes[topic.dataset.methodTopic];
  if (!note) return;
  topic.setAttribute('role', 'button');
  topic.setAttribute('tabindex', '0');
  topic.setAttribute('aria-label', `${note.title}: show explanation`);
  topic.setAttribute('aria-controls', 'method-popover');
  topic.setAttribute('aria-expanded', 'false');
  topic.addEventListener('pointerenter', () => {
    if (matchMedia('(hover: hover)').matches && !methodPopoverPinned) showMethodPopover(topic);
  });
  topic.addEventListener('pointerleave', () => {
    if (!methodPopoverPinned && document.activeElement !== topic) hideMethodPopover();
  });
  topic.addEventListener('focus', () => {
    if (!methodPopoverPinned) showMethodPopover(topic);
  });
  topic.addEventListener('blur', () => {
    if (!methodPopoverPinned) hideMethodPopover();
  });
  topic.addEventListener('click', () => {
    if (activeMethodTopic === topic && methodPopoverPinned) hideMethodPopover();
    else showMethodPopover(topic, true);
  });
  topic.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    topic.dispatchEvent(new MouseEvent('click', {bubbles: true}));
  });
}

// Size each part's outline and marker from its rendered text, in the visitor's own fonts.
// The SVG files carry pre-measured boxes as a fallback; this corrects them once fonts load.
function fitTopicBoxes(figure, source) {
  const probe = document.importNode(source.documentElement, true);
  probe.setAttribute('aria-hidden', 'true');
  probe.style.cssText = 'position:absolute;left:-10000px;top:0;width:1680px;height:auto;visibility:hidden;pointer-events:none';
  document.body.append(probe);
  const padX = 9, padY = 8;
  const measure = () => {
    const boxes = new Map();
    probe.querySelectorAll('[data-method-topic]').forEach(group => {
      let box = null;
      for (const child of group.children) {
        if (child.tagName === 'path' || child.matches('.topic-hl, .topic-marker')) continue;
        const b = child.getBBox();
        if (!b.width && !b.height) continue;
        box = box
          ? { x0: Math.min(box.x0, b.x), y0: Math.min(box.y0, b.y), x1: Math.max(box.x1, b.x + b.width), y1: Math.max(box.y1, b.y + b.height) }
          : { x0: b.x, y0: b.y, x1: b.x + b.width, y1: b.y + b.height };
      }
      if (box) boxes.set(group.dataset.key, box);
    });
    figure.querySelectorAll('svg [data-method-topic]').forEach(group => {
      const box = boxes.get(group.dataset.key);
      if (!box) return;
      const x = box.x0 - padX, y = box.y0 - padY, w = box.x1 - box.x0 + 2 * padX, h = box.y1 - box.y0 + 2 * padY;
      const rect = group.querySelector('.topic-hl');
      rect?.setAttribute('x', x.toFixed(1));
      rect?.setAttribute('y', y.toFixed(1));
      rect?.setAttribute('width', w.toFixed(1));
      rect?.setAttribute('height', h.toFixed(1));
      group.querySelectorAll('.topic-marker circle').forEach(circle => {
        circle.setAttribute('cx', (x + w).toFixed(1));
        circle.setAttribute('cy', y.toFixed(1));
      });
    });
  };
  requestAnimationFrame(() => document.fonts.ready.then(measure));
  document.fonts.addEventListener?.('loadingdone', measure);
}

document.querySelectorAll('.method-figure').forEach(async figure => {
  try {
    const response = await fetch(figure.dataset.svg);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const source = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (source.querySelector('parsererror')) throw new Error('Invalid SVG');
    const image = figure.querySelector('img');
    const desktop = document.importNode(source.documentElement, true);
    desktop.classList.add('method-desktop-figure');
    desktop.setAttribute('role', 'group');
    desktop.setAttribute('aria-label', image.alt);
    desktop.querySelectorAll('[data-method-topic]').forEach(wireMethodTopic);

    const parts = figure.dataset.svg.includes('architecture') ? [
      {box: [0, 0, 1170, 690], label: 'VLM and layer connection', topics: ['backbone', 'vision', 'task', 'stage_head', 'stage_tokens', 'state', 'fast', 'kv', 'registers', 'scratch', 'sinks']},
      {box: [1140, 0, 540, 690], label: 'Action expert', topics: ['expert', 'actions', 'conditioning', 'flow', 'scratch', 'sinks']},
    ] : [
      {box: [40, 0, 660, 610], label: 'Practice', topics: ['practice_agent', 'practice_loop', 'practice_tools', 'practice_outputs']},
      {box: [770, 160, 180, 260], label: 'Frozen context', topics: ['frozen']},
      {box: [1000, 0, 640, 610], label: 'Agent evaluation', topics: ['episode_agent', 'observation', 'controller']},
      {box: [40, 610, 850, 190], label: 'Task routing', topics: ['routing']},
      {box: [870, 610, 770, 190], label: 'Runtime fallback', topics: ['fallback']},
    ];
    const mobile = parts.map(({box, label, topics}) => {
      const svg = document.importNode(source.documentElement, true);
      svg.classList.add('method-mobile-part');
      svg.setAttribute('viewBox', box.join(' '));
      svg.setAttribute('width', box[2]);
      svg.setAttribute('height', box[3]);
      svg.setAttribute('role', 'group');
      svg.setAttribute('aria-label', label);
      svg.querySelectorAll('[data-method-topic]').forEach(topic => {
        if (topics.includes(topic.dataset.methodTopic)) wireMethodTopic(topic);
        else topic.remove();
      });
      if (figure.dataset.svg.includes('architecture')) {
        svg.querySelectorAll('text[x="1180"]').forEach(text => text.remove());
      }
      return svg;
    });
    image.replaceWith(desktop, ...mobile);
    fitTopicBoxes(figure, source);
  } catch (error) {
    // The SVG image remains visible if a browser cannot load inline SVG.
  }
});

methodPopoverClose.addEventListener('click', hideMethodPopover);
document.addEventListener('pointerdown', event => {
  if (methodPopoverPinned && !methodPopover.contains(event.target) && !event.target.closest('[data-method-topic]')) {
    hideMethodPopover();
  }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') hideMethodPopover();
});
window.addEventListener('resize', positionMethodPopover);
document.addEventListener('scroll', positionMethodPopover, true);

document.querySelectorAll('.diagram-toolbar .segmented').forEach(group => {
  const figure = document.getElementById(group.closest('.diagram-toolbar').dataset.figure);
  const options = [...group.querySelectorAll('[role="radio"]')];
  const choose = option => {
    options.forEach(other => {
      other.setAttribute('aria-checked', String(other === option));
      other.tabIndex = other === option ? 0 : -1;
    });
    figure.dataset.highlight = option.dataset.highlight;
  };
  options.forEach((option, index) => {
    option.tabIndex = option.getAttribute('aria-checked') === 'true' ? 0 : -1;
    option.addEventListener('click', () => choose(option));
    option.addEventListener('keydown', event => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = options[(index + step + options.length) % options.length];
      choose(next);
      next.focus();
    });
  });
});

// Pulse the hotspot markers once, the first time each diagram scrolls into view.
const introObserver = 'IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches
  ? new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add('is-intro');
        setTimeout(() => entry.target.classList.remove('is-intro'), 2600);
        introObserver.unobserve(entry.target);
      }
    }, { threshold: 0.45 })
  : null;
document.querySelectorAll('.method-figure').forEach(figure => introObserver?.observe(figure));

const copyCitation = document.getElementById('copy-citation');
copyCitation?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.getElementById('citation-bibtex').textContent);
    copyCitation.textContent = 'Copied';
  } catch (error) {
    getSelection().selectAllChildren(document.getElementById('citation-bibtex'));
    copyCitation.textContent = 'Press Ctrl+C';
  }
  setTimeout(() => { copyCitation.textContent = 'Copy'; }, 2000);
});

const pageGuide = document.querySelector('.page-guide');
const pageGuideToggle = pageGuide.querySelector('.page-guide-toggle');
const pageGuideCurrent = document.getElementById('page-guide-current');
const guideSections = [...pageGuide.querySelectorAll('[data-guide-section]')].map(link => ({
  link,
  entry: link.parentElement,
  target: document.querySelector(link.getAttribute('href')),
}));

function closePageGuide() {
  pageGuide.classList.remove('is-open');
  pageGuideToggle.setAttribute('aria-expanded', 'false');
}

function updatePageGuide() {
  const activationLine = matchMedia('(max-width: 75rem)').matches ? 80 : 160;
  let current = guideSections[0];
  for (const section of guideSections) {
    if (section.target.getBoundingClientRect().top <= activationLine) current = section;
    else break;
  }
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
    current = guideSections.at(-1);
  }
  for (const section of guideSections) {
    const active = section === current;
    section.entry.classList.toggle('is-active', active);
    if (active) section.link.setAttribute('aria-current', 'location');
    else section.link.removeAttribute('aria-current');
    let currentSubsection = null;
    if (active) {
      for (const link of section.entry.querySelectorAll('.page-guide-sublist a')) {
        if (document.querySelector(link.getAttribute('href')).getBoundingClientRect().top <= activationLine + 16) {
          currentSubsection = link;
        }
      }
    }
    for (const link of section.entry.querySelectorAll('.page-guide-sublist a')) {
      if (link === currentSubsection) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
  pageGuideCurrent.textContent = current.link.textContent;
  pageGuide.style.setProperty('--page-guide-header-bottom', `${document.querySelector('.page-header').getBoundingClientRect().bottom}px`);
  pageGuide.classList.add('is-ready');
}

let guideUpdatePending = false;
function queuePageGuideUpdate() {
  if (guideUpdatePending) return;
  guideUpdatePending = true;
  requestAnimationFrame(() => {
    guideUpdatePending = false;
    updatePageGuide();
  });
}

pageGuideToggle.addEventListener('click', () => {
  const open = !pageGuide.classList.contains('is-open');
  pageGuide.classList.toggle('is-open', open);
  pageGuideToggle.setAttribute('aria-expanded', String(open));
});
pageGuide.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', closePageGuide));
document.addEventListener('pointerdown', event => {
  if (!pageGuide.contains(event.target)) closePageGuide();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closePageGuide();
});
window.addEventListener('scroll', queuePageGuideUpdate, { passive: true });
window.addEventListener('resize', queuePageGuideUpdate);
updatePageGuide();
