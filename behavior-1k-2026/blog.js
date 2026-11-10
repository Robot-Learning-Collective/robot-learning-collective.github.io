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

Promise.all([
  fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/episodes/manifest.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }),
  fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/data/tasks_2025.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }),
])
  .then(([manifest, priorTasks]) => {
    const prior = new Set(priorTasks);
    const byTask = new Map();
    for (const episode of manifest.episodes) {
      if (!byTask.has(episode.task)) byTask.set(episode.task, []);
      byTask.get(episode.task).push(episode);
    }
    const tasks = [...byTask].map(([name, episodes]) => {
      episodes.sort((a, b) => a.instance - b.instance);
      return {
        name,
        q: episodes.reduce((sum, episode) => sum + Number(episode.q), 0) / episodes.length,
        firstEpisode: episodes[0].id,
        prior: prior.has(name),
      };
    }).sort((a, b) => b.q - a.q || a.name.localeCompare(b.name));
    if (!tasks.length) throw new Error('No task results are available');

    let activeBar = null;
    for (const task of tasks) {
      const bar = document.createElement('button');
      bar.type = 'button';
      bar.className = `task-bar${task.prior ? '' : ' new-task'}`;
      const fill = document.createElement('span');
      fill.className = 'task-bar-fill';
      fill.style.height = `${Math.max(1, task.q * 100)}%`;
      bar.append(fill);
      bar.setAttribute('aria-pressed', 'false');
      const name = task.name.replaceAll('_', ' ');
      const displayName = name.charAt(0).toUpperCase() + name.slice(1);
      bar.setAttribute('aria-label', `${displayName}, π0.5 Q ${task.q.toFixed(3)}`);
      bar.title = `${displayName} · Q ${task.q.toFixed(3)}`;
      bar.addEventListener('click', () => {
        if (activeBar) activeBar.setAttribute('aria-pressed', 'false');
        activeBar = bar;
        bar.setAttribute('aria-pressed', 'true');
        const taskName = document.createElement('span');
        taskName.className = 'selected-task-name';
        taskName.textContent = displayName;
        const taskScore = document.createElement('span');
        taskScore.className = 'selected-task-score';
        taskScore.textContent = `Q ${task.q.toFixed(3)}`;
        selectedTask.replaceChildren(taskName, taskScore);
        episodeFrame.title = `${displayName} episode viewer`;
        episodeFrame.src = `./episode.html?embed=1&episode=${encodeURIComponent(task.firstEpisode)}`;
      });
      taskBars.append(bar);
    }
    taskBars.firstElementChild.click();
  })
  .catch(() => {
    selectedTask.textContent = 'Task results are unavailable.';
  });

window.addEventListener('message', event => {
  if (event.source !== episodeFrame.contentWindow || event.origin !== location.origin) return;
  if (event.data?.type !== 'episode-viewer-height') return;
  const height = Number(event.data.height);
  if (Number.isFinite(height) && height > 0) episodeFrame.style.height = `${height}px`;
});

// The report's existing SVG components carry the example explanations.
const methodNotes = {
  backbone: {
    title: 'Model backbone',
    detail: 'Our policy is based on π0.5, which pairs PaliGemma (SigLIP + Gemma 2B) with a smaller action expert trained by flow matching to generate action chunks.'
  },
  stage: {
    title: 'Tracking progress',
    detail: 'Skill annotations define the stages. The model predicts the current one from images and task, with voting during inference.'
  },
  attention: {
    title: 'Readable attention',
    detail: 'Registers, scratch tokens and sinks give excess attention somewhere to go. Moving SigLIP outliers into registers made camera attention maps readable.'
  },
  kv: {
    title: 'VLM-to-expert connection',
    detail: 'A learned mixture lets each expert layer read keys and values from every VLM layer, not just the final one.'
  },
  actions: {
    title: '30-step control',
    detail: 'The model denoises 30 consecutive actions; each has 21 degrees of freedom matching the teleoperation controls.'
  },
  conditioning: {
    title: 'Direct task context',
    detail: 'Task and stage embeddings also reach the expert directly through two zero-initialized conditioning tokens.'
  },
  fast: {
    title: 'Training-only supervision',
    detail: 'Predicting discrete action tokens adds an auxiliary training loss. These tokens are absent at inference.'
  },
  flow: {
    title: 'Lower-variance training',
    detail: 'One VLM pass serves 10 expert samples. Stratified times and opposite-noise pairs reduce gradient variance.'
  },
  practice_loop: {
    title: 'Learning by practice',
    detail: 'Astra isolates hard steps, studies failed attempts and human demonstrations, then retries and writes recipes for what works.'
  },
  practice_tools: {
    title: 'Practice-only tools',
    detail: 'Astra can reset, restore saved scenes and inspect goal conditions while practicing. These tools are removed for evaluation.'
  },
  frozen: {
    title: 'Frozen context',
    detail: 'Practice saves a conversation, notes and recipes; no model weights change. Each evaluation episode starts from a fresh copy.'
  },
  observation: {
    title: 'Official observations',
    detail: 'Astra receives the head and wrist RGB-D views plus robot state, and measures positions from depth.'
  },
  controller: {
    title: 'Goal-based control',
    detail: 'Astra sends one motion goal at a time. The controller takes one step at each 30 Hz simulator tick and reports completion or a stall.'
  },
  routing: {
    title: 'Task routing',
    detail: 'We routed 14 tasks to Astra where the learned policy struggled and the agent did better in tests or the policy scored near zero.'
  },
  fallback: {
    title: 'Runtime fallback',
    detail: 'API errors, repeated query timeouts, or cost and wall-time limits hand the rest of the episode to the learned policy. A stalled motion does not.'
  }
};
const methodPopover = document.getElementById('method-popover');
const methodPopoverTitle = document.getElementById('method-popover-title');
const methodPopoverDetail = document.getElementById('method-popover-detail');
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
  methodPopover.hidden = true;
}

function showMethodPopover(topic, pinned = false) {
  if (activeMethodTopic && activeMethodTopic !== topic) {
    activeMethodTopic.setAttribute('aria-expanded', 'false');
  }
  activeMethodTopic = topic;
  methodPopoverPinned = pinned;
  topic.setAttribute('aria-expanded', 'true');
  const note = methodNotes[topic.dataset.methodTopic];
  methodPopoverTitle.textContent = note.title;
  methodPopoverDetail.textContent = note.detail;
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
      {box: [0, 0, 1170, 690], label: 'VLM and layer connection', topics: ['backbone', 'stage', 'attention', 'fast', 'kv']},
      {box: [1140, 0, 540, 690], label: 'Action expert', topics: ['actions', 'conditioning', 'flow']},
    ] : [
      {box: [40, 0, 660, 610], label: 'Practice', topics: ['practice_loop', 'practice_tools']},
      {box: [770, 160, 180, 260], label: 'Frozen context', topics: ['frozen']},
      {box: [1000, 0, 640, 610], label: 'Agent evaluation', topics: ['observation', 'controller']},
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
