const tabs = document.getElementById('example-tabs');
const video = document.getElementById('example-video');
const instruction = document.getElementById('example-instruction');

fetch('./task_examples.json')
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
  fetch('./data/manifest.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }),
  fetch('./data/tasks_2025.json').then(response => {
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
        selectedTask.textContent = `Selected task: ${displayName} · π0.5 Q ${task.q.toFixed(3)}`;
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

// Keep the report figures as real SVG elements so they remain sharp and inspectable.
document.querySelectorAll('.method-figure').forEach(async figure => {
  try {
    const response = await fetch(figure.dataset.svg);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const source = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (source.querySelector('parsererror')) throw new Error('Invalid SVG');
    const svg = document.importNode(source.documentElement, true);
    const image = figure.querySelector('img');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', image.alt);
    image.replaceWith(svg);
  } catch (error) {
    // The SVG image remains visible if a browser cannot load inline SVG.
  }
});

const methodPopover = document.getElementById('method-popover');
const methodPopoverTitle = document.getElementById('method-popover-title');
const methodPopoverDetail = document.getElementById('method-popover-detail');
const methodPopoverClose = document.getElementById('method-popover-close');
const methodHotspots = [...document.querySelectorAll('.method-hotspot')];
let activeMethodHotspot = null;
let methodPopoverPinned = false;

function positionMethodPopover() {
  if (!activeMethodHotspot || matchMedia('(max-width: 42rem)').matches) return;
  const rect = activeMethodHotspot.getBoundingClientRect();
  const width = methodPopover.offsetWidth;
  const height = methodPopover.offsetHeight;
  methodPopover.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - width - 12))}px`;
  const below = rect.bottom + 10;
  methodPopover.style.top = `${below + height <= innerHeight - 12 ? below : Math.max(12, rect.top - height - 10)}px`;
}

function hideMethodPopover() {
  if (activeMethodHotspot) activeMethodHotspot.setAttribute('aria-expanded', 'false');
  activeMethodHotspot = null;
  methodPopoverPinned = false;
  methodPopover.hidden = true;
}

function showMethodPopover(button, pinned = false) {
  if (activeMethodHotspot && activeMethodHotspot !== button) {
    activeMethodHotspot.setAttribute('aria-expanded', 'false');
  }
  activeMethodHotspot = button;
  methodPopoverPinned = pinned;
  button.setAttribute('aria-expanded', 'true');
  methodPopoverTitle.textContent = button.dataset.title;
  methodPopoverDetail.textContent = button.dataset.detail;
  methodPopover.hidden = false;
  positionMethodPopover();
}

for (const button of methodHotspots) {
  button.setAttribute('aria-controls', 'method-popover');
  button.addEventListener('pointerenter', () => {
    if (matchMedia('(hover: hover)').matches && !methodPopoverPinned) showMethodPopover(button);
  });
  button.addEventListener('pointerleave', () => {
    if (!methodPopoverPinned && document.activeElement !== button) hideMethodPopover();
  });
  button.addEventListener('focus', () => {
    if (!methodPopoverPinned) showMethodPopover(button);
  });
  button.addEventListener('blur', () => {
    if (!methodPopoverPinned) hideMethodPopover();
  });
  button.addEventListener('click', () => {
    if (activeMethodHotspot === button && methodPopoverPinned) hideMethodPopover();
    else showMethodPopover(button, true);
  });
}

methodPopoverClose.addEventListener('click', hideMethodPopover);
document.addEventListener('pointerdown', event => {
  if (methodPopoverPinned && !methodPopover.contains(event.target) && !event.target.closest('.method-hotspot')) {
    hideMethodPopover();
  }
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') hideMethodPopover();
});
window.addEventListener('resize', positionMethodPopover);
document.addEventListener('scroll', positionMethodPopover, true);
