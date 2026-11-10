const astraChart = document.getElementById('astra-q-chart');

Promise.all([
  fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/episodes/manifest.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }),
  fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/data/astra_q_by_task.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }),
]).then(([manifest, data]) => {
  const byTask = new Map();
  for (const episode of manifest.episodes) {
    if (!byTask.has(episode.task)) byTask.set(episode.task, []);
    byTask.get(episode.task).push(Number(episode.q));
  }
  const tasks = [...byTask].map(([name, scores]) => ({
    name,
    policyQ: scores.reduce((sum, q) => sum + q, 0) / scores.length,
    astra: data.astra_by_task[name],
  })).sort((a, b) => b.policyQ - a.policyQ || a.name.localeCompare(b.name));
  if (tasks.length !== 100 || Object.keys(data.astra_by_task).length !== 14) {
    throw new Error('Unexpected task count');
  }

  const ns = 'http://www.w3.org/2000/svg';
  const make = (tag, attrs = {}) => {
    const element = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
    return element;
  };
  const svg = make('svg', { viewBox: '0 0 820 180', role: 'img', 'aria-labelledby': 'astra-chart-title astra-chart-desc' });
  const title = make('title', { id: 'astra-chart-title' });
  title.textContent = 'Per-task Q score with Astra gains';
  const description = make('desc', { id: 'astra-chart-desc' });
  description.textContent = 'One hundred tasks sorted by learned-policy Q score. Blue bars show the learned policy. Green segments show the additional Q achieved by Astra on fourteen selected tasks.';
  svg.append(title, description);

  const left = 32;
  const bottom = 150;
  const height = 128;
  const width = 780;
  for (const value of [0, 0.5, 1]) {
    const y = bottom - value * height;
    svg.append(make('line', { x1: left, x2: left + width, y1: y, y2: y, class: value === 0 ? 'astra-axis' : 'astra-grid' }));
    const label = make('text', { x: left - 6, y: y + 3, 'text-anchor': 'end', class: 'astra-tick' });
    label.textContent = value.toFixed(value === 0 ? 0 : 1);
    svg.append(label);
  }
  const step = width / tasks.length;
  const barWidth = step - 1.25;
  tasks.forEach((task, index) => {
    const x = left + index * step + 0.625;
    const group = make('g');
    const policyHeight = task.policyQ * height;
    group.append(make('rect', { x, y: bottom - policyHeight, width: barWidth, height: policyHeight, class: 'astra-policy-bar' }));
    let tooltip = `${task.name.replaceAll('_', ' ')} · learned policy Q ${task.policyQ.toFixed(3)}`;
    if (task.astra) {
      const astraQ = task.astra.q_sum / task.astra.episodes;
      const gain = Math.max(0, astraQ - task.policyQ);
      group.append(make('rect', { x, y: bottom - (policyHeight + gain * height), width: barWidth, height: gain * height, class: 'astra-gain-bar' }));
      tooltip += ` · Astra Q ${astraQ.toFixed(3)} (${task.astra.episodes} episodes)`;
    }
    const barTitle = make('title');
    barTitle.textContent = tooltip;
    group.append(barTitle);
    svg.append(group);
  });
  astraChart.replaceChildren(svg);
}).catch(() => {
  astraChart.textContent = 'Task comparison is unavailable.';
});
