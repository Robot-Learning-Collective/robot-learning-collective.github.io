const failureDescriptions = {
  'Dexterity': 'The robot cannot pick up or place objects accurately.',
  'Order': 'The robot follows the wrong action sequence. After skipping a step, it may not know how to continue.',
  'Confusion': 'Seemingly random behavior, probably after the robot enters a state outside its training data.',
  'Robot fell': 'The robot falls, often after stumbling over another object.',
  'Reasoning': 'A locally nonobvious action is needed, such as regrasping a knife by its handle.',
  'Robot stuck': 'The policy keeps trying to move while contact prevents the robot base from moving.',
  'Unknown': 'The failure could not be attributed to a specific problem.',
  'Head camera visibility': 'The head camera does not show the floor directly ahead or shelves above the robot’s head.',
  'Reachability': 'The robot cannot reach the target; after a failed grasp, this can make completion impossible.',
  'Navigation': 'The policy sometimes confuses tasks and travels to a location associated with another task.',
  'Whole-body coordination': 'The arms, torso, and base are poorly coordinated when squatting, reaching, or moving through a tight space.',
  'Search': 'Random exploration can repeatedly take the robot through the same area.',
};

const failureChart = document.getElementById('failure-chart');
const failurePopover = document.getElementById('failure-popover');
const failurePopoverDetail = document.getElementById('failure-popover-detail');
let activeFailureRow = null;
let failurePopoverPinned = false;

function positionFailurePopover() {
  if (!activeFailureRow) return;
  const rect = activeFailureRow.getBoundingClientRect();
  const width = failurePopover.offsetWidth;
  const height = failurePopover.offsetHeight;
  failurePopover.style.left = `${Math.max(12, Math.min(rect.left + 30, innerWidth - width - 12))}px`;
  const below = rect.bottom + 8;
  failurePopover.style.top = `${below + height <= innerHeight - 12 ? below : Math.max(12, rect.top - height - 8)}px`;
}

function hideFailurePopover() {
  activeFailureRow = null;
  failurePopoverPinned = false;
  failurePopover.hidden = true;
}

function showFailurePopover(row, reason, pinned = false) {
  activeFailureRow = row;
  failurePopoverPinned = pinned;
  failurePopoverDetail.textContent = failureDescriptions[reason];
  failurePopover.hidden = false;
  positionFailurePopover();
}

function formatFailureCount(value) {
  return Number(value.toFixed(2)).toString();
}

function addFailureSeries(plot, year, value) {
  const line = document.createElement('span');
  line.className = `failure-series year-${year}`;
  const bar = document.createElement('span');
  bar.className = 'failure-bar';
  bar.style.width = `${100 * value / 33}%`;
  line.append(bar);
  plot.append(line);
}

fetch('https://storage.googleapis.com/rlc-behavior-2026-public-exhibit/data/failure_reasons_shared_2025_2026.json')
  .then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(data => {
    const legend = document.createElement('div');
    legend.className = 'failure-chart-legend';
    legend.innerHTML = '<span><i class="year-2026"></i>2026 learned policy</span><span><i class="year-2025"></i>2025 submission</span>';
    failureChart.append(legend);

    const reasons = Object.entries(data.reason_weights)
      .sort((a, b) => Math.max(...Object.values(b[1])) - Math.max(...Object.values(a[1])) || a[0].localeCompare(b[0]));
    for (const [reason, values] of reasons) {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'failure-row';
      row.setAttribute('aria-label', `${reason}: 2026 ${formatFailureCount(values['2026'])}, 2025 ${formatFailureCount(values['2025'])}. ${failureDescriptions[reason]}`);
      const name = document.createElement('span');
      name.className = 'failure-reason';
      name.textContent = reason;
      const plot = document.createElement('span');
      plot.className = 'failure-plot';
      addFailureSeries(plot, '2026', values['2026']);
      addFailureSeries(plot, '2025', values['2025']);
      row.append(name, plot);
      row.addEventListener('pointerenter', () => {
        if (!failurePopoverPinned) showFailurePopover(row, reason);
      });
      row.addEventListener('pointerleave', () => {
        if (!failurePopoverPinned && document.activeElement !== row) hideFailurePopover();
      });
      row.addEventListener('focus', () => {
        if (!failurePopoverPinned) showFailurePopover(row, reason);
      });
      row.addEventListener('blur', () => {
        if (!failurePopoverPinned) hideFailurePopover();
      });
      row.addEventListener('click', () => {
        if (activeFailureRow === row && failurePopoverPinned) hideFailurePopover();
        else showFailurePopover(row, reason, true);
      });
      failureChart.append(row);
    }

  })
  .catch(() => {
    failureChart.textContent = 'Failure analysis chart is unavailable.';
  });

document.addEventListener('pointerdown', event => {
  if (activeFailureRow && !event.target.closest('.failure-row')) hideFailurePopover();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') hideFailurePopover();
});
window.addEventListener('resize', positionFailurePopover);
document.addEventListener('scroll', () => {
  if (activeFailureRow && !failurePopoverPinned) hideFailurePopover();
  else positionFailurePopover();
}, true);
