const cookComparison = document.getElementById('cook-comparison');
const cookPolicy = document.getElementById('cook-policy-video');
const cookAstra = document.getElementById('cook-astra-video');
const cookTimeline = document.getElementById('cook-timeline');
const cookTime = document.getElementById('cook-time');
const cookCue = document.getElementById('cook-cue');
const cookSteps = document.querySelectorAll('#cook-steps span');
const cookPlay = document.getElementById('cook-play');
const cookOverlays = document.querySelectorAll('#cook-comparison .cook-overlay');

if (cookComparison && cookPolicy && cookAstra && cookTimeline && cookTime && cookCue) {
  const videos = [cookPolicy, cookAstra];
  const speed = 8;
  let scrubbing = false;
  let resumeAfterScrub = false;
  const cues = [
    [0, 'The VLA reaches the fridge and opens it faster than Astra.'],
    [53, 'The VLA carries cabbage and chili in both hands, saving time.'],
    [126, 'The VLA makes a critical mistake: it picks up the chili instead of the knife (behind the plate, possibly hard to see at this resolution and partly blocked by the right arm), which came next in every training demonstration.'],
    [164, 'The VLA struggles to put the chili on the cutting board. In the demonstrations, the cabbage is already diced by this point.'],
    [173, 'It eventually places the chili, then seems unsure what to do with both vegetables on the board.'],
    [262, 'Astra makes a weird arm motion to line up the knife grasp, but succeeds.'],
    [343, 'Why the VLA remains stuck here is unclear.'],
    [413, 'The VLA even demonstrates emergent behavior by recovering and picking up the chili from the floor.'],
    [435, 'Astra struggles to pour the vegetables smoothly into the frying pan, but enough pieces land inside.'],
    [475, 'The VLA is still stuck, perhaps reaching for the knife.'],
    [517, 'The VLA finally tries something else.'],
    [521, 'It misses the knife again, then spends the rest of the episode trying to open the fridge. Near the end, it seemingly gives up and stops moving.'],
  ];

  const duration = video => Number.isFinite(video.duration) ? video.duration : 0;
  const formatTime = seconds => {
    const rounded = Math.floor(seconds || 0);
    return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`;
  };
  const updateCue = () => {
    let index = 0;
    for (let next = 1; next < cues.length && cookPolicy.currentTime >= cues[next][0]; next++) index = next;
    if (cookCue.textContent !== cues[index][1]) cookCue.textContent = cues[index][1];
    cookSteps.forEach((step, stepIndex) => step.classList.toggle('is-active', stepIndex === index));
  };
  const updateTransport = () => {
    const total = duration(cookPolicy);
    if (total) cookTimeline.max = String(total);
    if (!scrubbing) cookTimeline.value = String(cookPolicy.currentTime);
    cookTime.textContent = `${formatTime(cookPolicy.currentTime)} / ${formatTime(total)}`;
    const action = cookPolicy.ended ? 'Replay' : cookPolicy.paused ? 'Play' : 'Pause';
    cookPolicy.setAttribute('aria-label', `${action} both videos from the VLA view`);
    if (cookPlay) {
      cookPlay.textContent = cookPolicy.paused ? (cookPolicy.ended ? '↺' : '▶') : '❚❚';
      cookPlay.setAttribute('aria-label', `${action} both videos`);
    }
    cookComparison.classList.toggle('is-playing', !cookPolicy.paused);
    cookAstra.setAttribute('aria-label', `${action} both videos from the Astra view`);
    updateCue();
  };
  const seekBoth = seconds => {
    for (const video of videos) {
      const end = duration(video);
      if (end) video.currentTime = Math.min(seconds, Math.max(0, end - 0.04));
    }
    updateTransport();
  };
  const pauseBoth = () => {
    videos.forEach(video => video.pause());
    updateTransport();
  };
  const playBoth = () => {
    if (cookPolicy.ended || cookPolicy.currentTime >= duration(cookPolicy) - 0.1) seekBoth(0);
    for (const video of videos) {
      video.playbackRate = speed;
      if (duration(video) && video.currentTime < duration(video) - 0.1) video.play().catch(updateTransport);
    }
    updateTransport();
  };

  const togglePlayback = () => cookPolicy.paused ? playBoth() : pauseBoth();
  cookPlay?.addEventListener('click', togglePlayback);
  cookOverlays.forEach(overlay => overlay.addEventListener('click', togglePlayback));
  for (const video of videos) {
    video.defaultPlaybackRate = speed;
    video.playbackRate = speed;
    video.addEventListener('loadedmetadata', updateTransport);
    video.addEventListener('click', togglePlayback);
    video.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      togglePlayback();
    });
  }
  cookPolicy.addEventListener('timeupdate', () => {
    if (!scrubbing && duration(cookAstra) && cookPolicy.currentTime < duration(cookAstra) - 0.1) {
      if (Math.abs(cookAstra.currentTime - cookPolicy.currentTime) > 0.75) cookAstra.currentTime = cookPolicy.currentTime;
      if (!cookPolicy.paused && cookAstra.paused) cookAstra.play().catch(() => {});
    }
    updateTransport();
  });
  cookPolicy.addEventListener('play', updateTransport);
  cookPolicy.addEventListener('pause', updateTransport);
  cookPolicy.addEventListener('ended', () => { cookAstra.pause(); updateTransport(); });
  cookTimeline.addEventListener('pointerdown', () => {
    scrubbing = true;
    resumeAfterScrub = !cookPolicy.paused;
    pauseBoth();
  });
  cookTimeline.addEventListener('input', () => {
    seekBoth(Number(cookTimeline.value));
    cookTime.textContent = `${formatTime(Number(cookTimeline.value))} / ${formatTime(duration(cookPolicy))}`;
  });
  const finishScrub = () => {
    if (!scrubbing) return;
    scrubbing = false;
    if (resumeAfterScrub) playBoth();
    else updateTransport();
  };
  cookTimeline.addEventListener('pointerup', finishScrub);
  cookTimeline.addEventListener('pointercancel', finishScrub);
  cookTimeline.addEventListener('change', finishScrub);
}
