const emergentClips = [
  {
    video: document.getElementById('beetroot-clip'),
    speed: 4,
    caption: document.getElementById('beetroot-cue'),
    steps: document.querySelectorAll('#beetroot-steps span'),
    cues: [
      [0, 'Struggles to pick up a beetroot.'],
      [14, 'Drops the beetroot.'],
      [35, 'Confidently picks up the beetroot from the floor, a recovery not shown in this task’s demonstrations.'],
      [52, 'Proceeds with the task successfully.'],
    ],
  },
  {
    video: document.getElementById('bin-clip'),
    speed: 2,
    caption: document.getElementById('bin-cue'),
    steps: document.querySelectorAll('#bin-steps span'),
    cues: [
      [0, 'Places the bin prematurely, then approaches with the can.'],
      [3, 'Places the can directly into the bin, a skill absent from this task’s demonstrations.'],
    ],
  },
];

for (const clip of emergentClips) {
  const updateCaption = () => {
    let cueIndex = 0;
    for (let index = 1; index < clip.cues.length; index++) {
      if (clip.video.currentTime < clip.cues[index][0]) break;
      cueIndex = index;
    }
    const caption = clip.cues[cueIndex][1];
    if (clip.caption.textContent !== caption) clip.caption.textContent = caption;
    clip.steps.forEach((step, index) => step.classList.toggle('is-active', index === cueIndex));
  };
  clip.video.defaultPlaybackRate = clip.speed;
  clip.video.playbackRate = clip.speed;
  clip.video.addEventListener('timeupdate', updateCaption);
  clip.video.addEventListener('seeked', updateCaption);
  clip.video.addEventListener('loadedmetadata', () => {
    clip.video.playbackRate = clip.speed;
    updateCaption();
  });
  clip.video.addEventListener('play', () => {
    clip.video.playbackRate = clip.speed;
    for (const other of emergentClips) {
      if (other !== clip) other.video.pause();
    }
  });
}
