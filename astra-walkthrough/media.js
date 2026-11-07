'use strict';
// Preview uses the media already published with the original walkthrough.
// Set this to the team's public bucket prefix before consolidating hosting.
// The prefix must contain media/episode.mp4 and media/observations/*.jpg.
const MEDIA_BASE = 'https://keertimanney.github.io/behavior/';
function mediaURL(path) {
  return new URL(path, MEDIA_BASE).href;
}
const rolloutMedia = document.getElementById('rollout');
rolloutMedia.src = mediaURL(rolloutMedia.dataset.media);
