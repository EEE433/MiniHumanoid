const previewButton = document.querySelector('#motion-button');
const buttonLabel = document.querySelector('#button-label');
const statusLabel = document.querySelector('#motion-status');
const robotPanel = document.querySelector('#robot-panel');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let previewTimer;

previewButton?.addEventListener('click', () => {
  const isPlaying = previewButton.getAttribute('aria-pressed') === 'true';
  window.clearTimeout(previewTimer);

  if (isPlaying) {
    previewButton.setAttribute('aria-pressed', 'false');
    buttonLabel.textContent = 'PREVIEW A MOTION';
    statusLabel.textContent = 'IDLE';
    robotPanel.classList.remove('is-moving');
    return;
  }

  previewButton.setAttribute('aria-pressed', 'true');
  buttonLabel.textContent = 'STOP PREVIEW';
  statusLabel.textContent = 'PREVIEW';
  robotPanel.classList.add('is-moving');

  if (!reducedMotion.matches) {
    previewTimer = window.setTimeout(() => {
      previewButton.setAttribute('aria-pressed', 'false');
      buttonLabel.textContent = 'PREVIEW A MOTION';
      statusLabel.textContent = 'IDLE';
      robotPanel.classList.remove('is-moving');
    }, 2400);
  }
});
