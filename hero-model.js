import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const host = document.querySelector('#hero-viewer');
const canvas = document.querySelector('#hero-model');
const status = document.querySelector('#motion-status');
const error = document.querySelector('#hero-model-error');
const panel = document.querySelector('#robot-panel');

if (host && canvas) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 10);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  scene.add(new THREE.HemisphereLight(0xf5f5f5, 0x777777, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 3.2);
  key.position.set(1.2, 1.8, 1.4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xe8e8e8, 1.25);
  fill.position.set(-1.3, 0.8, -1.1);
  scene.add(fill);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.enablePan = false;
  controls.minDistance = 0.25;
  controls.maxDistance = 1.8;
  controls.target.set(0, 0, 0);

  const resize = () => {
    const width = host.clientWidth;
    const height = host.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(host);
  resize();

  let robot;
  const labelPoints = {
    head: new THREE.Vector3(0, 0.145, 0),
    core: new THREE.Vector3(0, 0.015, 0),
    legs: new THREE.Vector3(0, -0.125, 0),
  };
  const labels = [...host.querySelectorAll('.hero-label')];

  function positionLabels() {
    if (!robot) return;
    for (const label of labels) {
      const point = labelPoints[label.dataset.anchor].clone();
      robot.localToWorld(point);
      point.project(camera);
      const x = (point.x * 0.5 + 0.5) * host.clientWidth;
      const y = (-point.y * 0.5 + 0.5) * host.clientHeight;
      label.style.left = `${x}px`;
      label.style.top = `${y}px`;
      label.classList.toggle('is-behind', point.z > 1 || Math.abs(point.x) > 1.15 || Math.abs(point.y) > 1.15);
    }
  }

  new GLTFLoader().load('assets/mini-humanoid.glb', (gltf) => {
    robot = gltf.scene;
    scene.add(robot);
    const bounds = new THREE.Box3().setFromObject(robot);
    const size = bounds.getSize(new THREE.Vector3());
    const center = bounds.getCenter(new THREE.Vector3());
    const span = Math.max(size.x, size.y, size.z);
    const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5))) * 1.65;
    controls.target.copy(center);
    camera.position.set(center.x + distance * 0.86, center.y + distance * 0.48, center.z + distance * 0.96);
    camera.near = Math.max(span / 100, 0.001);
    camera.far = span * 24;
    camera.updateProjectionMatrix();
    controls.minDistance = span * 0.65;
    controls.maxDistance = span * 7;
    panel.classList.add('model-loaded');
    status.textContent = 'MODEL READY';
    positionLabels();
  }, undefined, (loadError) => {
    console.error('Unable to load homepage robot model:', loadError);
    status.textContent = 'MODEL UNAVAILABLE';
    error.hidden = false;
  });

  function animate() {
    controls.update();
    positionLabels();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
}
