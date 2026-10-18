import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.querySelector('#robot-viewer');
const status = document.querySelector('#viewer-status');
const error = document.querySelector('#viewer-error');
const resetButton = document.querySelector('#reset-view');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#e5e5e5');

const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 20);
camera.position.set(0.55, 0.38, 0.55);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

scene.add(new THREE.HemisphereLight(0xf5f5f5, 0x777777, 2.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.0);
keyLight.position.set(1.2, 1.8, 1.4);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xe8e8e8, 1.2);
fillLight.position.set(-1.3, 0.8, -1.1);
scene.add(fillLight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.22;
controls.maxDistance = 2.2;
controls.target.set(0, 0, 0);

const floor = new THREE.GridHelper(1.2, 24, 0x888888, 0xcccccc);
floor.position.y = -0.18;
scene.add(floor);

let initialPosition;
let initialTarget;

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(canvas);
resize();

const loader = new GLTFLoader();
loader.load('assets/mini-humanoid.glb', (gltf) => {
  const robot = gltf.scene;
  scene.add(robot);
  const bounds = new THREE.Box3().setFromObject(robot);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const maxSize = Math.max(size.x, size.y, size.z);
  const fitDistance = maxSize / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5))) * 1.6;
  controls.target.copy(center);
  camera.position.set(center.x + fitDistance * 0.9, center.y + fitDistance * 0.55, center.z + fitDistance * 0.95);
  camera.near = Math.max(maxSize / 100, 0.001);
  camera.far = maxSize * 30;
  camera.updateProjectionMatrix();
  controls.minDistance = maxSize * 0.6;
  controls.maxDistance = maxSize * 8;
  initialPosition = camera.position.clone();
  initialTarget = center.clone();
  status.textContent = 'MODEL READY';
  document.querySelector('.viewer-meta i').classList.add('ready');
}, undefined, (loadError) => {
  console.error('Unable to load MiniHumanoid GLB:', loadError);
  status.textContent = 'MODEL UNAVAILABLE';
  error.hidden = false;
});

resetButton.addEventListener('click', () => {
  if (!initialPosition || !initialTarget) return;
  camera.position.copy(initialPosition);
  controls.target.copy(initialTarget);
  controls.update();
});

function render() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
render();
