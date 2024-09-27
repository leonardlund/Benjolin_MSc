import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@5/+esm";
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// BASIC PROPERTIES
const BASE_COLOR = 0xffffff;
const BASE_SIZE = 0.1;
const BASE_OPACITY = 1;

// DATA
const x = new Float32Array(dataset3D['x']); //.slice(0, 100);
const y = new Float32Array(dataset3D['y']); //.slice(0, 100);
const z = new Float32Array(dataset3D['z']); //.slice(0, 100);
const N_POINTS = x.length;
const point_sizes = new Float32Array(x.length).fill(BASE_SIZE);
const point_colors = new Float32Array(x.length).fill(BASE_COLOR);
const point_opacities = new Float32Array(x.length).fill(BASE_OPACITY);
let particles;

let renderer, scene, camera, material, controls, stats;
let raycaster, intersects;
let pointer, INTERSECTED;

const PARTICLE_SIZE = 4;

init();

function init() {

    // SCENE
    scene = new THREE.Scene();

    // CAMERA
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 10000 );
    camera.position.z = 1000;

    // RENDERER
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.setAnimationLoop( animate );
    document.body.appendChild( renderer.domElement );

    // controls
    controls = new OrbitControls( camera, renderer.domElement );
    controls.listenToKeyEvents( window ); // optional
    //controls.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)
    controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 1;
    controls.maxDistance = 1000;
    controls.maxPolarAngle = Math.PI / 2;

    // RENDERING POINTS AS CIRCLES
	//const sprite = new THREE.TextureLoader().load( 'imgs/disc.png' );
	//sprite.colorSpace = THREE.SRGBColorSpace;

    const colors = [];
    const sizes = new Float32Array( N_POINTS );
    const color = new THREE.Color();
	// LOAD DATASETS
	const geometry = new THREE.BufferGeometry();
	const vertices = [];
	for ( let i = 0; i < N_POINTS; i ++ ) {
        let this_x = x[i] * 100 - 50;
        let this_y = y[i] * 100 - 50;
        let this_z = z[i] * 100 - 50;
		vertices.push( this_x, this_y, this_z);

        //const vx = Math.random();
        //const vy = Math.random();
        //const vz = Math.random();
        //color.setRGB( vx, vy, vz );
        color.setRGB( 255, 0, 0 );

        colors.push( color.r, color.g, color.b );
        sizes[i] = PARTICLE_SIZE;
	}
	geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );
    geometry.setAttribute( 'customColor', new THREE.Float32BufferAttribute( colors, 3 ) );
    geometry.setAttribute( 'size', new THREE.Float32BufferAttribute( sizes, 1 ) );

    //material = new THREE.PointsMaterial( { size: 0.05, vertexColors: true, map: sprite } );
    material = new THREE.ShaderMaterial( {

        uniforms: {
            color: { value: new THREE.Color( 0xffffff ) },
            pointTexture: { value: new THREE.TextureLoader().load( 'imgs/disc.png' ) },
            alphaTest: { value: 0.9 }
        },
        vertexShader: document.getElementById( 'vertexshader' ).textContent,
        fragmentShader: document.getElementById( 'fragmentshader' ).textContent,
        //blending: THREE.AdditiveBlending,
        depthTest: false,
        transparent: true
    } );

	// RENDER POINTS
	particles = new THREE.Points( geometry, material );
	scene.add( particles );

    // CLICK INTERACTION
    raycaster = new THREE.Raycaster();
    pointer = new THREE.Vector2();
    window.addEventListener( 'resize', onWindowResize );
    document.addEventListener( 'pointermove', onPointerMove );

    //
    stats = new Stats();
    document.body.appendChild( stats.dom );


}

function onPointerMove( event ) {
    pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}

function animate() {
    //controls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true
    render();
}

function render() {
    const time = Date.now() * 0.5;
    pickHelper.pick(pickPosition, scene, camera, time);
    if (isMouseDown){ // check for click and not drag
        setTimeout(function(){ if ( !isMouseDown && timer < 500){ pickHelper.click(pickPosition, scene, camera, time); }}, 50);
    }
    renderer.render( scene, camera );

}

let clickedIndices = [];

// HANDLE PICK AND CLICK EVENTS
let canvas = renderer.domElement;
class PickHelper {
    constructor() {
      this.raycaster = new THREE.Raycaster();
      this.pickedObject = null;
      this.pickedObjectIndex = null;
      this.pickedObjectSavedColor = 0;
      this.clickedObject = null;
      this.clickedObjectIndex = null;
    }
    pick(normalizedPosition, scene, camera, time) {
        // restore the color if there is a picked object
        if (this.pickedObject) {
            if ( !clickedIndices.includes(this.pickedObjectIndex) ) {
                particles.geometry.attributes.size.array[ this.pickedObjectIndex ] = PARTICLE_SIZE;
            }
            this.pickedObject = undefined;
            this.pickedObjectIndex = undefined;
        }
        // cast a ray through the frustum
        this.raycaster.setFromCamera(normalizedPosition, camera);
        // get the list of objects the ray intersected
        const intersectedObjects = this.raycaster.intersectObjects(scene.children);
        if (intersectedObjects.length) {
            // pick the first object. It's the closest one
            this.pickedObject = intersectedObjects[0].object;
            this.pickedObjectIndex = intersectedObjects[0].index;
            particles.geometry.attributes.size.array[ this.pickedObjectIndex ] = PARTICLE_SIZE * 20;
            particles.geometry.attributes.size.needsUpdate = true;
            console.log("picked ID: "+intersectedObjects[0].index);
        }
    }
    click(normalizedPosition, scene, camera, time) {
        // restore the color if there is a picked object
        if (this.clickedObject) {
            this.clickedObject = undefined;
            this.clickedObjectIndex = undefined;
        }
        // cast a ray through the frustum
        this.raycaster.setFromCamera(normalizedPosition, camera);
        // get the list of objects the ray intersected
        const intersectedObjects = this.raycaster.intersectObjects(scene.children);
        if (intersectedObjects.length) {
            // click the first object. It's the closest one
            this.clickedObject = intersectedObjects[0].object;
            this.clickedObjectIndex = intersectedObjects[0].index;
            clickedIndices.push(this.clickedObjectIndex);
            // update size
            particles.geometry.attributes.size.array[ this.clickedObjectIndex ] = PARTICLE_SIZE * 20;
            particles.geometry.attributes.size.needsUpdate = true;
            // update color
            let newcolor = new THREE.Color();
            newcolor.setRGB( 0, 255, 0 );    
            particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 ] = newcolor.r;
            particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 + 1 ] = newcolor.g;
            particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 + 2 ] = newcolor.b;
            particles.geometry.attributes.customColor.needsUpdate = true;
            material.needsUpdate = true
            console.log("clicked ID: "+intersectedObjects[0].index);
        }
    }
}

const pickPosition = {x: 0, y: 0};
clearPickPosition();

function getCanvasRelativePosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * canvas.width  / rect.width,
        y: (event.clientY - rect.top ) * canvas.height / rect.height,
    };
}

function setPickPosition(event) {
    const pos = getCanvasRelativePosition(event);
    pickPosition.x = (pos.x / canvas.width ) *  2 - 1;
    pickPosition.y = (pos.y / canvas.height) * -2 + 1;  // note we flip Y
}

function clearPickPosition() {
    // unlike the mouse which always has a position
    // if the user stops touching the screen we want
    // to stop picking. For now we just pick a value
    // unlikely to pick something
    pickPosition.x = -100000;
    pickPosition.y = -100000;
}

window.addEventListener('mousemove', setPickPosition);
window.addEventListener('mouseout', clearPickPosition);
window.addEventListener('mouseleave', clearPickPosition);

window.addEventListener('touchstart', (event) => {
    // prevent the window from scrolling
    event.preventDefault();
    setPickPosition(event.touches[0]);
  }, {passive: false});
   
window.addEventListener('touchmove', (event) => {
    setPickPosition(event.touches[0]);
});

window.addEventListener('touchend', clearPickPosition);

const pickHelper = new PickHelper();
let isMouseDown = false;
let timer = 0;
let startTime = 0;
let endTime = 0;

canvas.onmousedown = function(){
    isMouseDown = true;
    startTime = new Date().getTime();
    let timer = 0;
}; 
canvas.onmouseup = function(){
    isMouseDown = false;
    endTime = new Date().getTime();
    timer = endTime -startTime;
}; 