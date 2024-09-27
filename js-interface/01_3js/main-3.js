import * as THREE from 'three';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@5/+esm";
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// DATA
const x = new Float32Array(dataset['x']); //.slice(0, 100);
const y = new Float32Array(dataset['y']); //.slice(0, 100);

// THREE.JS CONSTANTS
let camera, scene, renderer, stats, material;
let mouseX = 0, mouseY = 0;

let windowHalfX = window.innerWidth / 2;
let windowHalfY = window.innerHeight / 2;

// VARS FOR ZOOM FUNCTION
let fov = 10,
	near = 1,
	far = 20;
let viz_width = window.innerWidth / 2;
let height = window.innerHeight / 2;


// INITIALIZE SCENE
init();

// INITIALIZATION FUNCTION
function init() {

	// INITIALIZE CAMERA
	camera = new THREE.PerspectiveCamera( fov, window.innerWidth / window.innerHeight, near, far+1 );
	camera.position.z = 5;

	// INITIALIZE SCENE
	scene = new THREE.Scene();
	scene.background = new THREE.Color(0xefefef);
	//scene.fog = new THREE.FogExp2( 0x000000, 0.001 );

	// RENDERING POINTS AS CIRCLES
	const sprite = new THREE.TextureLoader().load( 'imgs/disc.png' );
	sprite.colorSpace = THREE.SRGBColorSpace;

	// LOAD DATASETS
	const geometry = new THREE.BufferGeometry();
	const vertices = [];
	for ( let i = 0; i < x.length; i ++ ) {
		vertices.push( x[i],y[i], 0);
	}
	geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );

	// POINTS MATERIAL
	material = new THREE.PointsMaterial( { size: 0.2, sizeAttenuation: true, map: sprite, alphaTest: 0.5, transparent: true } );
	material.color.setHSL( 1.0, 0.3, 0.7, THREE.SRGBColorSpace );

	// RENDER POINTS
	const particles = new THREE.Points( geometry, material );
	scene.add( particles );

	// RENDER A LINE
	/*const line_material = new THREE.LineBasicMaterial( { color: 0x0000ff } );
	const line_points = [];
	let line_point1_index = 4
	let line_point2_index = 6
	line_points.push( new THREE.Vector3( x[line_point1_index], y[line_point1_index], 0 ) );
	line_points.push( new THREE.Vector3( x[line_point2_index], y[line_point2_index], 0 ) );
	const line_geometry = new THREE.BufferGeometry().setFromPoints( line_points );
	const line = new THREE.Line( line_geometry, line_material );
	scene.add( line );*/


	//

	renderer = new THREE.WebGLRenderer();
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.setAnimationLoop( animate );
	document.body.appendChild( renderer.domElement );

	//let controls = new OrbitControls( camera, renderer.domElement );
	//controls.listenToKeyEvents( window );

	//

	//stats = new Stats();
	//document.body.appendChild( stats.dom );

	//

	//const gui = new GUI();
	//gui.add( material, 'sizeAttenuation' ).onChange( function () { material.needsUpdate = true; } );
	//gui.open();

	//

	//document.body.style.touchAction = 'none';
	//document.body.addEventListener( 'pointermove', onDocumentMouseMove );

	//

	//window.addEventListener( 'resize', onWindowResize );

}

function onWindowResize() {

	windowHalfX = window.innerWidth / 2;
	windowHalfY = window.innerHeight / 2;

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();

	renderer.setSize( window.innerWidth, window.innerHeight );

}

function onPointerMove( event ) {

	if ( event.isPrimary === false ) return;

	mouseX = event.clientX - windowHalfX;
	mouseY = event.clientY - windowHalfY;

}

//

function animate() {

	render();
	//stats.update();

}

function render() {

	//const time = Date.now() * 0.00005;

	//camera.position.x += ( mouseX - camera.position.x ) * 0.05;
	//camera.position.y += ( - mouseY - camera.position.y ) * 0.05;

	//camera.lookAt( scene.position );

	//const h = ( 360 * ( 1.0 + time ) % 360 ) / 360;
	//material.color.setHSL( h, 0.5, 0.5 );

	renderer.render( scene, camera );

}

// CAMERA ZOOM FUNCTIONS (from: https://observablehq.com/@grantcuster/using-three-js-for-2d-data-visualization)
function getScaleFromZ (camera_z_position) {
	// get d3 scale from three.js zoom
	let half_fov = fov/2;
	let half_fov_radians = toRadians(half_fov);
	let half_fov_height = Math.tan(half_fov_radians) * camera_z_position; 
	let fov_height = half_fov_height * 2;
	let scale = height / fov_height; // Divide visualization height by height derived from field of view
	return scale;
}
function getZFromScale(scale) {
	let half_fov = fov/2;
	let half_fov_radians = toRadians(half_fov);
	let scale_height = height / scale;
	let camera_z_position = scale_height / (2 * Math.tan(half_fov_radians));
	return camera_z_position;
}
function zoomHandler(d3_transform) {
	let scale = d3_transform.k;
	let x = -(d3_transform.x - viz_width/2) / scale;
	let y = (d3_transform.y - height/2) / scale;
	let z = getZFromScale(scale);
	camera.position.set(x, y, z);
}

function toRadians (angle) {
  return angle * (Math.PI / 180);
}


let zoom = d3.zoom()
	.scaleExtent([getScaleFromZ(far), getScaleFromZ(near)])
	.on('zoom', () =>  {
	let d3_transform = d3.event.transform;
	zoomHandler(d3_transform);
	});

let view = d3.select(renderer.domElement)
function setUpZoom() {
    view.call(zoom);    
    let initial_scale = getScaleFromZ(far);
    var initial_transform = d3.zoomIdentity.translate(viz_width/2, height/2).scale(initial_scale);    
    zoom.transform(view, initial_transform);
    camera.position.set(0, 0, far);
}
setUpZoom();

