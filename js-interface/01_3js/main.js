// to run $ npx vite

import * as THREE from 'three';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@5/+esm";

const color_array = [
	"#1f78b4",
	"#b2df8a",
	"#33a02c",
	"#fb9a99",
	"#e31a1c",
	"#fdbf6f",
	"#ff7f00",
	"#6a3d9a",
	"#cab2d6",
	"#ffff99"
  ];


// GENERATE RANDOM POINTS
let radius = 25;
let point_num = 1000;
// Random point in circle code from https://stackoverflow.com/questions/32642399/simplest-way-to-plot-points-randomly-inside-a-circle
function randomPosition(radius) {
  var pt_angle = Math.random() * 2 * Math.PI;
  var pt_radius_sq = Math.random() * radius * radius;
  var pt_x = Math.sqrt(pt_radius_sq) * Math.cos(pt_angle);
  var pt_y = Math.sqrt(pt_radius_sq) * Math.sin(pt_angle);
  return [pt_x, pt_y];
}
let data_points = [];
for (let i = 0; i < point_num; i++) {
  let position = randomPosition(radius);
  let name = 'Point ' + i;
  let group = Math.floor(Math.random() * 6);
  let point = { position, name, group };
  data_points.push(point);
}
var generated_points = data_points;


// INSTANCIATE THREE JS ENVIRONMENT
const fov = 40,
	near = 10,
	far = 100;
var height = 350,
	viz_width = 700;
let aspect = viz_width / height;
const camera = new THREE.PerspectiveCamera(fov, aspect, near, far + 1);
const renderer = new THREE.WebGLRenderer({antialias: true});

let scene = new THREE.Scene();
scene.background = new THREE.Color(0xefefef);

// zoom
function toRadians (angle) {
	return angle * (Math.PI / 180);
}

function getScaleFromZ (camera_z_position) {
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

let d3_zoom = d3.zoom()
	.scaleExtent([getScaleFromZ(far), getScaleFromZ(near)])
	.on('zoom', () =>  {
		let d3_transform = d3.event.transform;
		zoomHandler(d3_transform);
	});

var zoom = d3_zoom;
const view = d3.select(renderer.domElement)

function setUpZoom() {
    view.call(zoom);    
    let initial_scale = getScaleFromZ(far);
    var initial_transform = d3.zoomIdentity.translate(viz_width/2, height/2).scale(initial_scale);    
    zoom.transform(view, initial_transform);
    camera.position.set(0, 0, far);
}
setUpZoom();


var circle_sprite = new THREE.TextureLoader().load(
	"https://blog.fastforwardlabs.com/images/2018/02/circle-1518727951930.png"
)
var circle_sprite_aa = new THREE.TextureLoader().load(
	"https://blog.fastforwardlabs.com/images/2018/02/circle_aa-1518730700478.png"
)
  
// Non anti-aliased settings
let circle = {
	map: circle_sprite,
	transparent: true
}

// Anti-aliased settings
let circle_aa = {
	map: circle_sprite_aa,
	transparent: true,
	alphaTest: 0.5
}

let sprite_settings = circle;
let pointsGeometry = new THREE.BufferGeometry();

let colors = [];
const vertices = new Float32Array(point_num*3);
//for (let datum of generated_points) {
for (let i = 0; i < point_num; i+=3) {
	let datum = generated_points[i];
	// Set vector coordinates from data
  //let vertex = new THREE.Vector3(datum.position[0], datum.position[1], 0);
  vertices[i] = datum.position[0];
  vertices[i+1] = datum.position[1];
  vertices[i+2] = 0;
  //pointsGeometry.vertices.push(vertex);
  let color = new THREE.Color(color_array[datum.group]);
  colors.push(color);
}
pointsGeometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );
pointsGeometry.colors = colors;

let pointsMaterial = new THREE.PointsMaterial({
  size: 8,
  sizeAttenuation: false,
  vertexColors: THREE.VertexColors,
});

// Add settings from sprite_settings
for (let setting in sprite_settings) {
  pointsMaterial[setting] = sprite_settings[setting];
}

var points = new THREE.Points(pointsGeometry, pointsMaterial);

scene.add(points);
renderer.render( scene, camera );


// create renderer
/*const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

// create camera
const camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 500 );
camera.position.set( 0, 0, 100 );
camera.lookAt( 0, 0, 0 );

// create scene
const scene = new THREE.Scene();

//create a blue LineBasicMaterial
const material = new THREE.LineBasicMaterial( { color: 0x0000ff } );

// define points
const points = [];
points.push( new THREE.Vector3( - 10, 0, 0 ) );
points.push( new THREE.Vector3( 0, 10, 0 ) );
points.push( new THREE.Vector3( 10, 0, 0 ) );
const geometry = new THREE.BufferGeometry().setFromPoints( points );
const line = new THREE.Line( geometry, material );

scene.add( line );
renderer.render( scene, camera );


const dotGeometry = new THREE.BufferGeometry();
dotGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0,0,0]), 3));
const dotMaterial = new THREE.PointsMaterial({ size: 5, color: 0xff0000 });
const dot = new THREE.Points(dotGeometry, dotMaterial);
scene.add(dot);

renderer.render( scene, camera );*/


/*const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.setAnimationLoop( animate );
document.body.appendChild( renderer.domElement );

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const cube = new THREE.Mesh( geometry, material );
scene.add( cube );

camera.position.z = 5;

function animate() {

	cube.rotation.x += 0.01;
	cube.rotation.y += 0.01;

	renderer.render( scene, camera );

};*/