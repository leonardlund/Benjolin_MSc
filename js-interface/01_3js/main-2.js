import * as THREE from 'three';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@5/+esm";


const x = new Float32Array(dataset['x']); //.slice(0, 100);
const y = new Float32Array(dataset['y']); //.slice(0, 100);
let camera, scene, renderer, stats, material;

// create renderer
renderer = new THREE.WebGLRenderer();
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.setAnimationLoop( animate );
document.body.appendChild( renderer.domElement );

camera = new THREE.PerspectiveCamera( 55, window.innerWidth / window.innerHeight, 2, 2000 );
camera.position.z = 5;

scene = new THREE.Scene();
scene.background = new THREE.Color(0xefefef);
//scene.fog = new THREE.FogExp2( 0x000000, 0.001 );

const geometry = new THREE.BufferGeometry();
const vertices = [];

const sprite = new THREE.TextureLoader().load( 'imgs/disc.png' );
sprite.colorSpace = THREE.SRGBColorSpace;

for ( let i = 0; i < x.length; i ++ ) {
    vertices.push( x[i],y[i],0 );
}

geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( vertices, 3 ) );

material = new THREE.PointsMaterial( { size: 0.03, sizeAttenuation: true, map: sprite, alphaTest: 0.5, transparent: true } );
material.color.setHSL( 1.0, 0.3, 0.7, THREE.SRGBColorSpace );

const particles = new THREE.Points( geometry, material );
scene.add( particles );

// create a blue LineBasicMaterial
const line_material = new THREE.LineBasicMaterial( { color: 0x0000ff } );

// define points
const line_points = [];
let line_point1_index = 4
let line_point2_index = 6
line_points.push( new THREE.Vector3( x[line_point1_index], y[line_point1_index], 0 ) );
line_points.push( new THREE.Vector3( x[line_point2_index], y[line_point2_index], 0 ) );
const line_geometry = new THREE.BufferGeometry().setFromPoints( line_points );
const line = new THREE.Line( line_geometry, line_material );
scene.add( line );


//renderer.render( scene, camera );

function animate() {

	renderer.render( scene, camera );
	//stats.update();

}

function getClicked3DPoint(evt) {
	evt.preventDefault();

	mousePosition.x = ((evt.clientX - canvasPosition.left) / canvas.width) * 2 - 1;
	mousePosition.y = -((evt.clientY - canvasPosition.top) / canvas.height) * 2 + 1;

	rayCaster.setFromCamera(mousePosition, camera);
	var intersects = rayCaster.intersectObjects(scene.getObjectByName('MyObj_s').children, true);

	if (intersects.length > 0)
		return intersects[0].point;
};
