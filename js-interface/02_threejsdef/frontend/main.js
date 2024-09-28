import * as THREE from 'three';
import Stats from 'three/addons/libs/stats.module.js';
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@5/+esm";
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';


/* 
todo:
- implement exploration/composition MODES button
    - exploration highlights points in the scatterplot without adding a box (when a new point is highlighted the old one is de-highlighted)
    - composition mode allows to add boxes. When swithcing from composition to exploration the boxes already created are left where they are
- play button colored while playing
- establishing times
- render arrows instead of lines
- highlighting arrows
- time limit or infinite composition bar?
- OSC integration
- play and stop buttons
- completely responsive webpage
- touchscreen?

logic: 
- you can't click on point if it's the last one being clicked
- you can't create any more elements if limit has been reached
- if it's playing, clicking anywhere stops the playing OR disable all functions until playing stops

bugs: 
- what's the problem with meander code?
- what does the stop button do?
- why do I need to send two messages for them to work? -- probably a python thing
- render function has problems
*/



// BASIC PROPERTIES
const BASE_COLOR = 0xffffff;
const BASE_SIZE = 0.1;
const BASE_OPACITY = 1;

// DATA
const x = new Float32Array(dataset3D['x']); //.slice(0, 100);
const y = new Float32Array(dataset3D['y']); //.slice(0, 100);
const z = new Float32Array(dataset3D['z']); //.slice(0, 100);
const N_POINTS = x.length;
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
    camera = new THREE.PerspectiveCamera( 45, (window.innerWidth/2) / window.innerHeight , 1, 10000 );
    camera.position.z = 1000;

    // RENDERER
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth/2, window.innerHeight* 0.85 );
    renderer.setAnimationLoop( animate );
    document.getElementById( 'scatterPlot' ).appendChild( renderer.domElement );

    // ORBIT CONTROLS
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

    // LOAD DATASETS
    const colors = [];
    const sizes = new Float32Array( N_POINTS );
    const color = new THREE.Color();
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
    // GEOMETRY MATERIAL
    material = new THREE.ShaderMaterial( {
        uniforms: {
            color: { value: new THREE.Color( 0xffffff ) },
            pointTexture: { value: new THREE.TextureLoader().load( 'imgs/disc.png' ) },
            alphaTest: { value: 0.9 }
        },
        vertexShader: document.getElementById( 'vertexshader' ).textContent,
        fragmentShader: document.getElementById( 'fragmentshader' ).textContent,
        blending: THREE.AdditiveBlending,
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

    // STATS
    stats = new Stats();
    document.getElementById( 'scatterPlot' ).appendChild( stats.dom );

}

// UPDATE POINTER POSITION
function onPointerMove( event ) {
    pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
}
// UPDATE WINDOW SIZE
function onWindowResize() {
    camera.aspect = (window.innerWidth /2) / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth / 2, window.innerHeight * 0.85 );
}
// ANIMATE FOR CAMERA NAVIGATION
function animate() {
    //controls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true
    render();
}
// RENDER FUNCTION FOR ANIMATION
function render() {
    const time = Date.now() * 0.5;
    pickHelper.pick(pickPosition, scene, camera, time);
    if (isMouseDown){ // check for click and not drag
        setTimeout(function(){ if ( !isMouseDown && timer < 500){ pickHelper.click(pickPosition, scene, camera, time); }}, 5);
    }

    // UPDATE FOR THIS TO RUN ONLY ONCE (NOT RENDER ANY NEW POINTS DURING ANIMATION LOOP)
    /*for ( let i = 0; i < clickedIndices.length; i ++ ) {
        if ( i != 0 ){
            let linepoints = []; 
            linepoints.push( new THREE.Vector3( x[clickedIndices[i-1]] * 100 - 50, 
                                                y[clickedIndices[i-1]] * 100 - 50, 
                                                z[clickedIndices[i-1]]  * 100 - 50) ); 
            linepoints.push( new THREE.Vector3( x[clickedIndices[i]] * 100 - 50, 
                                                y[clickedIndices[i]] * 100 - 50, 
                                                z[clickedIndices[i]]  * 100 - 50) ); 
            const linegeometry = new THREE.BufferGeometry().setFromPoints( linepoints );
            const linematerial = new THREE.LineBasicMaterial( { color: 0x0000ff } );
            const line = new THREE.Line( linegeometry, linematerial );
            scene.add( line ); 
        }
    }*/

    renderer.render( scene, camera );
}

// HANDLE PICK AND CLICK EVENTS
let clickedIndices = [];
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
            sendBox(x[this.pickedObjectIndex], y[this.pickedObjectIndex]);
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
            if (intersectedObjects[0].index != this.clickedObjectIndex){
                console.log(intersectedObjects[0].index)
                // click the first object. It's the closest one
                this.clickedObject = intersectedObjects[0].object;
                this.clickedObjectIndex = intersectedObjects[0].index;
                clickedIndices.push(this.clickedObjectIndex);
                // update size
                particles.geometry.attributes.size.array[ this.clickedObjectIndex ] = PARTICLE_SIZE * 20;
                particles.geometry.attributes.size.needsUpdate = true;
                // update color
                let newcolor = new THREE.Color();
                newcolor.setRGB( Math.random(), Math.random(), Math.random() );
                particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 ] = newcolor.r;
                particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 + 1 ] = newcolor.g;
                particles.geometry.attributes.customColor.array[ this.clickedObjectIndex * 3 + 2 ] = newcolor.b;
                particles.geometry.attributes.customColor.needsUpdate = true;
                material.needsUpdate = true
                console.log("clicked ID: "+intersectedObjects[0].index);

                addBox(x[ this.clickedObjectIndex ], y[ this.clickedObjectIndex ], z[ this.clickedObjectIndex ], 
                    newcolor.getHexString(), this.clickedObjectIndex); 
                renderPath();
                //console.log(pts);        
            }
        }
    }
}

// RENDERING FUNCTIONS
function pointToBasic(pointIndex){
    // restore point rendering to the basic properties
    // update size
    particles.geometry.attributes.size.array[ pointIndex ] = PARTICLE_SIZE;
    particles.geometry.attributes.size.needsUpdate = true;
    // update color
    let newcolor = new THREE.Color();
    newcolor.setRGB( 255, 0, 0 );
    particles.geometry.attributes.customColor.array[ pointIndex * 3 ] = newcolor.r;
    particles.geometry.attributes.customColor.array[ pointIndex * 3 + 1 ] = newcolor.g;
    particles.geometry.attributes.customColor.array[ pointIndex * 3 + 2 ] = newcolor.b;
    particles.geometry.attributes.customColor.needsUpdate = true;
    material.needsUpdate = true
}

let arrowsIDs = []
// RENDER PATH
function renderPath(){
    // remove all previous arrows
    for (let i = 0; i < arrowsIDs.length; i++) {
        var selectedObject = scene.getObjectByName(arrowsIDs[i]);
        scene.remove( selectedObject );
    }
    // draw new arrows
    for (let i = 0; i < compositionArray.length; i++) {
        if (i !=0){ // check if not on the first object
            if (compositionArray[i] instanceof Meander){
                // check if before and after there are boxes
                if (compositionArray[i-1] instanceof Box && compositionArray[i+1] instanceof Box){ 
                    // add dotted line if meander
                    //x: [compositionArray[i-1].x,compositionArray[i+1].x],
                    //y: [compositionArray[i-1].y,compositionArray[i+1].y],
                }
            }
            if (compositionArray[i] instanceof Crossfade){
                // check if before and after there are boxes
                if (compositionArray[i-1] instanceof Box && compositionArray[i+1] instanceof Box){ 
                    // add dashed line if crossfade
                    //x: [compositionArray[i-1].x,compositionArray[i+1].x],
                    //y: [compositionArray[i-1].y,compositionArray[i+1].y],
                }
            }
            if (compositionArray[i] instanceof Box){
                // check if before there is a box
                if (compositionArray[i-1] instanceof Box){ 
                    // add straight line if jump
                    //x: [compositionArray[i-1].x,compositionArray[i+1].x],
                    //y: [compositionArray[i-1].y,compositionArray[i+1].y],
                    let linepoints = []; 
                    linepoints.push( new THREE.Vector3( compositionArray[i-1].x * 100 - 50, 
                                                        compositionArray[i-1].y * 100 - 50, 
                                                        compositionArray[i-1].z  * 100 - 50) ); 
                    linepoints.push( new THREE.Vector3( compositionArray[i].x * 100 - 50, 
                                                        compositionArray[i].y * 100 - 50, 
                                                        compositionArray[i].z  * 100 - 50) ); 
                    let linegeometry = new THREE.BufferGeometry().setFromPoints( linepoints );
                    let linematerial = new THREE.LineBasicMaterial( { color: 0x0000ff } );
                    let line = new THREE.Line( linegeometry, linematerial );
                    let line_name = "jump"+compositionArray[i-1].index+' '+compositionArray[i].index;
                    line.name = line_name;
                    arrowsIDs.push(line_name);
                    scene.add( line );
                }
            }
        }
    } 
}

const pickPosition = {x: 0, y: 0}; // pick position in 2D space
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

// COMPOSITION FUNCTIONALITIES
class Box{
    constructor(x, y, z, duration, arrayIndex){
        this.x = x;
        this.y = y;
        this.z = z;
        this.duration = duration;
        this.arrayIndex = arrayIndex; // index in the array
    }
}
class Meander{
    constructor(duration){
        this.duration = duration;
    }
}
class Crossfade{
    constructor(duration){
        this.duration = duration;
    }
}
const compositionArray = [];

// add a box when a point on the scatterplot is clicked
var numBoxes = 0
function addBox(boxx, boxy, boxz, randomcolor, arrayIndex) {
    let newBox = document.createElement("div");
    newBox.className = 'box';
    newBox.id = 'box '+numBoxes;
    newBox.style["background-color"] = '#'+randomcolor;
    newBox.style["height"] = '5vh';
    newBox.style["width"] = '60%';
    newBox.style["resize"] = 'vertical';
    newBox.style["overflow-x"] = 'auto';
    newBox.draggable = 'true';
    newBox.addEventListener('dragstart', dragStart);
    newBox.addEventListener('dragenter', dragEnter)
    newBox.addEventListener('dragover', dragOver);
    newBox.addEventListener('dragleave', dragLeave);
    newBox.addEventListener('drop', drop);
    document.getElementById("compose-bar").appendChild(newBox); 
    console.log(newBox.id);
    numBoxes += 1;
    var duration = 2;
    compositionArray.push(new Box(boxx, boxy, boxz, duration, arrayIndex));
    //console.log(compositionArray);
    observer.observe(newBox);
}

// prevent images inside boxes from being draggable
function dragImgInsideBox(e) {
    e.preventDefault();
    e.stopPropagation(); 
}

// add new crossfade
function addCrossfade(){

    const newImg = document.createElement("img");
    newImg.src = "imgs/arrow.png";
    newImg.id = "img-inside-box";
    newImg.className = "img-fluid img-inside-box";
    newImg.style["height"] = '80%';

    const newCrossfade = document.createElement("div");
    newCrossfade.className = 'crossfade text-center';
    newCrossfade.id = 'box ' + numBoxes;
    newCrossfade.style["background-color"] = 'DodgerBlue';
    newCrossfade.style["height"] = '5vh';
    newCrossfade.style["width"] = '60%';
    newCrossfade.style["resize"] = 'vertical';
    newCrossfade.style["overflow-x"] = 'auto';
    newCrossfade.draggable = 'true';

    newCrossfade.appendChild(newImg)
    document.getElementById("compose-bar").appendChild(newCrossfade); 

    newCrossfade.addEventListener('dragstart', dragStart);
    newCrossfade.addEventListener('dragenter', dragEnter)
    newCrossfade.addEventListener('dragover', dragOver);
    newCrossfade.addEventListener('dragleave', dragLeave);
    newCrossfade.addEventListener('drop', drop);

    newImg.addEventListener('dragstart', dragImgInsideBox);
    newImg.addEventListener('dragenter', dragImgInsideBox)
    newImg.addEventListener('dragover', dragImgInsideBox);
    newImg.addEventListener('dragleave', dragImgInsideBox);
    newImg.addEventListener('drop', dragImgInsideBox);

    console.log(newCrossfade.id);
    numBoxes += 1;

    var duration = 2;
    compositionArray.push(new Crossfade(duration));
    console.log(compositionArray);
    observer.observe(newCrossfade);
    // update visualization

}

// add new meander
function addMeander(){

    const newImg = document.createElement("img");
    newImg.src = "imgs/zigzag.png";
    newImg.id = "img-inside-box";
    newImg.className = "img-fluid img-inside-box";
    newImg.style["height"] = '80%';

    const newMeander = document.createElement("div");
    newMeander.className = 'meander text-center';
    newMeander.id = 'box ' + numBoxes;
    newMeander.style["background-color"] = 'DodgerBlue';
    newMeander.style["height"] = '5vh';
    newMeander.style["width"] = '60%';
    newMeander.style["resize"] = 'vertical';
    newMeander.style["overflow-x"] = 'auto';
    newMeander.draggable = 'true';
    
    newMeander.appendChild(newImg)
    document.getElementById("compose-bar").appendChild(newMeander); 

    newMeander.addEventListener('dragstart', dragStart);
    newMeander.addEventListener('dragenter', dragEnter)
    newMeander.addEventListener('dragover', dragOver);
    newMeander.addEventListener('dragleave', dragLeave);
    newMeander.addEventListener('drop', drop);

    newImg.addEventListener('dragstart', dragImgInsideBox);
    newImg.addEventListener('dragenter', dragImgInsideBox)
    newImg.addEventListener('dragover', dragImgInsideBox);
    newImg.addEventListener('dragleave', dragImgInsideBox);
    newImg.addEventListener('drop', dragImgInsideBox);

    console.log(newMeander.id);
    numBoxes += 1;

    var duration = 2;
    compositionArray.push(new Meander(duration));
    console.log(compositionArray);
    observer.observe(newMeander);
    // update visualization
}

// handling navbar elements
const insert_crossfade = document.getElementById("insert-crossfade"); 
insert_crossfade.addEventListener("click", addCrossfade); 
const insert_meander = document.getElementById("insert-meander"); 
insert_meander.addEventListener("click", addMeander); 

// dragging and dropping boxes
function dragStart(e) {
    //console.log('drag starts...');
    e.dataTransfer.setData('text/plain', e.target.id);
 }
 function dragEnter(e) {
    e.preventDefault();
    e.target.classList.add('drag-over');
 }
 function dragOver(e) {
    e.preventDefault();
    e.target.classList.add('drag-over');
 }
 function dragLeave(e) {
    e.target.classList.remove('drag-over');
 }
 function drop(e) {
    e.target.classList.remove('drag-over');

    // get the draggable element
    let id = e.dataTransfer.getData('text/plain');
    let draggable = document.getElementById(id);

    // get box indices
    let index_draggable = Number(id.split(' ')[1]);
    let index_target = Number(e.target.id.split(' ')[1]);
    var draggable_x = null;
    var draggable_y = null;
    var target_x = null;
    var target_y = null;
    if (compositionArray[index_draggable] instanceof Box){
        draggable_x = compositionArray[index_draggable].x;
        draggable_y = compositionArray[index_draggable].y;
    }
    if (compositionArray[index_target] instanceof Box){
        target_x = compositionArray[index_target].x;
        target_y = compositionArray[index_target].x;
    }

    if (index_draggable != index_target){
        
        // locate boxes to swap
        let target_node = e.target.parentElement.children[index_target];
        let draggable_node = e.target.parentElement.children[index_draggable];
        console.log('swapping box '+ index_draggable + ' with box '+ index_target);
        let parent = e.target.parentElement;
        // swap boxes
        exchangeElements(draggable_node, target_node);
        [compositionArray[index_draggable], compositionArray[index_target]] = [compositionArray[index_target], compositionArray[index_draggable]];
        console.log(compositionArray);

        // correct ids
        let new_target_node = document.getElementById('box '+ (index_target));
        let new_draggable_node = document.getElementById('box '+ (index_draggable)); 
        new_target_node.id = 'box ' + (index_draggable);
        new_draggable_node.id = 'box ' + (index_target);
    }
    // display the draggable element
    draggable.classList.remove('hide');
    // update scatterplot representation
    renderPath();

}

// exchange boxes
function exchangeElements(element1, element2){
    var clonedElement1 = element1.cloneNode(true);
    var clonedElement2 = element2.cloneNode(true);
    clonedElement1.addEventListener('dragstart', dragStart);
    clonedElement1.addEventListener('dragenter', dragEnter)
    clonedElement1.addEventListener('dragover', dragOver);
    clonedElement1.addEventListener('dragleave', dragLeave);
    clonedElement1.addEventListener('drop', drop);
    clonedElement2.addEventListener('dragstart', dragStart);
    clonedElement2.addEventListener('dragenter', dragEnter)
    clonedElement2.addEventListener('dragover', dragOver);
    clonedElement2.addEventListener('dragleave', dragLeave);
    clonedElement2.addEventListener('drop', drop);
    if (clonedElement1.children.length > 0){
        clonedElement1.children[0].addEventListener('dragstart', dragImgInsideBox);
        clonedElement1.children[0].addEventListener('dragenter', dragImgInsideBox)
        clonedElement1.children[0].addEventListener('dragover', dragImgInsideBox);
        clonedElement1.children[0].addEventListener('dragleave', dragImgInsideBox);
        clonedElement1.children[0].addEventListener('drop', dragImgInsideBox);
    }
    if (clonedElement2.children.length > 0){
        clonedElement2.children[0].addEventListener('dragstart', dragImgInsideBox);
        clonedElement2.children[0].addEventListener('dragenter', dragImgInsideBox)
        clonedElement2.children[0].addEventListener('dragover', dragImgInsideBox);
        clonedElement2.children[0].addEventListener('dragleave', dragImgInsideBox);
        clonedElement2.children[0].addEventListener('drop', dragImgInsideBox);
    }
    element2.parentNode.replaceChild(clonedElement1, element2);
    element1.parentNode.replaceChild(clonedElement2, element1);
    observer.observe(clonedElement1);
    observer.observe(clonedElement2);
}

function findIndexGivenCoords(x_coord, y_coord){
    // DANGEROUSS!!!!
    for (var i = 0; i < x.length; i++) {
        if(x[i] == x_coord && y[i] == y_coord){
            return i;
        }
    }
}

// implement trash bin
function dropOnTrashBin(e) {
    // update class list
    e.target.classList.remove('drag-over');
    // find and remove draggable element
    let id_draggable = e.dataTransfer.getData('text/plain');
    let draggable = document.getElementById(id_draggable);
    let parent = draggable.parentNode
    // reset color and size of scatterplot element
    let removed_box_index = Number(id_draggable.split(' ')[1]);
    if (compositionArray[removed_box_index] instanceof Box){
        pointToBasic(compositionArray[removed_box_index].arrayIndex);
        // remove scatterplot element
        //var scatterplot_index = findIndexGivenCoords(compositionArray[removed_box_index].x, compositionArray[removed_box_index].y);
        //sizes[scatterplot_index] = base_size;
        //colors[scatterplot_index] = base_color;
        // update visualization
    }
    draggable.remove();
    numBoxes -= 1;
    // adjust IDs of remaining boxes
    for (var i = removed_box_index; i < numBoxes; i++) {
        var old_id = parent.children[i].id.split(' ');
        old_id[1] = Number(old_id[1])-1;
        parent.children[i].id = old_id.join(' ');
    }
    // remove item from composition array
    let comp_index = Number(id_draggable.split(' ')[1]);
    compositionArray.splice(comp_index, 1);
    // reduce number of boxes
    console.log('removing element: '+ comp_index);
    console.log('composition array: '+ compositionArray);
    // update visualization
    renderPath();
}

// trash bin functionalities
var trashBin = document.getElementById('bin');
trashBin.addEventListener('dragover', dragOver);
trashBin.addEventListener('dragenter', dragEnter)
trashBin.addEventListener('dragleave', dragLeave);
trashBin.addEventListener('drop', dropOnTrashBin);
trashBin.parentNode.addEventListener('dragover', dragImgInsideBox);
trashBin.parentNode.addEventListener('dragenter', dragImgInsideBox);
trashBin.parentNode.addEventListener('dragleave', dragImgInsideBox);
trashBin.parentNode.addEventListener('drop', dragImgInsideBox);

// check for resize event
const observer = new ResizeObserver(function(mutations) {
    var resizedID = mutations[0].target.attributes.id.nodeValue;
    var resized_newHeight = mutations[0].contentRect.height; // height in px
    // scale px to width of marker: 100px = 20px marker
    console.log('resizing '+resizedID);
    console.log(compositionArray);

    // update box duration in composition array
    var boxNumber = Number(resizedID.split(' ')[1]);
    var heightToSec = 0.2;
    if(compositionArray[boxNumber] && resized_newHeight != 0){
        compositionArray[boxNumber].duration = resized_newHeight * heightToSec;
    }

    // check if resized element is a box
    if (compositionArray[boxNumber] instanceof Box){
        // find index of resized element in array of all point coordinates
        //var index = findIndexGivenCoords(compositionArray[boxNumber].x, compositionArray[boxNumber].y);
        //resize marker
        //var heightToPointSize = 0.25;
        //sizes[index] = resized_newHeight * heightToPointSize;
        //var update = {'marker':{color:colors, size:sizes, opacity:opacity}};
        //Plotly.restyle('scatterPlot', update, 0);
    }
    //console.log(mutations[0].target.attributes.id);
    //console.log(mutations[0].contentRect.width, mutations[0].contentRect.height);
});
