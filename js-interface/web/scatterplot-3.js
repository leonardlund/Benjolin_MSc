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

class Box{
    constructor(x, y, duration){
        this.x = x;
        this.y = y;
        this.duration = duration;
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
const base_size = 10;
const base_color = '#00000';
const base_opacity = 1;

// create scatterplot
var mainCanvas = document.getElementById('scatterPlot'); 


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
    mainCanvas.appendChild( renderer.domElement );

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
    mainCanvas.appendChild( stats.dom );


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









// check for resize event
const observer = new ResizeObserver(function(mutations) {
    //console.clear()

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
        var index = findIndexGivenCoords(compositionArray[boxNumber].x, compositionArray[boxNumber].y);
        //resize marker
        var heightToPointSize = 0.25;
        sizes[index] = resized_newHeight * heightToPointSize;
        var update = {'marker':{color:colors, size:sizes, opacity:opacity}};
        Plotly.restyle('scatterPlot', update, 0);
    }
    //console.log(mutations[0].target.attributes.id);
    //console.log(mutations[0].contentRect.width, mutations[0].contentRect.height);
});

function findIndexGivenCoords(x_coord, y_coord){
    // assumes there are no duplicates
    for (var i = 0; i < x.length; i++) {
        if(x[i] == x_coord && y[i] == y_coord){
            return i;
        }
    }
}

// add a box when a point on the scatterplot is clicked
var numBoxes = 0
function addBox(randomcolor, x, y) {
    newBox = document.createElement("div");
    newBox.className = 'box';
    newBox.id = 'box '+numBoxes;
    newBox.style["background-color"] = randomcolor;
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
    compositionArray.push(new Box(x, y, duration));
    render();
    //console.log(compositionArray);
    observer.observe(newBox);
}

function clickOnBox(e) {
    e.target.classList.add('click-on-box');
}

// highlight click on box
//let clicked = false;
//document.addEventListener('mousedown', e => { clicked = true; });
//document.addEventListener('mousemove', e => { clicked = false; });
//document.addEventListener('mouseup', event => {
document.addEventListener('click', event => {
    highlightBoxElement(event.target);
})

function highlightBoxElement(element){
    if (element.className == 'box'){
        // highlight box
        element.classList.add('click-on-box');
        // de-highlight all other boxes
        var all_click_on_box = document.getElementsByClassName('click-on-box');
        for (var i = 0; i < all_click_on_box.length; i++) {
            if(all_click_on_box[i].id != element.id){
                all_click_on_box[i].classList.remove('click-on-box');
            }
        }
        // highlight marker on plot (decrease opacity of all the other markers)
        var boxNumber = Number(element.id.split(' ')[1]);
        // listen to box
        sendBox(compositionArray[boxNumber].x, compositionArray[boxNumber].y);
        // find index of resized element in array
        var index = findIndexGivenCoords(compositionArray[boxNumber].x, compositionArray[boxNumber].y);
        opacity[index] = 1;
        // reset opacity of all other elements
        for (var i = 0; i < opacity.length; i++) {
            if(i != index){
                opacity[i] = 0.3;
            }
        }
        var update = {'marker':{color:colors, size:sizes, opacity:opacity}};
        Plotly.restyle('scatterPlot', update, 0);
        // reset opacity of lines too
        var graphDiv = document.getElementById('scatterPlot');
        for (let i = 1; i < graphDiv.data.length; i++) {
            var update = graphDiv.data[i].line;
            update.opacity = 0.2;
            Plotly.restyle('scatterPlot', update, i);    
        }
    }
    else if (element.classList.contains('meander')){
        // highlight meander
        element.classList.add('click-on-box');
        // send OSC and listen to meander 
        var compositionIndex = Number(element.id.split(' ')[1]);
        if (compositionIndex != 0 && compositionArray[compositionIndex] instanceof Meander){
            // check if before and after there are boxes
            if (compositionArray[compositionIndex-1] instanceof Box && compositionArray[compositionIndex+1] instanceof Box){
                sendMeander(compositionArray[compositionIndex-1].x, compositionArray[compositionIndex-1].y, 
                    compositionArray[compositionIndex+1].x, compositionArray[compositionIndex+1].y, 
                    compositionArray[compositionIndex].duration);
    }
        }
        // highlight arrow on plot (decrease opacity of all the other markers)
        // de-highlight all other boxes
        var all_click_on_box = document.getElementsByClassName('click-on-box');
        for (var i = 0; i < all_click_on_box.length; i++) {
            if(all_click_on_box[i].id != element.id){
                all_click_on_box[i].classList.remove('click-on-box');
            }
        }
    }
    else if (element.classList.contains('crossfade')){
        // highlight crossfade
        element.classList.add('click-on-box');
        // send OSC and listen to crossfade 
        var compositionIndex = Number(element.id.split(' ')[1]);
        if (compositionIndex != 0 && compositionArray[compositionIndex] instanceof Crossfade){
            // check if before and after there are boxes
            if (compositionArray[compositionIndex-1] instanceof Box && compositionArray[compositionIndex+1] instanceof Box){
                sendCrossfade(compositionArray[compositionIndex-1].x, compositionArray[compositionIndex-1].y, 
                            compositionArray[compositionIndex+1].x, compositionArray[compositionIndex+1].y, 
                            compositionArray[compositionIndex].duration);
            }
        }
        // highlight arrow on plot (decrease opacity of all the other markers)
        // de-highlight all other boxes
        var all_click_on_box = document.getElementsByClassName('click-on-box');
        for (var i = 0; i < all_click_on_box.length; i++) {
            if(all_click_on_box[i].id != element.id){
                all_click_on_box[i].classList.remove('click-on-box');
            }
        }
    }
    else{
        // de-highlight all other boxes
        var all_click_on_box = document.getElementsByClassName('click-on-box');
        for (var i = 0; i < all_click_on_box.length; i++) {
            if(all_click_on_box[i].id != element.id){
                all_click_on_box[i].classList.remove('click-on-box');
            }
        }
        for (var i = 0; i < opacity.length; i++) {
                opacity[i] = 1;
        }
        var update = {'line':{color:colors, size:sizes, opacity:opacity}};
        Plotly.restyle('scatterPlot', update, 0);    
        // reset opacity of lines too
        var graphDiv = document.getElementById('scatterPlot');
        for (let i = 1; i < graphDiv.data.length; i++) {
            var update = graphDiv.data[i].line;
            update.opacity = 1;
            Plotly.restyle('scatterPlot', update, i);    
        }
    }
};

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
    const id = e.dataTransfer.getData('text/plain');
    const draggable = document.getElementById(id);

    // get box indices
    const index_draggable = Number(id.split(' ')[1]);
    const index_target = Number(e.target.id.split(' ')[1]);
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
        const target_node = e.target.parentElement.children[index_target];
        const draggable_node = e.target.parentElement.children[index_draggable];
        console.log('swapping box '+ index_draggable + ' with box '+ index_target);
        const parent = e.target.parentElement;
        // swap boxes
        exchangeElements(draggable_node, target_node);
        [compositionArray[index_draggable], compositionArray[index_target]] = [compositionArray[index_target], compositionArray[index_draggable]];
        console.log(compositionArray);

        // correct ids
        new_target_node = document.getElementById('box '+ (index_target));
        new_draggable_node = document.getElementById('box '+ (index_draggable)); 
        new_target_node.id = 'box ' + (index_draggable);
        new_draggable_node.id = 'box ' + (index_target);
    }
    // display the draggable element
    draggable.classList.remove('hide');
    render();
}

// implement trash bin
function dropOnTrashBin(e) {
    // update class list
    e.target.classList.remove('drag-over');
    // find and remove draggable element
    const id_draggable = e.dataTransfer.getData('text/plain');
    const draggable = document.getElementById(id_draggable);
    const parent = draggable.parentNode
    // reset color and size of scatterplot element
    removed_box_index = Number(id_draggable.split(' ')[1]);
    if (compositionArray[removed_box_index] instanceof Box){
        // remove scatterplot element
        var scatterplot_index = findIndexGivenCoords(compositionArray[removed_box_index].x, compositionArray[removed_box_index].y);
        sizes[scatterplot_index] = base_size;
        colors[scatterplot_index] = base_color;
        var update = {'marker':{color: colors, size:sizes, opacity:opacity}};
        Plotly.restyle('scatterPlot', update, 0);
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
    const comp_index = Number(id_draggable.split(' ')[1]);
    compositionArray.splice(comp_index, 1);
    // reduce number of boxes
    console.log('removing element: '+ comp_index);
    console.log('composition array: '+ compositionArray);
    render(); // render scatterplot
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

function render(){
    //remove all traces besides trace 0
    var graphDiv = document.getElementById('scatterPlot');
    for (let i = 1; i < graphDiv.data.length; i++) {
        Plotly.deleteTraces(graphDiv, i);
    }
    //draw new traces
    for (let i = 0; i < compositionArray.length; i++) {
        if (i !=0){ // check if not on the first object
            if (compositionArray[i] instanceof Meander){
                // check if before and after there are boxes
                if (compositionArray[i-1] instanceof Box && compositionArray[i+1] instanceof Box){ 
                    // add dotted line if meander
                    Plotly.addTraces('scatterPlot', {
                        x: [compositionArray[i-1].x,compositionArray[i+1].x],
                        y: [compositionArray[i-1].y,compositionArray[i+1].y],
                        hoverinfo:'skip',
                        mode: 'lines',
                        line: {
                            color: 'rgb(219, 64, 82)',
                            width: 2,
                            dash: 'dot', // solid, dash, dashdot, dot, dash
                            opacity: 1
                          }
                    });
                }
            }
            if (compositionArray[i] instanceof Crossfade){
                // check if before and after there are boxes
                if (compositionArray[i-1] instanceof Box && compositionArray[i+1] instanceof Box){ 
                    // add dashed line if crossfade
                    Plotly.addTraces('scatterPlot', {
                        x: [compositionArray[i-1].x,compositionArray[i+1].x],
                        y: [compositionArray[i-1].y,compositionArray[i+1].y],
                        hoverinfo:'skip',
                        mode: 'lines',
                        line: {
                            color: 'rgb(219, 64, 82)',
                            width: 2,
                            opacity: 1,
                            dash: 'dash' // solid, dash, dashdot, dot, dash
                          }
                    });
                }
            }
            if (compositionArray[i] instanceof Box){
                // check if before there is a box
                if (compositionArray[i-1] instanceof Box){ 
                    // add straight line if jump
                    Plotly.addTraces('scatterPlot', {
                        x: [compositionArray[i-1].x,compositionArray[i].x],
                        y: [compositionArray[i-1].y,compositionArray[i].y],
                        hoverinfo:'skip',
                        mode: 'lines',
                        line: {
                            color: 'rgb(219, 64, 82)',
                            width: 2,
                            opacity: 1,
                            dash: 'solid' // solid, dash, dashdot, dot, dash
                          }
                    });
                }
            }
        }
    } 
}

// stop images inside boxes to be draggable
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
    render();
    console.log(compositionArray);
    observer.observe(newCrossfade);
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
    render();
    console.log(compositionArray);
    observer.observe(newMeander);

}

// handling navbar elements
const insert_crossfade = document.getElementById("insert-crossfade"); 
insert_crossfade.addEventListener("click", addCrossfade); 
const insert_meander = document.getElementById("insert-meander"); 
insert_meander.addEventListener("click", addMeander); 

function play(){
    var timeout = 0;
    for (let i = 0; i < compositionArray.length; i++) {
        if (compositionArray[i] instanceof Box){
            setTimeout(function() {
                console.log('playing: ',compositionArray[i]);
                //sendBox(compositionArray[i].x, compositionArray[i].y);
                highlightBoxElement(document.getElementById('box '+i));
            }, timeout);
        }
        else if (compositionArray[i] instanceof Meander){
            setTimeout(function() {
                console.log(compositionArray[i]);
                highlightBoxElement(document.getElementById('box '+i));
                //sendBox(compositionArray[i].x, compositionArray[i].y);
            }, timeout);
        }
        else if (compositionArray[i] instanceof Crossfade){
            setTimeout(function() {
                console.log(compositionArray[i]);
                highlightBoxElement(document.getElementById('box '+i));
                //sendBox(compositionArray[i].x, compositionArray[i].y);
            }, timeout);
        }
        timeout += (compositionArray[i].duration * 1000);
        //console.log(timeout);
    }
};