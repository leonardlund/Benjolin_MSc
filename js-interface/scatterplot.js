/* 
todo:
- click on arrows
- update duration when larger box
- highlighting arrows
- time limit or infinite composition bar?
- garbage bin to remove elements dynamically
- OSC integration
- play and stop buttons
- load latent space from file
- latent space array as global var
- completely responsive webpage
- touchscreen?

logic: 
- you can't click on point if it's the last one being clicked
*/

// Read lantent space coordinates
/*const fs = require('fs');

fs.readFile('scatterplot.txt', (err, data) => {
  if (err) throw err;
  console.log(data.toString());
});*/


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

// create scatterplot
var myScatterPlot = document.getElementById('scatterPlot'), 
    x = new Float32Array([1,2,3,4,5,6,0,4,-1,-2,-3,-5,-6]),
    y = new Float32Array([1,6,3,6,1,3,8,2,-3,-7,-2,-8,-6]),
    colors = ['#00000','#00000','#00000','#00000','#00000','#00000','#00000',
            '#00000','#00000','#00000','#00000','#00000','#00000'],
    sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    opacity = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    data = [ { 
        x:x, y:y, 
        type:'scatter',
        mode:'markers',
        marker:{size:sizes, color:colors, opacity:opacity} 
    } ],
    line = {

    },
    layout = {
        xaxis: {
            range: [ -10, 10 ],
            showgrid: true,
            zeroline: false,
            showline: false
        },
        yaxis: {
            range: [ -10, 10 ],
            showgrid: true,
            zeroline: false,
            showline: false
        },
        title:'Latent space',
        showlegend: false
    };

Plotly.newPlot('scatterPlot', data, layout);

// handle clicks on scatterplot
myScatterPlot.on('plotly_click', function(data){
    
    if (data.points[0].curveNumber == 0){
    // select a random color
    var randomcolor = '#'+(0x1000000+Math.random()*0xffffff).toString(16).substr(1,6);
    // change color and size of selected point
    var pn='',
        tn='',
        colors=[];
    
    // only take into account trace 0
    pn = data.points[0].pointNumber;
    tn = data.points[0].curveNumber;
    colors = data.points[0].data.marker.color;
    sizes = data.points[0].data.marker.size;
    colors[pn] = randomcolor;
    sizes[pn] = 15;
    var update = {'marker':{color: colors, size:sizes, opacity:opacity}};
    Plotly.restyle('scatterPlot', update, [tn]);

    var pts = '';
    pts = 'x = '+data.points[0].x +'\ny = '+data.points[0].y.toPrecision(4) + '\n\n';
    var x = data.points[0].x;
    var y = data.points[0].y;

    // create box with random color
    addBox(randomcolor, x, y); 
    console.log(pts);
    // send OSC message

    }
});


// check for resize event
const observer = new ResizeObserver(function(mutations) {
    //console.clear()
    var resizedID = mutations[0].target.attributes.id.nodeValue;
    var resized_newHeight = mutations[0].contentRect.height; // height in px
    // scale px to width of marker: 100px = 20px marker
    console.log(resizedID);

    // update box duration in composition array
    var boxNumber = Number(resizedID.split(' ')[1]);
    var heightToSec = 1;
    compositionArray[boxNumber].duration = resized_newHeight * heightToSec;

    // check if resized element is a box
    if (resizedID.split(' ').length > 2){
        resize_x = Number(resizedID.split(' ')[2]);
        resize_y = Number(resizedID.split(' ')[3]);
    }
    // find index of resized element in array
    var index = findIndexGivenCoords(resize_x, resize_y);
    //resize marker
    var heightToPointSize = 0.25;
    sizes[index] = resized_newHeight * heightToPointSize;
    var update = {'marker':{color:colors, size:sizes, opacity:opacity}};
    Plotly.restyle('scatterPlot', update, 0);
    
    //console.log(mutations[0].target.attributes.id);
    //console.log(mutations[0].contentRect.width, mutations[0].contentRect.height);
});

function findIndexGivenCoords(x_coord, y_coord){
    // assumes there are no duplicates
    for (var i = 0; i < x.length; ++i) {
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
    newBox.id = 'box '+numBoxes+' '+x+' '+y;
    newBox.style["background-color"] = randomcolor;
    newBox.style["height"] = '10vh';
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
    var duration = 10;
    compositionArray.push(new Box(x, y, duration));
    render();
    console.log(compositionArray);
    observer.observe(newBox);
}

function clickOnBox(e) {
    e.target.classList.add('click-on-box');
}

// highlight click on box
let clicked = false;
document.addEventListener('mousedown', e => { clicked = true; });
document.addEventListener('mousemove', e => { clicked = false; });
document.addEventListener('mouseup', event => {
    //console.log(event.target.className);
    if(clicked) {
        if (event.target.className == 'box'){
            // highlight box
            event.target.classList.add('click-on-box');
            // de-highlight all other boxes
            var all_click_on_box = document.getElementsByClassName('click-on-box');
            for (var i = 0; i < all_click_on_box.length; ++i) {
                if(all_click_on_box[i].id != event.target.id){
                    all_click_on_box[i].classList.remove('click-on-box');
                }
            }
            // highlight marker on plot (decrease opacity of all the other markers)
            highlight_x = Number(event.target.id.split(' ')[2]);
            highlight_y = Number(event.target.id.split(' ')[3]);
            // find index of resized element in array
            var index = findIndexGivenCoords(highlight_x, highlight_y);
            // reset opacity of all other elements
            for (var i = 0; i < opacity.length; ++i) {
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
        else if (event.target.className == 'meander'){
            // highlight meander
            // highlight arrow on plot (decrease opacity of all the other markers)
            // de-highlight all other boxes
            var all_click_on_box = document.getElementsByClassName('click-on-box');
            for (var i = 0; i < all_click_on_box.length; ++i) {
                if(all_click_on_box[i].id != event.target.id){
                    all_click_on_box[i].classList.remove('click-on-box');
                }
            }
        }
        else if (event.target.className == 'crossfade'){
            // highlight crossfade
            // highlight arrow on plot (decrease opacity of all the other markers)
            // de-highlight all other boxes
            var all_click_on_box = document.getElementsByClassName('click-on-box');
            for (var i = 0; i < all_click_on_box.length; ++i) {
                if(all_click_on_box[i].id != event.target.id){
                    all_click_on_box[i].classList.remove('click-on-box');
                }
            }
        }
        else{
            // de-highlight all other boxes
            var all_click_on_box = document.getElementsByClassName('click-on-box');
            for (var i = 0; i < all_click_on_box.length; ++i) {
                if(all_click_on_box[i].id != event.target.id){
                    all_click_on_box[i].classList.remove('click-on-box');
                }
            }
            for (var i = 0; i < opacity.length; ++i) {
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
    }
    clicked = false;
})

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
    if (id.split(' ').length > 2){
        draggable_x = Number(id.split(' ')[2]);
        draggable_y = Number(id.split(' ')[3]);
    }
    if (e.target.id.split(' ').length > 2){
        target_x = Number(e.target.id.split(' ')[2]);
        target_y = Number(e.target.id.split(' ')[3]);
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
        if (target_x != null && draggable_x != null){
            // draggable and target are boxes
            new_target_node = document.getElementById('box '+ (index_target)+' '+target_x+' '+target_y);
            new_draggable_node = document.getElementById('box '+ (index_draggable)+' '+draggable_x+' '+draggable_y); 
            new_target_node.id = 'box ' + (index_draggable)+' '+(target_x)+' '+(target_y);
            new_draggable_node.id = 'box ' + (index_target)+' '+(draggable_x)+' '+(draggable_y);
        }
        else if (target_x != null && draggable_x == null){
            // target is box and draggable is arrow
            new_target_node = document.getElementById('box '+ (index_target)+' '+target_x+' '+target_y);
            new_draggable_node = document.getElementById('box '+ (index_draggable)); 
            new_target_node.id = 'box ' + (index_draggable)+' '+target_x+' '+target_y;
            new_draggable_node.id = 'box ' + (index_target);
        }
        else if (target_x == null && draggable_x != null){
            // draggable is box and target is arrow
            new_target_node = document.getElementById('box '+ (index_target));
            new_draggable_node = document.getElementById('box '+ (index_draggable)+' '+draggable_x+' '+draggable_y); 
            new_target_node.id = 'box ' + (index_draggable);
            new_draggable_node.id = 'box ' + (index_target)+' '+draggable_x+' '+draggable_y;
        }
        else{
            new_target_node = document.getElementById('box '+ (index_target));
            new_draggable_node = document.getElementById('box '+ (index_draggable)); 
            new_target_node.id = 'box ' + (index_draggable);
            new_draggable_node.id = 'box ' + (index_target);
        }
    }
    // display the draggable element
    draggable.classList.remove('hide');
    render();
}

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
}


function render(){
    //remove all traces besides trace 0
    var graphDiv = document.getElementById('scatterPlot');
    for (let i = 1; i < graphDiv.data.length-1; i++) {
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
    newCrossfade.style["height"] = '10vh';
    newCrossfade.style["width"] = '60%';
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

    var duration = 10;
    compositionArray.push(new Crossfade(duration));
    render();
    console.log(compositionArray);
}

// stop images to be draggable
function dragImgInsideBox(e) {
    e.preventDefault();
    e.stopPropagation(); 
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
    newMeander.style["height"] = '10vh';
    newMeander.style["width"] = '60%';
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

    var duration = 10;
    compositionArray.push(new Meander(duration));
    render();
    console.log(compositionArray);

}

// handling navbar elements
const insert_crossfade = document.getElementById("insert-crossfade"); 
insert_crossfade.addEventListener("click", addCrossfade); 
const insert_meander = document.getElementById("insert-meander"); 
insert_meander.addEventListener("click", addMeander); 


// OSC communication
/*var oscPort = new osc.WebSocketPort({
    url: "ws://localhost:8081", // URL to your Web Socket server.
    metadata: true
});

oscPort.open();

oscPort.on("message", function (oscMsg) {
    console.log("An OSC message just arrived!", oscMsg);
});*/