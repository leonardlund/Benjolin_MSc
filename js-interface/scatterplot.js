// Requiring fs module in which 
// readFile function is defined.
/*const fs = require('fs');

fs.readFile('scatterplot.txt', (err, data) => {
  if (err) throw err;
  console.log(data.toString());
});*/


// dictionary containing instruction for playback
const compositionDict = {
    Box: {
        coordinates: [0.2, 0.1],
        duration: 10
    },
    Box: {
        coordinates: [0.4, -0.2],
        duration: 5
    },
    Meander: {
        pointsList_x: [0.4, -0.2, 0.1, 0.5],
        pointsList_y: [0.2, 0.1, 1, 0.3],
        duration: 10
    },
    Box: {},
    Crossfade:{}
};


// add a box when a point on the scatterplot is clicked
var numBoxes = 0
function addBox(randomcolor) { 
    newBox = document.createElement("div");
    newBox.className = 'box';
    newBox.id = 'box ' + numBoxes;
    newBox.style["background-color"] = randomcolor;
    newBox.style["height"] = '10vh';
    newBox.style["width"] = '60%';
    newBox.draggable = 'true';
    newBox.addEventListener('dragstart', dragStart);
    newBox.addEventListener('dragenter', dragEnter)
    newBox.addEventListener('dragover', dragOver);
    newBox.addEventListener('dragleave', dragLeave);
    newBox.addEventListener('drop', drop);
    document.getElementById("compose-bar").appendChild(newBox); 
    console.log(newBox.id);
    numBoxes += 1;
} 

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

    if (index_draggable != index_target){
        // locate boxes to swap
        const target_node = e.target.parentElement.children[index_target];
        const draggable_node = e.target.parentElement.children[index_draggable];
        console.log('swapping box '+ index_draggable + ' with box '+ index_target);
        const parent = e.target.parentElement;
        // swap boxes
        exchangeElements(draggable_node, target_node);
        new_target_node = document.getElementById('box '+ index_target);
        new_draggable_node = document.getElementById('box '+ index_draggable);
        new_target_node.id = 'box ' + (index_draggable);
        new_draggable_node.id = 'box ' + (index_target);
    }

    // display the draggable element
    draggable.classList.remove('hide');
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

// create scatterplot
var myScatterPlot = document.getElementById('scatterPlot'), 
    x = new Float32Array([1,2,3,4,5,6,0,4,-1,-2,-3,-5,-6]),
    y = new Float32Array([1,6,3,6,1,3,8,2,-3,-7,-2,-8,-6]),
    colors = ['#00000','#00000','#00000','#00000','#00000','#00000','#00000',
            '#00000','#00000','#00000','#00000','#00000','#00000'],
    sizes = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    data = [ { 
        x:x, y:y, 
        type:'scatter',
        mode:'markers', 
        marker:{size:sizes, color:colors} } ],
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
        title:'Latent space'
    };

Plotly.newPlot('scatterPlot', data, layout);

// handle clicks on scatterplot
myScatterPlot.on('plotly_click', function(data){
    
    // select a random color
    var randomcolor = '#'+(0x1000000+Math.random()*0xffffff).toString(16).substr(1,6);
    // create box with random color
    addBox(randomcolor); 
    // change color and size of selected point
    var pn='',
        tn='',
        colors=[];
    for(var i=0; i < data.points.length; i++){ //iterate over traces
      pn = data.points[i].pointNumber;
      tn = data.points[i].curveNumber;
      colors = data.points[i].data.marker.color;
      sizes = data.points[i].data.marker.size;
    };
    colors[pn] = randomcolor;
    sizes[pn] = 15;
    var update = {'marker':{color: colors, size:sizes}};
    Plotly.restyle('scatterPlot', update, [tn]);

    // print to console
    var pts = '';
    for(var i=0; i < data.points.length; i++){ //iterate over traces
        pts = 'x = '+data.points[i].x +'\ny = '+
            data.points[i].y.toPrecision(4) + '\n\n';
    }
    console.log(pts);
    // send OSC message
});

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

}

// handling navbar elements
const insert_crossfade = document.getElementById("insert-crossfade"); 
insert_crossfade.addEventListener("click", addCrossfade); 
const insert_meander = document.getElementById("insert-meander"); 
insert_meander.addEventListener("click", addMeander); 



// handle click on box
// data structure with boxes and points and order
// add arrow
// play the whole sequence


// things to do: 
// make navbar on top
// update data structure on event
// on click of point in graph: create a box, color the clicked point, listen to how that sounds like
// on click on box: highlight point, listen to how that point sounds like
// on click+drag on box: exchange box position in the timeline with box over which it is dragged
// on 
// how to add arrows?



// OSC communication
/*var oscPort = new osc.WebSocketPort({
    url: "ws://localhost:8081", // URL to your Web Socket server.
    metadata: true
});

oscPort.open();

oscPort.on("message", function (oscMsg) {
    console.log("An OSC message just arrived!", oscMsg);
});*/