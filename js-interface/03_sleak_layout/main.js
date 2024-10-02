let numBoxes = 0;

// window.innerHeight = 514 = 2 min = 120 s
// a radius of 55 means a diameter of 110 --> 110 / 514 = 0.214. multiplied by 120 --> 25.16 s
const MAX_R = 55; // corresponds to a duration 25.16 s
const MIN_R = 5; // circle_duration[s] = (r*2 / window.innerHeight) * 120 
const BASIC_ELEMENT_T = 5000 // new element when created has duration 5s
const MAX_T = 10000 // max element duration is 20s
const MIN_T = 1000 // min element duration is 1s

const MAX_COMPOSITION_DURATION = 120000 // 12000 milliseconds = 2 minutes
function timesToPxHeight (time_ms) {
    // adaptively calculate element height in pixel corresponding to time in milliseconds
    // window height : max duration = height_px : time_ms
    let conversion_factor = window.innerHeight / MAX_COMPOSITION_DURATION;
    let height_px = time_ms * conversion_factor;
    return height_px
}

const COMPOSITION_BAR_WIDTH_PX = 150;
const MARGIN_PX = 20
// BOX --> CIRCLE
function drawBox(colorHue){
    let newBox = document.createElement("div");
    newBox.id = "box "+numBoxes;
    document.getElementById("composition-bar").appendChild(newBox); 

    let boxStartHeight = timesToPxHeight( BASIC_ELEMENT_T );
    var R = Raphael("box "+numBoxes, COMPOSITION_BAR_WIDTH_PX, boxStartHeight + MARGIN_PX );
    var s = R.circle( COMPOSITION_BAR_WIDTH_PX/2 , (boxStartHeight + MARGIN_PX) / 2 , boxStartHeight / 2 ).attr({
            fill: "hsb("+colorHue+", .5, .5)",
            stroke: "none",
            opacity: .3
        });
    var c = R.circle( COMPOSITION_BAR_WIDTH_PX/2 , (boxStartHeight + MARGIN_PX) / 2 , boxStartHeight / 2 ).attr({
            fill: "none",
            stroke: "hsb("+colorHue+", 1, 1)",
            "stroke-width": 8,
            opacity: 0.3
        });
    c.sized = s;
    c.parentDiv = document.getElementById(newBox.id);
    c.raph = R;
    c.drag(move, start, up);
    s.outer = c;
    //c.hover(hoverIn_outer, hoverOut_outer);
    //s.hover(hoverIn, hoverOut);
    
    newBox.draggable = 'true';
    numBoxes += 1;
    return R
}
// CIRCLE INTERACTIONS
var start = function () {
    this.ISMOVING = true;
    this.ox = this.attr("cx");    
    this.oy = this.attr("cy");
    this.or = this.attr("r");
    this.attr({opacity: 1});
    this.sized.ox = this.sized.attr("cx");    
    this.sized.oy = this.sized.attr("cy");
    this.sized.or = this.attr("r");
    this.sized.attr({opacity: 1});
};
var move = function (dx, dy) {
    let newr = this.or + (dy < 0 ? -1 : 1) * Math.sqrt(2*dy*dy);
    let max_r_px = timesToPxHeight( MAX_T );
    let min_r_px = timesToPxHeight( MIN_T );
    if ( newr < max_r_px/2 && newr > min_r_px/2 ) {
        this.attr({r: newr});
        this.sized.attr({r: newr });
        this.parentDiv.style["height"] = newr*2+20;
        this.raph.setSize(200, newr*2+20);
        this.attr({cy: (newr*2+20)/2});
        this.sized.attr({cy: (newr*2+20)/2});
    }
};
var up = function () {
    this.ISMOVING = false;
    this.attr({opacity: 0.05 });
    this.sized.attr({opacity: .8 });
}
// HOVER CIRCLE INTERACTIONS
var hoverIn_outer = function() {
    this.attr({"opacity": 1});
    this.sized.attr({"opacity": 1});
};
var hoverOut_outer = function() {
    if ( !this.ISMOVING ){
        this.attr({"opacity": 0.05});
        this.sized.attr({"opacity": 0.8});
    }
}
var hoverIn = function() {
    this.attr({"opacity": 1});
    this.outer.attr({"opacity": 1});
};
var hoverOut = function() {
    if ( !this.ISMOVING ){
        this.attr({"opacity": 0.8});
        this.outer.attr({"opacity": 0.05});
    }
}

// CROSSFADE
function drawCrossfade(){
    let newBox = document.createElement("div");
    newBox.id = "box "+numBoxes;
    document.getElementById("composition-bar").appendChild(newBox); 

    let boxStartHeight = timesToPxHeight( BASIC_ELEMENT_T );
    // from https://jsfiddle.net/TfE2X/
    var R = Raphael("box "+numBoxes, COMPOSITION_BAR_WIDTH_PX, boxStartHeight+MARGIN_PX);
    path = R.path("M"+(COMPOSITION_BAR_WIDTH_PX/2)+" 0L"+(COMPOSITION_BAR_WIDTH_PX/2)+" "+boxStartHeight).attr({
        stroke: '#FFFFFF',
        'stroke-width': 3,
        'arrow-end':'classic-wide-long',
        opacity: 0.3
    });
    var pathArray = path.attr("path");
    handle = R.circle(COMPOSITION_BAR_WIDTH_PX/2,boxStartHeight-5,10).attr({
        fill: "#FFFFFF",
        cursor: "pointer",
        "stroke-width": 10,
        stroke: "transparent",
        opacity: 0.3
    });
    handle.pathArray = pathArray;
    handle.path = path;
    handle.parentDiv = document.getElementById(newBox.id);
    handle.raph = R;
    handle.drag(move_crossfade, start_crossfade, up_crossfade);

    newBox.draggable = 'true';

    numBoxes += 1;
    return R
}
// CROSSFADE INTERACTIONS
var start_crossfade = function () {
    this.cy = this.attr("cy");
    this.attr({opacity: 1});
};
var move_crossfade = function (dx, dy) {
    var Y = this.cy + dy;
    let max_r_px = timesToPxHeight( MAX_T );
    let min_r_px = timesToPxHeight( MIN_T );
    if ( Y < max_r_px && Y > min_r_px ) {
        this.attr({ cy: Y });
        this.pathArray[1][2] = Y;
        this.path.attr({path: this.pathArray});
        this.parentDiv.style["height"] = Y+20;
        this.raph.setSize(200, Y+20);
    }
};
var up_crossfade = function () {
    this.attr({opacity: 0.3});
};


function drawMeander(){
    let newBox = document.createElement("div");
    newBox.id = "box "+numBoxes;
    document.getElementById("composition-bar").appendChild(newBox); 

    let boxStartHeight = timesToPxHeight( BASIC_ELEMENT_T );
    var R = Raphael("box "+numBoxes, COMPOSITION_BAR_WIDTH_PX, boxStartHeight+MARGIN_PX);
    path1 = R.path("M"+(COMPOSITION_BAR_WIDTH_PX/2)+" 0L"+(COMPOSITION_BAR_WIDTH_PX/2+15)+" "+ (boxStartHeight/4)).attr({
        stroke: '#FFFFFF',
        'stroke-width': 3,
        opacity: 0.3
    });
    path2 = R.path("M"+(COMPOSITION_BAR_WIDTH_PX/2+15)+" "+(boxStartHeight/4)+"L"+(COMPOSITION_BAR_WIDTH_PX/2-15)+" "+(boxStartHeight*2/4)).attr({
        stroke: '#FFFFFF',
        'stroke-width': 3,
        opacity: 0.3
    });
    path3 = R.path("M"+(COMPOSITION_BAR_WIDTH_PX/2-15)+" "+(boxStartHeight*2/4)+"L"+(COMPOSITION_BAR_WIDTH_PX/2)+" "+(boxStartHeight*3/4)).attr({
        stroke: '#FFFFFF',
        'stroke-width': 3,
        opacity: 0.3
    });
    path4 = R.path("M"+(COMPOSITION_BAR_WIDTH_PX/2)+" "+(boxStartHeight*3/4)+"L"+(COMPOSITION_BAR_WIDTH_PX/2)+" "+boxStartHeight).attr({
        stroke: '#FFFFFF',
        'stroke-width': 3,
        'arrow-end':'classic-wide-long',
        opacity: 0.3
    });

    var pathArray1 = path1.attr("path");
    var pathArray2 = path2.attr("path");
    var pathArray3 = path3.attr("path");
    var pathArray4 = path4.attr("path");
    handle_meander = R.circle(COMPOSITION_BAR_WIDTH_PX/2,boxStartHeight-5,10).attr({
        fill: "#FFFFFF",
        cursor: "pointer",
        "stroke-width": 10,
        stroke: "transparent",
        opacity: 0.3
    });
    handle_meander.pathArray1 = pathArray1;
    handle_meander.pathArray2 = pathArray2;
    handle_meander.pathArray3 = pathArray3;
    handle_meander.pathArray4 = pathArray4;
    handle_meander.path1 = path1;
    handle_meander.path2 = path2;
    handle_meander.path3 = path3;
    handle_meander.path4 = path4;
    handle_meander.parentDiv = document.getElementById("box "+numBoxes);
    handle_meander.raph = R;
    handle_meander.drag(move_meander, start_meander, up_meander);

    newBox.draggable = 'true';

    numBoxes += 1;
    return R
}


var start_meander = function () {
    this.cy = this.attr("cy");
    this.attr({opacity: 1});
};
var move_meander = function (dx, dy) {
    var Y = this.cy + dy;
    let max_r_px = timesToPxHeight( MAX_T );
    let min_r_px = timesToPxHeight( MIN_T );
    if ( Y < max_r_px && Y > min_r_px ) {
        this.attr({ cy: Y });
        this.pathArray1[1][2] = Y/4;
        this.pathArray2[0][2] = Y/4;
        this.pathArray2[1][2] = Y/4*2;
        this.pathArray3[0][2] = Y/4*2;
        this.pathArray3[1][2] = Y/4*3;
        this.pathArray4[0][2] = Y/4*3;
        this.pathArray4[1][2] = Y;
        this.path1.attr({path: this.pathArray1});
        this.path2.attr({path: this.pathArray2});
        this.path3.attr({path: this.pathArray3});
        this.path4.attr({path: this.pathArray4});
        this.parentDiv.style["height"] = Y+20;
        this.raph.setSize(200, Y+20);
    }
};
var up_meander = function () {
    this.attr({opacity: 0.3});
};

// DRAW TIMELINE
var R_timeline = Raphael("timeline", 50, window.innerHeight - (90 + 60 + 20) );
path_timeline = R_timeline.path("M25 0L25 "+(window.innerHeight - (90 + 60 +50))).attr({
    stroke: '#FFFFFF',
    'stroke-width': 1,
    'arrow-end':'classic-wide-long',
    opacity: 0.5
});

// DEMO OBJECTS (WHEN BOXES ARE EXCHANGED NEEDS TO BE EXCHANGED IN "raphaels")
let raphaels = [];
let R1 = drawBox(Math.random());
raphaels.push(R1)
let R2 = drawCrossfade();
raphaels.push(R2)
let R3 = drawBox(Math.random());
raphaels.push(R3)
let R4 = drawMeander();
raphaels.push(R4)
let R5 = drawBox(Math.random());
raphaels.push(R5)


// FUNCTION FOR HIGHLIGHTING BOXES
function highlightNone (event){
    //var hovered_on_id = Number(event.target.id.split(' ')[1]);
    for (var i = 0; i < raphaels.length; i++) {
        raphaels[i].forEach(function (el) 
        {
            el.attr({"opacity": 0.3});
        });
    }
}

function highlightBox (event, box_n, highlight_amt){    
    for (var i = 0; i < raphaels.length; i++) {
        raphaels[i].forEach(function (el) 
        {
            el.attr({"opacity": 0.3});
        });
    }
    raphaels[box_n].forEach(function (el) 
    {
        el.attr({"opacity": highlight_amt});
    });
}

function highlightAll (event, highlight_amt){    
    for (var i = 0; i < raphaels.length; i++) {
        raphaels[i].forEach(function (el) 
        {
            el.attr({"opacity": highlight_amt});
        });
    }
}


// HOVER FUNCTIONS
document.getElementById("scatterPlot").addEventListener("mouseover", (event) => {
    highlightNone(event); 
}); 

// MICRO-INTERACTIONS AT BUTTONS
document.getElementById("insert-crossfade").addEventListener("mouseover", (event) => {
    highlightNone(event); 
}); 
document.getElementById("insert-meander").addEventListener("mouseover", (event) => {
    highlightNone(event); 
}); 
document.getElementById("bin").addEventListener("mouseover", (event) => {
    highlightNone(event); 
}); 
document.getElementById("play").addEventListener("mouseover", (event) => {
    highlightAll(event, 0.7); 
}); 
document.getElementById("stop").addEventListener("mouseover", (event) => {
    highlightNone(event); 
}); 


// SET UP HOVER INTERACTIONS AT THE BOX CREATION
document.getElementById("box 0").addEventListener("mouseover", (event) => {
    highlightBox(event, 0, 0.7); 
}); 
document.getElementById("box 1").addEventListener("mouseover", (event) => {
    highlightBox(event, 1, 0.7); 
}); 
document.getElementById("box 2").addEventListener("mouseover", (event) => {
    highlightBox(event, 2, 0.7); 
}); 
document.getElementById("box 3").addEventListener("mouseover", (event) => {
    highlightBox(event, 3, 0.7); 
}); 
document.getElementById("box 4").addEventListener("mouseover", (event) => {
    highlightBox(event, 4, 0.7); 
}); 


// SET UP CLICK INTERACTIONS AT THE BOX CREATION
let SELECTED_ELEMENT = null;
document.getElementById("box 0").addEventListener("click", (event) => {
    let item_index = 0;
    SELECTED_ELEMENT = item_index;
    highlightBox(event, item_index, 1); 
}); 
document.getElementById("box 1").addEventListener("click", (event) => {
    let item_index = 1;
    SELECTED_ELEMENT = item_index;
    highlightBox(event, item_index, 1); 
}); 
document.getElementById("box 2").addEventListener("click", (event) => {
    let item_index = 2;
    SELECTED_ELEMENT = item_index;
    highlightBox(event, item_index, 1); 
}); 
document.getElementById("box 3").addEventListener("click", (event) => {
    let item_index = 3;
    SELECTED_ELEMENT = item_index;
    highlightBox(event, item_index, 1); 
}); 
document.getElementById("box 4").addEventListener("click", (event) => {
    let item_index = 4;
    SELECTED_ELEMENT = item_index;
    highlightBox(event, item_index, 1); 
}); 


// dragging and dropping boxes
function dragStart(e) {
    e.dataTransfer.setData('text/plain', e.target.id);
}

function dragEnter(e) {
    e.preventDefault();
    let entered_box_n = null;
    if ( e.target.nodeName == 'DIV' ){
        entered_box_n = Number(e.target.id.split(" ")[1]);
    } else if ( e.target.nodeName == 'svg' ){
        entered_box_n = Number(e.target.parentElement.id.split(" ")[1]);
    } else {
        entered_box_n = Number(e.target.parentElement.parentElement.id.split(" ")[1]);
    }
    if ( entered_box_n != null ){
        raphaels[entered_box_n].forEach(function (el) 
        {
            el.attr({"opacity": 1});
        });   
    }

    //e.target.classList.add('drag-over');
}
function dragOver(e) {
    e.preventDefault();
    let entered_box_n = null;
    if ( e.target.nodeName == 'DIV' ){
        entered_box_n = Number(e.target.id.split(" ")[1]);
    } else if ( e.target.nodeName == 'svg' ){
        entered_box_n = Number(e.target.parentElement.id.split(" ")[1]);
    } else {
        entered_box_n = Number(e.target.parentElement.parentElement.id.split(" ")[1]);
    }
    if ( entered_box_n != null ){
        raphaels[entered_box_n].forEach(function (el) 
        {
            el.attr({"opacity": 1});
        });
    }

    //e.target.classList.add('drag-over');
}
function dragLeave(e) {
    //e.target.classList.remove('drag-over');
    e.preventDefault();
    let left_box_n = null;
    if ( e.target.nodeName == 'DIV' ){
        left_box_n = Number(e.target.id.split(" ")[1]);
    } else if ( e.target.nodeName == 'svg' ){
        left_box_n = Number(e.target.parentElement.id.split(" ")[1]);
    } else {
        left_box_n = Number(e.target.parentElement.parentElement.id.split(" ")[1]);
    }
    if ( left_box_n != null ){
        raphaels[left_box_n].forEach(function (el) 
        {
            el.attr({"opacity": 0.3});
        });   
    }
}

// DRAG BEHAVIOR (ADD WHEN BOX IS CREATED)
document.getElementById("box 0").addEventListener('dragstart', dragStart);
document.getElementById("box 0").addEventListener('dragenter', dragEnter)
document.getElementById("box 0").addEventListener('dragover', dragOver);
document.getElementById("box 0").addEventListener('dragleave', dragLeave);

document.getElementById("box 1").addEventListener('dragstart', dragStart);
document.getElementById("box 1").addEventListener('dragenter', dragEnter)
document.getElementById("box 1").addEventListener('dragover', dragOver);
document.getElementById("box 1").addEventListener('dragleave', dragLeave);

document.getElementById("box 2").addEventListener('dragstart', dragStart);
document.getElementById("box 2").addEventListener('dragenter', dragEnter)
document.getElementById("box 2").addEventListener('dragover', dragOver);
document.getElementById("box 2").addEventListener('dragleave', dragLeave);

document.getElementById("box 3").addEventListener('dragstart', dragStart);
document.getElementById("box 3").addEventListener('dragenter', dragEnter)
document.getElementById("box 3").addEventListener('dragover', dragOver);
document.getElementById("box 3").addEventListener('dragleave', dragLeave);

document.getElementById("box 4").addEventListener('dragstart', dragStart);
document.getElementById("box 4").addEventListener('dragenter', dragEnter)
document.getElementById("box 4").addEventListener('dragover', dragOver);
document.getElementById("box 4").addEventListener('dragleave', dragLeave);


// BOX EXCHANGE
// WINDOW SIZE UPDATE
// CLICK ON POINT CLOUD --> CREATE BOX
// BUTTONS
// OSC COMMUNICATION
// HOVER FOR A LONG TIME INCREASES VOLUME AND BIGGER POINTER
// LOGIC WITH LONG AND SHORT TERM PLAY FUNCTIONS
// RENDER 




