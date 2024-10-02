let numBoxes = 0;

document.getElementById("insert-crossfade").onclick = function(){
    console.log("Here")
    let newBox = document.createElement("div");
        newBox.className = 'box';
        newBox.id = 'box '+numBoxes;
        newBox.style["background-color"] = "rgba(255, 0, 0, 0.)";
        newBox.style["height"] = '5vh';
        newBox.style["width"] = '100%';
    newCircle(newBox.id);
    document.getElementById("composition-timeline").appendChild(newBox);
    numBoxes += 1;
};


// window.height (px) = 2 min
// when window changes change all multipliers
// diameter of circle / height of arrow are = to diameter / window.height [min]
var R = Raphael("box 1", 200, 120);
var s = R.circle(100, 60, 50).attr({
        fill: "hsb(.8, .5, .5)",
        stroke: "none",
        opacity: .8
    });
var c = R.circle(100, 60, 50).attr({
        fill: "none",
        stroke: "hsb(.8, 1, 1)",
        "stroke-width": 6,
        opacity: 0.05
    });

// window.innerHeight = 514 = 2 min = 120 s
// a radius of 55 means a diameter of 110 --> 110 / 514 = 0.214. multiplied by 120 --> 25.16 s
const MAX_R = 55; // corresponds to a duration 25.16 s
const MIN_R = 5; // circle_duration[s] = (r*2 / window.innerHeight) * 120 
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
    if ( newr < MAX_R && newr > MIN_R ) {
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
c.sized = s;
c.raph = R;
c.parentDiv = document.getElementById("box 1");
c.drag(move, start, up);  

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

s.outer = c;
c.hover(hoverIn_outer, hoverOut_outer);
s.hover(hoverIn, hoverOut);



var R2 = Raphael("box 2", 200, 120);
var s2 = R2.circle(100, 60, 50).attr({
        fill: "hsb(.1, .5, .5)",
        stroke: "none",
        opacity: .8
    });
var c2 = R2.circle(100, 60, 50).attr({
        fill: "none",
        stroke: "hsb(.1, 1, 1)",
        "stroke-width": 6,
        opacity: 0.05
    });
c2.sized = s2;
c2.parentDiv = document.getElementById("box 2");
c2.raph = R2;
c2.drag(move, start, up);
s2.outer = c2;
c2.hover(hoverIn_outer, hoverOut_outer);
s2.hover(hoverIn, hoverOut);


// from https://jsfiddle.net/TfE2X/
var R3 = Raphael("box 3", 200, 120);
path = R3.path("M100 0L100 110").attr({
    stroke: '#FFFFFF',
    'stroke-width': 2,
    'arrow-end':'classic-wide-long'
});
var pathArray = path.attr("path");
handle = R3.circle(100,110,10).attr({
    fill: "#FFFFFF",
    cursor: "pointer",
    "stroke-width": 10,
    stroke: "transparent",
    opacity: 0.3
});

var start_arrow = function () {
    this.cy = this.attr("cy");
    this.attr({opacity: 1});
},
move_arrow = function (dx, dy) {
    var Y = this.cy + dy;
    console.log(Y);
    if ( Y < MAX_R*2 && Y > MIN_R*2 ){
        this.attr({
            cy: Y});
        pathArray[1][2] = Y;
        path.attr({path: pathArray});
        this.parentDiv.style["height"] = Y+20;
        this.raph.setSize(200, Y+20);
    }
},
up_arrow = function () {
    this.attr({opacity: 0.3});
};
handle.parentDiv = document.getElementById("box 3");
handle.raph = R3;
handle.drag(move_arrow, start_arrow, up_arrow);

var R4 = Raphael("box 4", 200, 120);
var s4 = R4.circle(100, 60, 50).attr({
        fill: "hsb(.3, .5, .5)",
        stroke: "none",
        opacity: .8
    });
var c4 = R4.circle(100, 60, 50).attr({
        fill: "none",
        stroke: "hsb(.3, 1, 1)",
        "stroke-width": 6,
        opacity: 0.05
    });
c4.sized = s4;
c4.parentDiv = document.getElementById("box 4");
c4.raph = R4;
c4.drag(move, start, up);
s4.outer = c4;
c4.hover(hoverIn_outer, hoverOut_outer);
s4.hover(hoverIn, hoverOut);


var R5 = Raphael("box 5", 200, 120);
path1 = R5.path("M100 0L120 30").attr({
    stroke: '#FFFFFF',
    'stroke-width': 2
});
path2 = R5.path("M120 30L70 60").attr({
    stroke: '#FFFFFF',
    'stroke-width': 2
});
path3 = R5.path("M70 60L100 90").attr({
    stroke: '#FFFFFF',
    'stroke-width': 2
});
path4 = R5.path("M100 90L100 120").attr({
    stroke: '#FFFFFF',
    'stroke-width': 2,
    'arrow-end':'classic-wide-long'
});

var pathArray1 = path1.attr("path");
console.log(pathArray1)
var pathArray2 = path2.attr("path");
var pathArray3 = path3.attr("path");
var pathArray4 = path4.attr("path");
handle_meander = R5.circle(100,110,10).attr({
    fill: "#FFFFFF",
    cursor: "pointer",
    "stroke-width": 10,
    stroke: "transparent",
    opacity: 0.3
});


var start_meander = function () {
    this.cy = this.attr("cy");
    this.attr({opacity: 1});
},
move_meander = function (dx, dy) {
    var Y = this.cy + dy;
    console.log(Y);
    if ( Y < MAX_R*2 && Y > MIN_R*2 ){
        this.attr({ cy: Y });
        pathArray1[1][2] = Y/4;
        pathArray2[0][2] = Y/4;
        pathArray2[1][2] = Y/4*2;
        pathArray3[0][2] = Y/4*2;
        pathArray3[1][2] = Y/4*3;
        pathArray4[0][2] = Y/4*3;
        pathArray4[1][2] = Y;
        path1.attr({path: pathArray1});
        path2.attr({path: pathArray2});
        path3.attr({path: pathArray3});
        path4.attr({path: pathArray4});
        this.parentDiv.style["height"] = Y+20;
        this.raph.setSize(200, Y+20);
    }
},
up_meander = function () {
    this.attr({opacity: 0.3});
};
handle_meander.parentDiv = document.getElementById("box 5");
handle_meander.raph = R5;
handle_meander.drag(move_meander, start_meander, up_meander);

var R6 = Raphael("box 6", 200, 120);
var s6 = R6.circle(100, 60, 50).attr({
        fill: "hsb(.95, .5, .5)",
        stroke: "none",
        opacity: .8
    });
var c6 = R6.circle(100, 60, 50).attr({
        fill: "none",
        stroke: "hsb(.95, 1, 1)",
        "stroke-width": 6,
        opacity: 0.05
    });
c6.sized = s6;
c6.parentDiv = document.getElementById("box 6");
c6.raph = R6;
c6.drag(move, start, up);
s6.outer = c6;
c6.hover(hoverIn_outer, hoverOut_outer);
s6.hover(hoverIn, hoverOut);


var R_timeline = Raphael("timeline", 30, window.innerHeight - (90 + 60 + 20) );
path_timeline = R_timeline.path("M15 0L15 "+(window.innerHeight - (90 + 60 +50))).attr({
    stroke: '#FFFFFF',
    'stroke-width': 1.5,
    'arrow-end':'classic-wide-long',
    opacity: 0.5
});



