// Requiring fs module in which 
// readFile function is defined.
/*const fs = require('fs');

fs.readFile('scatterplot.txt', (err, data) => {
  if (err) throw err;
  console.log(data.toString());
});*/


// Define the addItem() function 
// to be called through onclick 

function addBox(randomcolor) { 
    newBox = document.createElement("div"); 
    newBox.style["background-color"] = randomcolor;
    newBox.style["height"] = '10vh';
    newBox.style["width"] = '60%';
    document.getElementById("compose-bar").appendChild(newBox); 
} 

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

myScatterPlot.on('plotly_click', function(data){
    
    // select a random color
    var randomcolor = '#'+(0x1000000+Math.random()*0xffffff).toString(16).substr(1,6);
    
    // create box with random color
    addBox(randomcolor); 

    // change color of clicked point
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


/*
myScatterPlot.on('plotly_click', function(data){
    var pn='',
        tn='',
        colors=[];
    for(var i=0; i < data.points.length; i++){ //iterate over traces
      pn = data.points[i].pointNumber;
      tn = data.points[i].curveNumber;
      colors = data.points[i].data.marker.color;
    };
    colors[pn] = '#C54C82';
    
    var update = {'marker':{color: colors, size:30}};
    Plotly.restyle('scatterPlot', update, [tn]);
  });


// possible things to do: 
// on click of point in graph: create a box, color the clicked point, listen to how that sounds like
// on click on box: highlight point, listen to how that point sounds like
// on click+drag on box: exchange box position in the timeline with box over which it is dragged
// on 
// how to add arrows?

/*
var myPlot = document.getElementById('scatterPlot'),
    x = [1, 2, 3, 4, 5, 6],
    y = [1, 2, 3, 2, 3, 4],
    colors = ['#00000','#00000','#00000',
              '#00000','#00000','#00000'],
    data = [{x:x, y:y, type:'scatter',
             mode:'markers', marker:{size:35, color:colors}}],
    layout = {
        hovermode:'closest',
        title:'Click on a Point to Change Color<br>Double Click (anywhere) to Change it Back',
        xaxis: {range: [0.5, 6.5]},  // Lines added
        yaxis: {range: [0.5, 4.5]}   // Lines added
     };

Plotly.newPlot('scatterPlot', data, layout);

myPlot.on('plotly_click', function(data){
  var pn='',
      tn='',
      colors=[];
  for(var i=0; i < data.points.length; i++){
    pn = data.points[i].pointNumber;
    tn = data.points[i].curveNumber;
    colors = data.points[i].data.marker.color;
  };
  colors[pn] = '#C54C82';
    
  var update = {'marker':{color: colors, size:30}};
  Plotly.restyle('scatterPlot', update, [tn]);
});

myPlot.on('plotly_doubleclick', function(){
  var orgColors = ['#00000','#00000','#00000',
                   '#00000','#00000','#00000'];
  var update = {'marker':{color: orgColors, size:16}};
  Plotly.restyle('scatterPlot', update);
});*/