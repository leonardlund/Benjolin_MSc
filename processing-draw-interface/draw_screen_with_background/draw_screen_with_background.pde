import oscP5.*;
import netP5.*;

OscP5 oscP5;
NetAddress myRemoteLocation;

PImage bg;

int rectX, rectY;      // Position of square button
int rectSize = 100;     // Diameter of square
color rectColor;
color rectHighlight;
boolean rectOver = false;

void setup() {

  //fullScreen();
  //background(0);
  size(850, 944); //size of the image in pixels + space for the button (on y)
  bg = loadImage("latent_space2.png");
  image(bg, 0, rectSize);

  // button properties
  rectColor = color(10);
  rectHighlight = color(51);
  rectX = width-rectSize;
  rectY = 0;

  oscP5 = new OscP5(this,12000);
  //myRemoteLocation = new NetAddress("127.0.0.1",12000);  
  myRemoteLocation = new NetAddress("193.157.200.67",12000);
  
}

void draw() {
  

  update(mouseX, mouseY); // check if mouse is on the button
  
  if (rectOver) {
    fill(rectHighlight);
  } else {
    fill(rectColor);
  }
  stroke(255);
  rect(rectX, rectY, rectSize, rectSize);
  
  stroke(200, 25, 25);  
  strokeWeight(10);  // Thicker

  if ((mousePressed == true) && (pmouseY > rectSize)) {
    line(mouseX, mouseY, pmouseX, pmouseY);
    println(float(mouseX)/width, float(mouseY)/height, 
    float(pmouseX)/width, float(pmouseY)/height);
      OscMessage myMessage = new OscMessage("/parameters");
      //myMessage.add(float(mouseX)/(width)); 
      //myMessage.add((float(mouseY) - rectSize)/(height - rectSize)); 
      myMessage.add((float(pmouseY) - rectSize)/(height - rectSize)); 
      myMessage.add(float(pmouseX)/width); 
      oscP5.send(myMessage, myRemoteLocation); // send the message
  }
  else{
      OscMessage myMessage = new OscMessage("/parameters");
      myMessage.add(0.0); 
      myMessage.add(1.0); 
      myMessage.add(0.0); 
      myMessage.add(1.0); 
      
      /* send the message */
      oscP5.send(myMessage, myRemoteLocation); 
  }
}

void update(int x, int y) {
  if ( overRect(rectX, rectY, rectSize, rectSize) ) {
    rectOver = true;
  } else {
    rectOver = false;
  }
}

void mousePressed() {
  if (rectOver) {
    //reset drawing
    image(bg, 0, rectSize);
    
    //send reset to PD
    OscMessage myMessage = new OscMessage("/reset");
    myMessage.add(1.0); 
    oscP5.send(myMessage, myRemoteLocation); //send message
  }
}

boolean overRect(int x, int y, int width, int height)  {
  if (mouseX >= x && mouseX <= x+width && 
      mouseY >= y && mouseY <= y+height) {
    return true;
  } else {
    return false;
  }
}
