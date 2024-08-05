import oscP5.*;
import netP5.*;

OscP5 oscP5;
NetAddress myRemoteLocation;

PImage bg;

void setup() {
  //size(600, 600);
  size(489, 363);
  bg = loadImage("sample_visualization.png");
  background(bg);
  strokeWeight(10);
  //background(100);
  oscP5 = new OscP5(this,12000);
  myRemoteLocation = new NetAddress("127.0.0.1",12000);
}

void draw() {
  stroke(200, 25, 25);
  if (mousePressed == true) {
    line(mouseX, mouseY, pmouseX, pmouseY);
    println(float(mouseX)/600, float(mouseY)/600, 
    float(pmouseX)/600, float(pmouseY)/600);
      OscMessage myMessage = new OscMessage("/test");
      myMessage.add(float(mouseX)/600); 
      myMessage.add(float(mouseY)/600); 
      myMessage.add(float(pmouseX)/600); 
      myMessage.add(float(pmouseY)/600); 
    
      /* send the message */
      oscP5.send(myMessage, myRemoteLocation); 
  }
}
