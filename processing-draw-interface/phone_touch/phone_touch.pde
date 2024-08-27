import oscP5.*;
import netP5.*;

OscP5 oscP5;
NetAddress myRemoteLocation;


void setup() {

  fullScreen();
  background(0);
  strokeWeight(16);  // Thicker
  stroke(153);
  oscP5 = new OscP5(this,12000);
  //myRemoteLocation = new NetAddress("127.0.0.1",12000);  
  myRemoteLocation = new NetAddress("192.168.1.17",12000);
  
}

void draw() {
  stroke(255);
  if (mousePressed == true) {
    line(mouseX, mouseY, pmouseX, pmouseY);
    println(float(mouseX)/width, float(mouseY)/height, 
    float(pmouseX)/width, float(pmouseY)/height);
      OscMessage myMessage = new OscMessage("/test");
      myMessage.add(float(mouseX)/width); 
      myMessage.add(float(mouseY)/height); 
      myMessage.add(float(pmouseX)/width); 
      myMessage.add(float(pmouseY)/height); 
      
      /* send the message */
      oscP5.send(myMessage, myRemoteLocation); 
  }
  else{
      OscMessage myMessage = new OscMessage("/test");
      myMessage.add(0.0); 
      myMessage.add(1.0); 
      myMessage.add(0.0); 
      myMessage.add(1.0); 
      
      /* send the message */
      oscP5.send(myMessage, myRemoteLocation); 
  }
}
