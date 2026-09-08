#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards
Servo servoHor;
Servo servoVer;
int posHor = 0;    // variable to store the servo position
int posVer = 0;

void setup() {
  Serial.begin(9600); // Start serial communication at 9600 baud
  servoVer.attach(6);
  startupSequence();
}

void loop(){
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    int sep = command.indexOf(',');
    posHor = command.substring(0, sep).toInt();
    posVer = command.substring(sep+1).toInt();
    if (posHor == 360 and posVer == 360){
      endingSequence();
    }
    else{
      // posHor = int(map(posHor, -320, 320, 51.5, 128.5));
      // posVer = int(map(posVer, -240, 240, 125.2, 54.8));
      posHor = int(map(posHor, -320, 320, 58.5, 131.5));
      posVer = int(map(posVer, -240, 240, 124.2, 53.8));
      servoHor.write(posHor);
      servoVer.write(posVer);
      }
    }
  }

void startupSequence(){
servoHor.write(90);
servoVer.write(90);
delay(1000);
servoHor.write(150);
servoVer.write(150);
delay(1000);
servoVer.write(90);
servoHor.write(90);
delay(1000);
servoVer.write(115);
delay(100);
servoVer.write(75);
delay(100);
servoVer.write(115);
delay(100);
servoVer.write(75);
delay(100);
servoVer.write(90);
}

void endingSequence(){
  servoHor.write(0);
  servoVer.write(150);
}

