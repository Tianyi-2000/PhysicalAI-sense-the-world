#include <Modulino.h>
#include <Arduino_RouterBridge.h>

ModulinoDistance distance;
int last_mm = -1;

int get_distance() { return last_mm; }

void setup() {
  Bridge.begin();
  Modulino.begin();          // if no readings, change to Modulino.begin(Wire1);
  distance.begin();
  Bridge.provide("get_distance", get_distance);
}

void loop() {
  if (distance.available()) {
    last_mm = distance.get();
    Monitor.println(last_mm);
  }
}