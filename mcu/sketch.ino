// MCU side of the LED demo: Linux calls "set_led_state" over Arduino Bridge,
// and this sketch toggles the built-in LED (active-low).
// Registered with provide_safe (not provide) because the callback touches GPIO.
#include <Arduino_RouterBridge.h>

void set_led_state(bool state) {
  digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  if (!Bridge.begin()) {
    while (true) {}
  }
  Bridge.provide_safe("set_led_state", set_led_state);
}

void loop() {}
