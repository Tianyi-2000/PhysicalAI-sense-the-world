// MCU side of Sense the World. Linux calls these over Arduino Bridge:
//   show_thinking()                -> scanning bar on the LED matrix, RGB LED blue
//   show_result(label, confidence) -> scroll the label; RGB LED green (sure) or yellow (unsure)
//   show_error()                   -> clear the matrix, RGB LED red
//   set_led_state(state)           -> built-in LED (the original LED demo)
// The MCU owns the limits: it cleans the label, clamps the confidence, and picks the colors.
// Handlers use provide_safe because they touch GPIO. They only update state; loop() does the
// drawing, so scrolling a long label never blocks the Bridge.
#include <Arduino_RouterBridge.h>
#include <ArduinoGraphics.h>
#include <Arduino_LED_Matrix.h>

const int MATRIX_WIDTH = 13;
const unsigned int MAX_LABEL = 16;
const int SURE_PERCENT = 30;  // same threshold as CONFIDENT in web/src/components.tsx
const unsigned long SCROLL_MS = 70;
const unsigned long SCAN_MS = 60;

enum Mode { IDLE, THINKING, RESULT };

Arduino_LED_Matrix matrix;
Mode mode = IDLE;
String label;
int scrollX = 0;
int scanX = 0;
int scanStep = 1;
unsigned long lastFrame = 0;

void setRgb(bool red, bool green, bool blue) {
  // Active low
  digitalWrite(LED3_R, red ? LOW : HIGH);
  digitalWrite(LED3_G, green ? LOW : HIGH);
  digitalWrite(LED3_B, blue ? LOW : HIGH);
}

void set_led_state(bool state) {
  digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
}

void show_thinking() {
  mode = THINKING;
  scanX = 0;
  scanStep = 1;
  setRgb(false, false, true);
}

void show_result(String text, int confidence) {
  label = "";
  for (unsigned int i = 0; i < text.length() && label.length() < MAX_LABEL; i++) {
    char c = toupper(text[i]);
    if (c == ',') break;  // "tabby, tabby cat" -> "TABBY"
    if (isalnum(c) || c == ' ' || c == '-') label += c;
  }
  bool sure = constrain(confidence, 0, 100) >= SURE_PERCENT;
  setRgb(!sure, true, false);  // green, or yellow (red + green) when unsure
  scrollX = MATRIX_WIDTH;      // start just off the right edge
  mode = RESULT;
}

void show_error() {
  mode = IDLE;
  matrix.clear();
  setRgb(true, false, false);
}

void drawScan() {
  matrix.beginDraw();
  matrix.ArduinoGraphics::clear();
  for (int y = 0; y < 8; y++) {
    matrix.set(scanX, y, 1, 1, 1);
  }
  matrix.endDraw();
  scanX += scanStep;
  if (scanX == 0 || scanX == MATRIX_WIDTH - 1) scanStep = -scanStep;
}

void drawLabel() {
  matrix.beginDraw();
  matrix.ArduinoGraphics::clear();
  matrix.text(label, scrollX, 1);
  matrix.endDraw();
  int width = label.length() * matrix.textFontWidth();
  if (--scrollX < -width) scrollX = MATRIX_WIDTH;  // keep scrolling until the next photo
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  pinMode(LED3_R, OUTPUT);
  pinMode(LED3_G, OUTPUT);
  pinMode(LED3_B, OUTPUT);
  setRgb(false, false, false);

  matrix.begin();
  matrix.textFont(Font_5x7);
  matrix.stroke(0xFFFFFFFF);  // text() draws nothing without a stroke color
  matrix.clear();

  if (!Bridge.begin()) {
    while (true) {}
  }
  Bridge.provide_safe("set_led_state", set_led_state);
  Bridge.provide_safe("show_thinking", show_thinking);
  Bridge.provide_safe("show_result", show_result);
  Bridge.provide_safe("show_error", show_error);
}

void loop() {
  unsigned long now = millis();
  if (mode == THINKING && now - lastFrame >= SCAN_MS) {
    lastFrame = now;
    drawScan();
  } else if (mode == RESULT && now - lastFrame >= SCROLL_MS) {
    lastFrame = now;
    drawLabel();
  }
}
