// MCU side of Sense the World. Linux calls these over Arduino Bridge:
//   show_thinking()                -> scanning bar on the LED matrix, RGB LED blue
//   show_result(label, confidence) -> scroll the label; RGB LED green (sure) or yellow (unsure)
//   show_error()                   -> scroll "ERROR", RGB LED red
//   show_frame(bits)               -> draw a 13x8 picture: 104 '0'/'1' chars, row by row
//   show_distance(mm)              -> show a distance in cm on the LED matrix (up to 999)
//   set_led(r, g, b)               -> RGB LED, 0-255 per channel; on/off only, on at >= 128
//   get_distance()                 -> latest Modulino Distance reading in mm, -1 before the first
//   set_led_state(state)           -> built-in LED (the original LED demo)
// With nothing to show (at boot, 20 s after a result, or 2 s after the last distance), the
// matrix shows a blinking eye.
// The MCU owns the limits: it cleans the label, clamps the confidence, and picks the colors.
// Handlers that touch GPIO or the matrix use provide_safe. Anything animated is drawn in loop(),
// so scrolling a long label never blocks the Bridge. get_distance only returns a number, so it
// uses provide and answers without waiting for loop().
#include <Arduino_RouterBridge.h>
#include <ArduinoGraphics.h>
#include <Arduino_LED_Matrix.h>
#include <Modulino.h>

const int MATRIX_WIDTH = 13;
const int MATRIX_HEIGHT = 8;
const unsigned int MAX_LABEL = 16;
const int SURE_PERCENT = 30;  // same threshold as CONFIDENT in web/src/components.tsx
const unsigned long SCROLL_MS = 70;
const unsigned long SCAN_MS = 60;
const unsigned long EYE_MS = 80;
const unsigned long IDLE_AFTER_MS = 20000;  // back to the eye after a result or a stuck "thinking"
const unsigned long DISTANCE_STALE_MS = 2000;  // back to the eye when the live demo stops sending
const unsigned long BLINK_EVERY_MS = 4000;

// Idle eye. '#' is on. The pupil is drawn on top, see drawEye().
const char* EYE_OPEN[MATRIX_HEIGHT] = {
  ".............",
  "....#####....",
  "..##.....##..",
  ".#.........#.",
  ".#.........#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const char* EYE_HALF[MATRIX_HEIGHT] = {
  ".............",
  ".............",
  ".............",
  ".###########.",
  ".#.........#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const char* EYE_CLOSED[MATRIX_HEIGHT] = {
  ".............",
  ".............",
  ".............",
  ".............",
  ".#.........#.",
  "..##.....##..",
  "....#####....",
  ".............",
};
const int GAZE[] = {0, -3, 0, 3};  // pupil shift per blink cycle: center, left, center, right

enum Mode { IDLE, THINKING, RESULT, FRAME, DISTANCE };

Arduino_LED_Matrix matrix;
ModulinoDistance distanceSensor;
int sensorMm = -1;  // written in loop(), read by get_distance()
Mode mode = IDLE;
String label;
int scrollX = 0;
int scanX = 0;
int scanStep = 1;
unsigned long lastFrame = 0;
unsigned long lastCommand = 0;

void setRgb(bool red, bool green, bool blue) {
  // Active low
  digitalWrite(LED3_R, red ? LOW : HIGH);
  digitalWrite(LED3_G, green ? LOW : HIGH);
  digitalWrite(LED3_B, blue ? LOW : HIGH);
}

void startScroll(const String& text) {
  matrix.textFont(Font_5x7);
  label = text;
  scrollX = MATRIX_WIDTH;  // start just off the right edge
  mode = RESULT;
  lastCommand = millis();
}

void set_led_state(bool state) {
  digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
}

void show_thinking() {
  mode = THINKING;
  scanX = 0;
  scanStep = 1;
  lastCommand = millis();
  setRgb(false, false, true);
}

void show_result(String text, int confidence) {
  String clean;
  for (unsigned int i = 0; i < text.length() && clean.length() < MAX_LABEL; i++) {
    char c = toupper(text[i]);
    if (c == ',') break;  // "tabby, tabby cat" -> "TABBY"
    if (isalnum(c) || c == ' ' || c == '-') clean += c;
  }
  bool sure = constrain(confidence, 0, 100) >= SURE_PERCENT;
  setRgb(!sure, true, false);  // green, or yellow (red + green) when unsure
  startScroll(clean);
}

void show_error() {
  setRgb(true, false, false);
  startScroll("ERROR");
}

void show_frame(String bits) {
  mode = FRAME;  // stays until the next command
  matrix.beginDraw();
  matrix.ArduinoGraphics::clear();
  for (unsigned int i = 0; i < bits.length() && i < MATRIX_WIDTH * MATRIX_HEIGHT; i++) {
    if (bits[i] == '1') matrix.set(i % MATRIX_WIDTH, i / MATRIX_WIDTH, 1, 1, 1);
  }
  matrix.endDraw();
}

void show_distance(int mm) {
  mode = DISTANCE;  // back to the eye once updates stop, see loop()
  lastCommand = millis();
  String cm((constrain(mm, 0, 9994) + 5) / 10);
  matrix.textFont(Font_4x6);  // digits are 3 px + 1 px gap, so "999" fits in 13 columns
  matrix.beginDraw();
  matrix.ArduinoGraphics::clear();
  matrix.text(cm, (MATRIX_WIDTH - (int)cm.length() * matrix.textFontWidth() + 1) / 2, 1);
  matrix.endDraw();
}

void set_led(int red, int green, int blue) {
  setRgb(red >= 128, green >= 128, blue >= 128);  // amber (255, 160, 0) shows as yellow
}

int get_distance() {
  return sensorMm;
}

void drawRows(const char* rows[MATRIX_HEIGHT]) {
  for (int y = 0; y < MATRIX_HEIGHT; y++) {
    for (int x = 0; x < MATRIX_WIDTH; x++) {
      if (rows[y][x] == '#') matrix.set(x, y, 1, 1, 1);
    }
  }
}

void drawPupil(int left, int top, int bottom) {
  for (int x = left; x < left + 3; x++) {
    for (int y = top; y <= bottom; y++) matrix.set(x, y, 1, 1, 1);
  }
}

void drawEye() {
  unsigned long now = millis();
  // The blink is the last 320 ms of each cycle, one 80 ms frame each: half, shut, half, open.
  long blink = (long)(now % BLINK_EVERY_MS) - (long)(BLINK_EVERY_MS - 320);
  int pupilX = 5 + GAZE[(now / BLINK_EVERY_MS) % 4];  // 3 px wide, so 2..8 stays inside the eye

  matrix.beginDraw();
  matrix.ArduinoGraphics::clear();
  if (blink >= 80 && blink < 160) {
    drawRows(EYE_CLOSED);
  } else if (blink >= 0 && blink < 240) {
    drawRows(EYE_HALF);
    drawPupil(pupilX, 4, 4);
  } else {
    drawRows(EYE_OPEN);
    drawPupil(pupilX, 3, 4);
  }
  matrix.endDraw();
}

void drawScan() {
  matrix.beginDraw();
  matrix.ArduinoGraphics::clear();
  for (int y = 0; y < MATRIX_HEIGHT; y++) {
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
  matrix.stroke(0xFFFFFFFF);  // text() draws nothing without a stroke color
  matrix.clear();

  if (!Bridge.begin()) {
    while (true) {}
  }
  Modulino.begin();  // Qwiic. If get_distance stays -1, try Modulino.begin(Wire1)
  distanceSensor.begin();  // without a sensor, get_distance just returns -1
  Bridge.provide_safe("set_led_state", set_led_state);
  Bridge.provide_safe("show_thinking", show_thinking);
  Bridge.provide_safe("show_result", show_result);
  Bridge.provide_safe("show_error", show_error);
  Bridge.provide_safe("show_frame", show_frame);
  Bridge.provide_safe("show_distance", show_distance);
  Bridge.provide_safe("set_led", set_led);
  Bridge.provide("get_distance", get_distance);
}

void loop() {
  unsigned long now = millis();
  if (distanceSensor.available()) {
    sensorMm = distanceSensor.get();
  }

  bool stale = (mode == RESULT || mode == THINKING) && now - lastCommand >= IDLE_AFTER_MS;
  stale = stale || (mode == DISTANCE && now - lastCommand >= DISTANCE_STALE_MS);
  if (stale) {
    mode = IDLE;
    setRgb(false, false, false);
  }

  if (mode == IDLE && now - lastFrame >= EYE_MS) {
    lastFrame = now;
    drawEye();
  } else if (mode == THINKING && now - lastFrame >= SCAN_MS) {
    lastFrame = now;
    drawScan();
  } else if (mode == RESULT && now - lastFrame >= SCROLL_MS) {
    lastFrame = now;
    drawLabel();
  }
}
