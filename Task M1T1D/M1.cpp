/*
  FINAL DISTINCTION VERSION (WITHOUT LDR LED)
  -------------------------------------------
  Uses:
  - External Interrupt (INT0) for PIR
  - Pin Change Interrupt (PCINT) for Button
  - Timer1 Compare Interrupt (1 second periodic)
  - Ultrasonic scheduled in loop
  - LDR timer-driven sampling (NO LED)
  - Dedicated LED for PIR and Ultrasonic
*/

#define PIR_PIN         2
#define BUTTON_PIN      8

#define TRIG_PIN        6
#define ECHO_PIN        7

#define LED_ALARM       3
#define LED_PIR         4
#define LED_ULTRASONIC  5

#define BUZZER_PIN      10
#define LDR_PIN         A0

// ---------------- Interrupt Flags ----------------
volatile bool pirFlag = false;
volatile bool buttonFlag = false;
volatile bool timerFlag = false;
volatile uint8_t lastPortBState = 0;

// ---------------- Runtime Variables ----------------
unsigned long lastUltrasonicRead = 0;
const unsigned long ultrasonicInterval = 250;

unsigned long pirLastSeen = 0;
const unsigned long pirActiveWindow = 5000;

unsigned long buttonPressedSince = 0;
const unsigned long buttonRecentWindow = 4000;

unsigned long lastButtonHandled = 0;
const unsigned long debounceMs = 50;

bool objectClose = false;
float lastDistance = -1;

int LDR_THRESHOLD = 300;
bool alarmActive = false;
bool userToggleState = false;

// ---------------- SETUP ----------------
void setup() {

  Serial.begin(9600);

  pinMode(PIR_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(LED_ALARM, OUTPUT);
  pinMode(LED_PIR, OUTPUT);
  pinMode(LED_ULTRASONIC, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(LED_ALARM, LOW);
  digitalWrite(LED_PIR, LOW);
  digitalWrite(LED_ULTRASONIC, LOW);

  attachInterrupt(digitalPinToInterrupt(PIR_PIN), pirISR, RISING);

  setupPCINT();
  setupTimer1();

  Serial.println("=== FINAL DISTINCTION SYSTEM (NO LDR LED) ===");
}

// ---------------- External Interrupt ISR ----------------
void pirISR() {
  pirFlag = true;
}

// ---------------- PCINT Setup ----------------
void setupPCINT() {
  cli();
  PCICR |= (1 << PCIE0);
  PCMSK0 |= (1 << PCINT0);   // D8
  lastPortBState = PINB;
  sei();
}

// ---------------- PCINT ISR ----------------
ISR(PCINT0_vect) {

  uint8_t currentState = PINB;
  uint8_t changed = currentState ^ lastPortBState;

  if (changed & (1 << PB0)) {
    if (!(currentState & (1 << PB0))) {
      buttonFlag = true;
    }
  }

  lastPortBState = currentState;
}

// ---------------- Timer1 Setup ----------------
void setupTimer1() {

  cli();

  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;

  OCR1A = 15624;

  TCCR1B |= (1 << WGM12);
  TCCR1B |= (1 << CS12) | (1 << CS10);

  TIMSK1 |= (1 << OCIE1A);

  sei();
}

// ---------------- Timer ISR ----------------
ISR(TIMER1_COMPA_vect) {
  timerFlag = true;
}

// ---------------- MAIN LOOP ----------------
void loop() {

  handlePIR();
  handleButton();
  handleTimer();
  handleUltrasonic();
  evaluateAlarm();

  delay(1);
}

// ---------------- PIR Handling ----------------
void handlePIR() {

  if (pirFlag) {

    pirFlag = false;
    pirLastSeen = millis();

    digitalWrite(LED_PIR, HIGH);
    tone(BUZZER_PIN, 900, 120);

    Serial.println("[EVENT] Motion detected (External INT)");
  }

  if (millis() - pirLastSeen < pirActiveWindow) {
    digitalWrite(LED_PIR, HIGH);
  } else {
    digitalWrite(LED_PIR, LOW);
  }
}

// ---------------- Button Handling ----------------
void handleButton() {

  if (!buttonFlag) return;

  unsigned long now = millis();
  buttonFlag = false;

  if (now - lastButtonHandled < debounceMs) return;
  lastButtonHandled = now;

  buttonPressedSince = now;
  userToggleState = !userToggleState;

  tone(BUZZER_PIN, 1000, 80);

  Serial.println("[EVENT] Button Pressed (PCINT)");
}

// ---------------- Timer Handling ----------------
void handleTimer() {

  if (!timerFlag) return;

  timerFlag = false;

  int ldrValue = analogRead(LDR_PIN);

  Serial.print("[TIMER] LDR: ");
  Serial.println(ldrValue);

  if (ldrValue < LDR_THRESHOLD) {
    tone(BUZZER_PIN, 800, 150);
    Serial.println("[ACTION] Dark detected");
  }
}

// ---------------- Ultrasonic ----------------
void handleUltrasonic() {

  if (millis() - lastUltrasonicRead < ultrasonicInterval) return;

  lastUltrasonicRead = millis();

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 20000);

  if (duration == 0) {
    objectClose = false;
    digitalWrite(LED_ULTRASONIC, LOW);
    return;
  }

  lastDistance = (duration * 0.034) / 2;

  Serial.print("[ULTRA] Distance: ");
  Serial.println(lastDistance);

  if (lastDistance < 20 && lastDistance > 0) {
    objectClose = true;
    digitalWrite(LED_ULTRASONIC, HIGH);
    tone(BUZZER_PIN, 1200, 80);
  } else {
    objectClose = false;
    digitalWrite(LED_ULTRASONIC, LOW);
  }
}

// ---------------- Alarm Logic ----------------
void evaluateAlarm() {

  bool pirActive = (millis() - pirLastSeen < pirActiveWindow);
  bool buttonRecent = (millis() - buttonPressedSince < buttonRecentWindow);

  bool newAlarm = pirActive && (buttonRecent || objectClose);

  if (newAlarm && !alarmActive) {
    Serial.println("[ALARM] ACTIVATED");
    tone(BUZZER_PIN, 1500, 200);
  }

  if (!newAlarm && alarmActive) {
    Serial.println("[ALARM] CLEARED");
  }

  alarmActive = newAlarm;

  if (alarmActive)
    digitalWrite(LED_ALARM, HIGH);
  else
    digitalWrite(LED_ALARM, userToggleState ? HIGH : LOW);
}
