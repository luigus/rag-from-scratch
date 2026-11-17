# Fault Code Reference (Aurora class 450)

*Fictional operator: Atlantic Rail. All data in this corpus is invented for testing.*

The first digit of a fault code shows the system: F-1xx train communication, F-2xx doors, F-3xx pantograph and power supply, F-4xx air systems, F-5xx passenger comfort. Many codes differ only in the order of their digits, so read them carefully.

## Communication

- **F-124**: passenger alarm activated. The driver must stop at the next station and check the car shown on the display.
- **F-127**: train line communication timeout between two coupled units. Uncouple and couple again at the next stop.
- **F-172**: battery voltage low. Keep the pantograph raised so the batteries can recharge.

## Doors

- **F-271**: door motor overcurrent. The door motor DOR-2710 must be checked at the next depot visit.
- **F-712** is not a valid code: if it appears, restart the train management system.

## Pantograph and power

- **F-350**: pantograph position sensor fault. The train may continue with the pantograph raised, but the fault must be reported to the depot.
- **F-503** is listed in the comfort section below, not here.

## Air systems

- **F-421**: main air compressor start failure. The second compressor takes over automatically.

## Passenger comfort

- **F-503**: HVAC filter clogged. Replace filter HVC-4120 at the next A-check.
- **F-530**: cabin temperature sensor fault. The air conditioning keeps running with the last valid temperature.
