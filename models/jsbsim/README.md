# Rascal model provenance

Source: https://github.com/ArduPilot/ardupilot/tree/2a54162249a9b02845d5b62920d29d5edaece82e/Tools/autotest/aircraft/Rascal

Vendored unchanged from that commit:

- `Rascal.xml` → `aircraft/Rascal110/Rascal110.xml` (JSBSim dynamics, not the visual `Models/Rascal110.xml`).
- `Engines/Zenoah_G-26A.xml` → `engine/Zenoah_G-26A.xml`.
- `Engines/18x8.xml` → `engine/18x8.xml`.

The local directory/XML naming exposes the airframe as `Rascal110`.
Upstream identifies this dynamics model as BETA; loading it does not establish
research validation or successful trim. Its engine uses JSBSim's electric-engine
representation. The upstream TCP input on port 5505 is retained; restricted
execution environments may report a nonfatal socket-bind error.
