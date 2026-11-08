#!/usr/bin/env bash
set -euo pipefail
python experiments/01_open_loop.py
python experiments/02_trim.py
python experiments/03_sensor_validation.py
python experiments/04_ukf_validation.py
python experiments/05_truth_control.py
python experiments/06_estimated_control.py
python experiments/07_fault_gyro_bias.py
python experiments/08_fault_gps_dropout.py
python experiments/09_fault_airspeed_bias.py
python experiments/10_fault_actuator_loe.py
python experiments/13_acceptance_report.py
