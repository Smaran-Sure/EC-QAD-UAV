import numpy as np
from dataclasses import replace
from ecqad_uav.sensors.sensor_suite import SensorSuite
from ecqad_uav.faults.fault_injector import FaultState
from ecqad_uav.plant.jsbsim_plant import JSBSimPlant
from ecqad_uav.utils.config import load_yaml


def test_sensor_rates_and_noise_statistics():
    cfg=load_yaml('sensors.yaml'); sensor=SensorSuite(cfg,200)
    plant=JSBSimPlant(load_yaml('aircraft.yaml')); plant.initialize()
    truth=plant.read_truth(); gyro=[]; accel=[]; gps=[]
    for i in range(20000):
        packet=sensor.sample(replace(truth,t=i/200),FaultState())
        if packet.imu is not None:
            gyro.append(packet.imu.gyro_rad_s-sensor.gyro_bias0)
            accel.append(packet.imu.accel_mps2-sensor.accel_bias0)
        if packet.gps is not None: gps.append(packet.gps.position_ned_m)
    assert len(gyro)==10000 and len(gps)==500
    assert np.allclose(np.std(gyro,axis=0),sensor.gyro_noise,rtol=.05)
    assert np.allclose(np.std(accel,axis=0),sensor.accel_noise,rtol=.05)
    assert np.allclose(np.std(gps,axis=0),sensor.gps_pos_noise,rtol=.12)
    assert np.max(abs(np.mean(gyro,axis=0)))<sensor.gyro_noise*.05
