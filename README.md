# traffic_signs_classification
Detection and classification of traffic signs in lgsvl simulator.


### Simulation section
To open simulator run the following command:
```
CarlaUE4.exe -windowed -ResX=640 -ResY=640 -quality=low
```
To load testing maps into the simulator use config.py from simulation directory. It's script directly from
carla simulator package. The following command is used to load a map from .xodr file:
```
python3 config.py -x simulation/config.xodr 
```