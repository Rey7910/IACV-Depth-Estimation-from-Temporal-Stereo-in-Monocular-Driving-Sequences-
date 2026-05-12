# Depth Estimation from Temporal Stereo in Monocular Driving Sequences

This project investigates depth estimation by treating two consecutive monocular images captured by a 
moving vehicle as a virtual stereo pair. Unlike classical stereo vision with fixed, parallel cameras, temporal 
stereo relies on ego-motion to create an effective baseline between frames. The objective is to estimate 
scene depth by exploiting geometric constraints between consecutive views, while analyzing the 
fundamental differences between spatial stereo and motion-induced stereo in autonomous driving 
scenarios. 
In addition to triangulation-based depth recovery, the project explores time-to-impact (TTI) as a geometry
driven tool for depth estimation. TTI provides a depth-related measure that can be inferred directly from 
image velocities and known frame timing. This formulation is particularly valuable in driving scenarios 
characterized by forward motion and small baselines, where classical triangulation becomes ill
conditioned.

This project is being developed with the guide of professors Luca Magri and Carlo Sgaravatti for the course Image Analysis and Computer Vision in charge of professor Vincenzo Caglioti.