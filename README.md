![hrc-hub-overview](posetestbot.png)

# HRC-Hub
Software for the Human-Robot collaboration Hub. A manufacturer-independend and human-centered framework for small and medium-sized companies. The HRC-Hub enables easy assembly automation by leveraging augmented reality to enable easy programming by demonstration.

# Camera to end-effector transformation via ArUco Pose Estimation

**Goal**: Get transformation matrix from sensor to robot flange.

### Steps:
1. Tool calibration on robot
2. Base calibration of template on robot
3. For each sensor:
     1. Start pose receiver to record current poses and motion from robot
     2. Start frame capturing for sensor
     3. Start program on robot
4. Select and separate frames into motions
5. Get closest points for all frames during motion
6. ArUcoPoseEstimationGrid
7. getSensor2ee_transformations input closest_poses_aruco.csv
8. averageTransformationSensor input closest_poses_aruco_transformation.csv

# Record test data and create object masks

**Goal**: Create an output folder, which can be usesd to run pose estimation with existing methods (FoundationPose, MegaPose and SAM6D).

**Required Input**:
- Object name/number
- Name of run with all parameters
  - Resolution
  - Light
  - Template Type
  - Pose via ArUco or EE pose
  - ...
- Proposed naming scheme: `/RUN_DESCRIPTION_XYZ_OBJECT/SENSOR/rgb/depth/mask...`

### Steps:
1. Start recording
    - Robot poses --> PoseReceiver/pose_receiver_udp_json.py
    - camera streams
2. *TODO*: Adapted Select and Separate: Keep all frames in folder. Save Frame info and robot pose to json, delete all frames between motions.
3. *TODO*: Get clostest points with Sync Delay (or better to merge 2 and 3 into one operation?) --> Rename output files in ascending order from 0
4. *OPTIONAL*: Run ArUco pose estimation
5. Blenderproc to generate masks and Ground Truth Information (*TODO*: If ArUco pose estimation is used, and alternative version of the script shall be used)
6. *TODO*: Final helper script to convert folder/files into suited input for FoundationPose/MegaPose/SAM6D


# Format of BOP datasets

https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md

```
DATASET_NAME
├─ camera[_TYPE].json
├─ dataset_info.json
├─ test_targets_bop19.json
├─ models[_MODELTYPE][_eval]
│  ├─ models_info.json
│  ├─ obj_OBJ_ID.ply
├─ train|val|test[_TYPE]|onboarding_static|onboarding_dynamic
│  ├─ SCENE_ID|OBJ_ID
│  │  ├─ scene_camera.json
│  │  ├─ scene_gt.json
│  │  ├─ scene_gt_info.json
│  │  ├─ depth
│  │  ├─ mask
│  │  ├─ mask_visib
│  │  ├─ rgb|gray
```