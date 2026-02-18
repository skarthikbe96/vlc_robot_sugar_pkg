from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # -------------------------
    # Launch arguments (one set)
    # -------------------------
    declares = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # Perception inputs/frames
        DeclareLaunchArgument('raw_pc_topic', default_value='/camera_pan_tilt/points_xyzrgb'),
        DeclareLaunchArgument('camera_frame', default_value='panda_mounted_husky/camera_pan_tilt_link/camera_pan_tilt'),
        DeclareLaunchArgument('base_frame', default_value='panda_mounted_husky/camera_pan_tilt_link/camera_pan_tilt'),

        # Paper preprocessing
        DeclareLaunchArgument('use_workspace_crop', default_value='true'),
        DeclareLaunchArgument('crop_x_min', default_value='-0.42'),
        DeclareLaunchArgument('crop_x_max', default_value='0.8'),
        DeclareLaunchArgument('crop_y_min', default_value='-0.60'),
        DeclareLaunchArgument('crop_y_max', default_value='0.15'),
        DeclareLaunchArgument('crop_z_min', default_value='0.3'),
        DeclareLaunchArgument('crop_z_max', default_value='0.65'),

        DeclareLaunchArgument('voxel_size', default_value='0.005'),     # paper 0.5cm
        DeclareLaunchArgument('num_points', default_value='10000'),
        DeclareLaunchArgument('sampling_method', default_value='random'),  # random|fps
        DeclareLaunchArgument('publish_debug_clouds', default_value='true'),
        DeclareLaunchArgument('debug_dir', default_value='/tmp/sugar_debug'),
        DeclareLaunchArgument('clear_debug_dir_on_start', default_value='true'),
        DeclareLaunchArgument('save_debug_npy', default_value='true'),

        # Sugar bridge / inference
        DeclareLaunchArgument('conda_env', default_value='robo3d'),
        DeclareLaunchArgument('conda_exe', default_value='/home/rebellion/anaconda3/bin/conda'),
        DeclareLaunchArgument('sugar_script',
            default_value='/home/rebellion/mobile_robotics/vlc_robot_sugar_ws/src/robot_sugar_pkg/robot_sugar/sugar_infer_standalone.py'),

        # REG (OCID-Ref)
        DeclareLaunchArgument('reg_config',
            default_value='/home/rebellion/mobile_robotics/vlc_robot_sugar_ws/src/robot_sugar_pkg/robot_sugar/robo3d/configs/downstream/pct_ocidref.yaml'),
        DeclareLaunchArgument('reg_checkpoint',
            default_value='/home/rebellion/mobile_robotics/robot_sugar_ws/src/sugar_policy_ros2/robot_sugar/data3d/experiments/downstreams/ocidref/pc10k-512x32-openclip-init.objaverse4_nolvis.multi.4096.mtgrasp-freeze.enc/ckpts/model_step_80000.pt'),
        DeclareLaunchArgument('mask_thresh', default_value='0.5'),

        # GPS
        DeclareLaunchArgument('gps_config',
            default_value='/home/rebellion/mobile_robotics/vlc_robot_sugar_ws/src/robot_sugar_pkg/robot_sugar/robo3d/configs/pretrain/pct_pretrain.yaml'),
        DeclareLaunchArgument('gps_checkpoint',
            default_value='/home/rebellion/mobile_robotics/robot_sugar_ws/src/sugar_policy_ros2/robot_sugar/data3d/experiments/pretrain/shapenet/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.shapenet.single/ckpts/model_step_100000.pt'),
        DeclareLaunchArgument('grasp_thresh', default_value='0.4'),

        # Inference runtime
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('unit_scale', default_value='1.0'),
        DeclareLaunchArgument('keep_ratio', default_value='0.95'),
        DeclareLaunchArgument('timeout_s', default_value='120.0'),
        DeclareLaunchArgument('place_lift', default_value='0.15'),

        # Logging / start delay
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('start_delay_s', default_value='2.0'),
    ]

    lc = {k: LaunchConfiguration(k) for k in [
        'use_sim_time',
        'raw_pc_topic', 'camera_frame', 'base_frame',
        'use_workspace_crop', 'crop_x_min', 'crop_x_max', 'crop_y_min', 'crop_y_max', 'crop_z_min', 'crop_z_max',
        'voxel_size', 'num_points', 'sampling_method', 'publish_debug_clouds',
        'debug_dir', 'clear_debug_dir_on_start', 'save_debug_npy',
        'conda_env', 'conda_exe', 'sugar_script',
        'reg_config', 'reg_checkpoint', 'mask_thresh',
        'gps_config', 'gps_checkpoint', 'grasp_thresh',
        'device', 'unit_scale', 'keep_ratio', 'timeout_s', 'place_lift',
        'log_level', 'start_delay_s'
    ]}

    # ---------------------------------------
    # 1) Paper preprocessing perception node
    # ---------------------------------------
    perception_params = {
        'use_sim_time': lc['use_sim_time'],
        'pc_topic': lc['raw_pc_topic'],
        'base_frame': lc['base_frame'],
        'camera_frame': lc['camera_frame'],

        'use_workspace_crop': lc['use_workspace_crop'],
        'crop_x_min': ParameterValue(lc['crop_x_min'], value_type=float),
        'crop_x_max': ParameterValue(lc['crop_x_max'], value_type=float),
        'crop_y_min': ParameterValue(lc['crop_y_min'], value_type=float),
        'crop_y_max': ParameterValue(lc['crop_y_max'], value_type=float),
        'crop_z_min': ParameterValue(lc['crop_z_min'], value_type=float),
        'crop_z_max': ParameterValue(lc['crop_z_max'], value_type=float),

        'voxel_size': ParameterValue(lc['voxel_size'], value_type=float),
        'num_points': ParameterValue(lc['num_points'], value_type=int),
        'sampling_method': lc['sampling_method'],
        'publish_debug_clouds': lc['publish_debug_clouds'],

        'debug_dir': lc['debug_dir'],
        'clear_debug_dir_on_start': lc['clear_debug_dir_on_start'],
        'save_debug_npy': lc['save_debug_npy'],
    }

    perception = Node(
        package='robot_sugar_pkg',
        executable='sugar_perception_node',
        name='sugar_perception_node',
        output='screen',
        parameters=[perception_params],
        arguments=['--ros-args', '--log-level', lc['log_level']],
    )

    # ---------------------------------------------------
    # 2) Sugar bridge consumes the FINAL 4096 point cloud
    # ---------------------------------------------------
    # IMPORTANT: feed processed cloud into sugar_ros_bridge
    processed_pc_topic = '/sugar/final_4096_cloud'

    sugar_params = {
        'use_sim_time': lc['use_sim_time'],
        'conda_env': lc['conda_env'],
        'conda_exe': lc['conda_exe'],
        'sugar_script': lc['sugar_script'],

        'reg_config': lc['reg_config'],
        'reg_checkpoint': lc['reg_checkpoint'],
        'mask_thresh': ParameterValue(lc['mask_thresh'], value_type=float),

        'gps_config': lc['gps_config'],
        'gps_checkpoint': lc['gps_checkpoint'],
        'grasp_thresh': ParameterValue(lc['grasp_thresh'], value_type=float),

        'pc_topic': processed_pc_topic,   # <--- key change
        'camera_frame': lc['camera_frame'],
        'base_frame': lc['base_frame'],

        'device': lc['device'],
        'num_points': ParameterValue(lc['num_points'], value_type=int),
        'unit_scale': ParameterValue(lc['unit_scale'], value_type=float),
        'keep_ratio': ParameterValue(lc['keep_ratio'], value_type=float),
        'timeout_s': ParameterValue(lc['timeout_s'], value_type=float),
        'place_lift': ParameterValue(lc['place_lift'], value_type=float),
    }

    sugar = Node(
        package='robot_sugar_pkg',
        executable='sugar_ros_bridge',
        name='sugar_bridge',
        output='screen',
        parameters=[sugar_params],
        arguments=['--ros-args', '--log-level', lc['log_level']],
    )

    # Delay sugar a bit so perception publishes first
    sugar_delayed = TimerAction(period=lc['start_delay_s'], actions=[sugar])

    return LaunchDescription(declares + [perception, sugar_delayed])
