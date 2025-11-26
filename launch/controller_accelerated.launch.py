"""
Launch file for Fractional MPC Controller with CPU+GPU Acceleration
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    fractional_mpc_dir = get_package_share_directory('fractional_mpc_controller')

    # Launch arguments
    enable_gpu = LaunchConfiguration('enable_gpu', default='true')
    enable_warm_start = LaunchConfiguration('enable_warm_start', default='true')
    worker_threads = LaunchConfiguration('worker_threads', default='2')
    control_dt = LaunchConfiguration('control_dt', default='0.01')
    mpc_horizon = LaunchConfiguration('mpc_horizon', default='25')
    integral_gain = LaunchConfiguration('integral_gain', default='2.0')
    state_cost_position = LaunchConfiguration('state_cost_position', default='600.0')
    state_cost_velocity = LaunchConfiguration('state_cost_velocity', default='12.0')
    control_cost = LaunchConfiguration('control_cost', default='0.15')
    spring_k = LaunchConfiguration('spring_const', default='1.0')
    
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument(
            'enable_gpu',
            default_value='true',
            description='Enable GPU acceleration with CuPy'
        ),
        DeclareLaunchArgument(
            'enable_warm_start',
            default_value='true',
            description='Enable warm-starting in MPC solver'
        ),
        DeclareLaunchArgument(
            'worker_threads',
            default_value='4',
            description='Number of background worker threads for I/O'
        ),
        DeclareLaunchArgument(
            'control_dt',
            default_value='0.01',
            description='Control loop sampling time (seconds)'
        ),
        DeclareLaunchArgument(
            'mpc_horizon',
            default_value='5',
            description='MPC prediction horizon (default=15 for 46% speed improvement vs 25)'
        ),
        DeclareLaunchArgument(
            'integral_gain',
            default_value='.8',
            description='Integral action gain (Ki) - increase to reduce steady-state error'
        ),
        DeclareLaunchArgument(
            'state_cost_position',
            default_value='600.0',
            description='MPC cost weight for position tracking (Q_pos)'
        ),
        DeclareLaunchArgument(
            'state_cost_velocity',
            default_value='1.2',
            description='MPC cost weight for velocity tracking (Q_vel)'
        ),
        DeclareLaunchArgument(
            'control_cost',
            default_value='20.2',
            description='MPC cost weight for control (R)'
        ),
        DeclareLaunchArgument(
            'spring_const',
            default_value='10.0',
            description='Spring-Mass constant'
        ),
        
        # Accelerated MPC Controller Node
        Node(
            package='fractional_mpc_controller',
            executable='fractional_mpc_controller_accelerated',
            name='fractional_mpc_controller_accel',
            output='screen',
            parameters=[
                {'n_joints': 9},
                {'fractional_order': 0.30},
                {'control_dt': control_dt},
                {'mpc_horizon': mpc_horizon},
                {'joint_names': ['base_to_cylinder_joint', 'cylinder_to_arm1_joint',  'arm_link1_to_arm2_joint', 'arm_link2_to_arm3_joint',
                                 'arm_link3_to_arm4_joint', 'arm_link4_to_armFork_joint', 'basket_link_to_armFork_joint', 'basket_crane_link_to_basket_link','winch_link_to_basket_crane_link']}, 
                #{'joint_names': [f'joint_{i}' for i in range(9)]},
                # Acceleration parameters
                {'enable_gpu': enable_gpu},
                {'enable_warm_start': enable_warm_start},
                {'worker_threads': worker_threads},
                # Cost weights (balanced to avoid oscillations + offset)
                {'state_cost_position': state_cost_position},
                {'state_cost_velocity': state_cost_velocity},
                {'control_cost': control_cost},
                {'integral_gain': integral_gain},
                {'spring_const': spring_k},
                # Control limits
                {'u_min': [-3.1415927] * 9},
                {'u_max': [ 3.1415927] * 9},
            ],
            remappings=[
                ('joint_states', '/joint_states'),
                ('reference_state', '/reference_state'),
                ('control_command', '/ugv0/telehandler/joint_states'),
                ('mpc_predicted_state', '/mpc_predicted_state'),
            ]
        ),
    ])
