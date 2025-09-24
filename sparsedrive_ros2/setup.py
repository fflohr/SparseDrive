from setuptools import setup, find_packages

package_name = 'sparsedrive_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/sdc_car.png', 'resource/legend.png']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dev',
    maintainer_email='dev@null.com',
    description='ROS2 wrapper for SparseDrive',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sparsedrive_node = sparsedrive_ros2.sparsedrive_node:main',
            'bev_renderer = sparsedrive_ros2.visualization.bev_render:main',
        ],
    },
)