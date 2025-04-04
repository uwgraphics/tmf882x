from setuptools import find_packages, setup

package_name = 'tmf882x'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carter',
    maintainer_email='cpsiff@gmail.com',
    description='Read from and visualize TMF882X sensors',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tmf882x_pub = tmf882x.tmf882x_pub:main',
            'tmf882x_vis = tmf882x.tmf882x_vis:main',
            'plane_vis = tmf882x.plane_vis:main',
        ],
    },
)
