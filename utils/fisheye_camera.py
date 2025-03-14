#!/usr/bin/env python
from __future__ import annotations
# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from carla import ColorConverter as cc
import numpy as np
import weakref
from scipy.spatial.transform import Rotation as R
from camera_models.base_projection import BaseProjection
import cv2
from typing import Tuple


def process_image(image: carla.libcarla.Image)-> np.ndarraty:
    """ The callback function which gets raw image and convert it to an array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array[:, :, ::-1] 


class PinholeCamera:
    """Simulate simple pinhole camera in carla.
 
    Args:
        parent_actor: vehicle actor to attach the camera
        width: width of the image
        height: height of the image
        fov: field of view in degrees
        tick: simulation seconds between sensor captures (ticks).
        x: x position with respect to the ego vehicle in meters
        y: y position with respect to the ego vehicle in meters
        z: z position with respect to the ego vehicle in meters
        roll: roll angle in degrees
        pitch: pitch angle in degrees
        yaw: yaw angle in degrees
        camera_type: can be: 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation' or 'sensor.camera.depth'
    """
    def __init__(self, parent_actor: carla.Actor, width: int, height: int, fov: int=90, tick: float=0.0,
                 x: float=-6.5, y:float=0.0, z:float=2.7,
                 roll:float=0.0, pitch:float=0.0, yaw:float=0.0,
                 camera_type: str ='sensor.camera.rgb')-> None:
        """Init."""
        if camera_type not in [ 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']:
            raise ValueError(f"Camera type {camera_type} is not supported!")
        # Carla related parameters
        self._parent = parent_actor
        # Visualization related parameters
        self.camera_type = camera_type
        self.image = None
        self.frame = 0
        # Set up the sensor
        blueprint = self._parent.get_world().get_blueprint_library().find(camera_type)
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', str(width))
        blueprint.set_attribute('image_size_y', str(height))
        blueprint.set_attribute('fov', str(fov))
        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', str(tick))

        # Provide the position of the sensor relative to the vehicle.
        transform  = carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))

        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.sensor = self._parent.get_world().spawn_actor(blueprint, transform, attach_to=self._parent)

        # Estimate intrinsic matrix for the camera
        # For pinhole camera the relation:
        # r = f * tan(theta)
        # theta is angle between principle point and incoming ray
        # r is the distance from principle point to incoming ray
        # r = width / 2.0; theta = np.deg2rad(FOV / 2.0) or FOV * pi /360
        # for more info see: 
        # https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses
        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] = calibration[1, 1] = float(width) / (2.0 * np.tan(float(fov) * np.pi / 360.0))
        self.sensor.calibration = calibration

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._parse_image(weak_self, image))

    def destroy(self)-> None:
        """Destroys camera."""
        self.sensor.destroy()

    @staticmethod
    def _parse_image(weak_self: PinholeCamera, image: carla.libcarla.Image)-> None:
        """Parse image and postprocess it."""
        self = weak_self()
        if not self:
            return
        if self.camera_type=='sensor.camera.depth':
            image.convert(cc.LogarithmicDepth) #  'Camera Depth (Logarithmic Gray Scale)'     
        elif self.camera_type== 'sensor.camera.semantic_segmentation':
            image.convert(cc.CityScapesPalette)
        else:         
            image.convert(cc.Raw)
            
        self.image = process_image(image)
        self.frame += 1



class FisheyeCamera:
    """ FisheyeCamera class that simulates equidistant projection fish eye camera.
    
    Args:
        parent_actor: parent carla actor (e.g. vehicle) to attach the camera
        width: image width
        height: image height
        fov: field of view in degrees
        tick: simulation seconds between sensor captures (ticks=0.0 maximum possible).
        x: x position with respect to the ego vehicle in meters
        y: y position with respect to the ego vehicle in meters
        z: z position with respect to the ego vehicle in meters
        roll: roll angle in degrees
        pitch: pitch angle in degrees
        yaw: yaw angle in degrees
        camera_type: can be: 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation' or 'sensor.camera.depth'
    """
    def __init__(self, parent_actor: carla.Actor, camera_model: BaseProjection, width: int=640, height: int=640, fov:int=180, tick:float=0.0,
                 x: float=-6.5, y: float=0.0, z:float=2.7, roll:float=0.0, pitch:float=0.0, yaw: float=0.0, k0: float=0.0, k1: float=0.0, k2: float=0.0, k3: float=0.0, k4: float=0.0,
                 camera_type='sensor.camera.rgb')-> None:
        # Carla parameters
        self._parent = parent_actor # vehicle where camera will be attached
        self.image = None
        self._five_pinhole_image = None
        self.frame = 0
      
        # initialize fisheye camera projection
        self.projection_model = camera_model.from_fov(width=width, height=height, fov=fov, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)

        # Create cube from 5 pinhole cameras for reprojection to fish eye

        # We create pinhole with the same focal length as fish eye camera and FOV = 90
        # From the formula r = f * tan(theta) we can get 
        #  width / 2 = f * tan(FOV/2) = f * tan(45 deg); width = 2.0 * f (tan(45 deg) = 1), also we assume width = height
        pinhole_width = int(2.0 *  self.projection_model.fx)
        pinhole_height = int(2.0 *  self.projection_model.fy)

        # initialize all cameras
        main_rot = R.from_euler('xyz',[roll, pitch, yaw], degrees=True).as_matrix()
        self._front_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=roll, pitch=pitch, yaw=yaw,camera_type=camera_type)

        # First we rotate camera 90 degrees to the left
        # Then chain it to the main rotation to the vehicle
        left_local_rot =  R.from_euler('xyz',[0.0, 0.0, -90], degrees=True).as_matrix()
        left_rot = R.from_matrix( main_rot @ left_local_rot).as_euler('xyz', degrees=True)
        self._left_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=left_rot[0], pitch=left_rot[1], yaw=left_rot[2],camera_type=camera_type)

        # Second we rotate camera 90 degrees to the right
        # Then chain it to the main rotation to the vehicle
        right_local_rot =  R.from_euler('xyz',[0.0, 0.0, 90], degrees=True).as_matrix()
        right_rot = R.from_matrix(main_rot @ right_local_rot).as_euler('xyz', degrees=True)
        self._right_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=right_rot[0], pitch=right_rot[1], yaw=right_rot[2],camera_type=camera_type)

        # Third we rotate camera 90 degrees to the top
        # Then chain it to the main rotation to the vehicle
        top_local_rot =  R.from_euler('xyz',[0.0, 90.0, 0.0], degrees=True).as_matrix()
        top_rot = R.from_matrix(main_rot @ top_local_rot).as_euler('xyz', degrees=True)
        self._top_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=top_rot[0], pitch=top_rot[1], yaw=top_rot[2],camera_type=camera_type)

        # Fourth we rotate camera 90 degrees to the bottom
        # Then chain it to the main rotation to the vehicle
        bottom_local_rot =  R.from_euler('xyz',[0.0, -90.0, 0.0], degrees=True).as_matrix()
        bottom_rot = R.from_matrix(main_rot @ bottom_local_rot).as_euler('xyz', degrees=True)
        self._bottom_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=bottom_rot[0], pitch=bottom_rot[1], yaw=bottom_rot[2],camera_type=camera_type)
        

        # For all 5 image matrices intrinsc will be the same therefore will take from one
        self.pinhole_intrisic_matrix = self._front_pinhole.sensor.calibration

        # Compute mapping
        self.maptable = self.compute_mapping(fisheye_width=width, fisheye_height=height, projection_model=self.projection_model, pinhole_intrisic_matrix=self.pinhole_intrisic_matrix)


    def compute_mapping(self, fisheye_width: int, fisheye_height: int, projection_model: BaseProjection, pinhole_intrisic_matrix: np.ndarray)-> np.ndarray:
        """Compute mapping for inverse warping between 5 pinhole to fish eye."""

        # Get image coordinates
        y, x = np.meshgrid(range(fisheye_height), range(fisheye_width), indexing='ij')

        # Here pixels coords of the shape [height, width, 2]
        fisheye_image_coords = np.concatenate((x[..., None], y[..., None]), axis=-1)
        shape = fisheye_image_coords.shape
        fisheye_image_coords = fisheye_image_coords.reshape(-1, 2)
        maptable = np.zeros_like(fisheye_image_coords).T

        fisheye_rays = projection_model.from_2d_to_3d(fisheye_image_coords)
        fisheye_rays = fisheye_rays.T  

        # Get coords from front given the fact that front camera FOV 90 for horizontal and vertical
        # we can compute the x and y coords        
        pinhole_width =  int(2.0 * projection_model.fx)
        pinhole_height = int(2.0 * projection_model.fy)

        # Front camera 
        front_camera_mask = np.ones((shape[0]*shape[1])).astype(np.bool)
        front_camera_mask, front_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=front_camera_mask, camera_direction="front")
        maptable[:, front_camera_mask] = front_cam_img_coords

        # Left camera
        left_camera_mask = (fisheye_image_coords[:, 0] <=fisheye_width / 2.0)
        left_camera_mask, left_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=left_camera_mask, camera_direction="left")
        maptable[:, left_camera_mask] = left_cam_img_coords

        # Right camera  
        right_camera_mask =  (fisheye_image_coords[:, 0] >fisheye_width / 2.0)
        right_camera_mask, right_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=right_camera_mask,camera_direction="right")
        maptable[:, right_camera_mask] = right_cam_img_coords

        # Top camera
        top_camera_mask = (fisheye_image_coords[:, 1] <=fisheye_height /2.0) 
        top_camera_mask, top_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=top_camera_mask, camera_direction="top")
        maptable[:, top_camera_mask] = top_cam_img_coords

        # Bottom camera
        bottom_camera_mask = (fisheye_image_coords[:, 1] >fisheye_height /2.0) 
        bottom_camera_mask, bottom_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=bottom_camera_mask, camera_direction="bottom")
        maptable[:, bottom_camera_mask] = bottom_cam_img_coords
        return maptable.T.reshape(shape).astype(np.float32)

    def get_coordinates_for_five_pinhole_image(self, fisheye_rays: np.ndarray, pinhole_width: int, pinhole_height: int, pinhole_intrisic_matrix: np.ndarray, camera_mask: str, camera_direction: str, margin: float = 1.5)-> Tuple[np.ndarray, np.ndarray]:
        """Gets coordinates for the box image for given camera in a maptable."""

        if camera_direction == "front":
            cam_transform = np.eye(3)
            box_idx = np.asarray([2 * pinhole_width, 0.0])[:, None]
        if camera_direction == "left":
            cam_transform = R.from_euler('xyz',[0.0, 90, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([0.0, 0.0])[:, None]

        if camera_direction == "right":
            cam_transform = R.from_euler('xyz',[0.0, -90, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([4 * pinhole_width, 0.0])[:, None]

        if camera_direction == "top":
            cam_transform = R.from_euler('xyz',[-90, 0.0, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([pinhole_width, 0.0])[:, None]
    
        if camera_direction == "bottom":
            cam_transform = R.from_euler('xyz',[90, 0.0, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([3 * pinhole_width, 0.0])[:, None]

        fisheye_rays = fisheye_rays[:, camera_mask].copy()

        transform = pinhole_intrisic_matrix @ cam_transform
        cam_img_coords = transform @ fisheye_rays
        cam_img_coords = cam_img_coords[:2, :] / cam_img_coords[2][None, :]

        mask_image_coords = (cam_img_coords[0]>=-margin) & (cam_img_coords[0]< pinhole_width + margin)  & (cam_img_coords[1]>=-margin) & (cam_img_coords[1]< pinhole_height +margin) 
        cam_img_coords = cam_img_coords[:, mask_image_coords]    

        # Fix for floating point coordinates by using margin
        # This part removes black dots on the borders of the five pinhole cube
        cam_img_coords[0, cam_img_coords[0, :] >= pinhole_width -1.0] = pinhole_width -1.0
        cam_img_coords[1, cam_img_coords[1, :] >= pinhole_height -1.0] = pinhole_height -1.0
        cam_img_coords[0, cam_img_coords[0, :] <= 0.0] = 0.0
        cam_img_coords[1, cam_img_coords[1, :] <=0.0] = 0.0

        cam_img_coords += box_idx
        camera_mask[camera_mask] = mask_image_coords


        return camera_mask, cam_img_coords


    def destroy(self)-> None:
        """Delete all cameras."""
        actors = [
            self._front_pinhole,
            self._left_pinhole,
            self._right_pinhole,
            self._top_pinhole,
            self._bottom_pinhole,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def create_fisheye_image(self)->None:
        """Creates fisheye image.
        
        Note: this function should be called in order to update image for fish eye camera.
        """
        self._five_pinhole_image = np.hstack((self._left_pinhole.image,
                              self._top_pinhole.image,
                             self._front_pinhole.image,
                            self._bottom_pinhole.image,                                
                            self._right_pinhole.image)).astype(np.float32)

        remapped_img = cv2.remap(self._five_pinhole_image, self.maptable[..., 0], self.maptable[..., 1], cv2.INTER_NEAREST)

        self.image = remapped_img.astype('uint8')

        self.frame += 1.0
