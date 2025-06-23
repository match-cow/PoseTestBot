## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##    Multi-Camera RealSense Visualization Only    ##
#####################################################

import argparse
import time
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """Class to handle individual RealSense camera operations."""
    
    def __init__(self, serial_number, fps=30):
        self.serial_number = serial_number
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.is_running = False
        
    def configure_camera(self):
        """Configure the camera pipeline."""
        # Enable device by serial number
        self.config.enable_device(self.serial_number)
        
        # Get device to check product line
        ctx = rs.context()
        devices = ctx.query_devices()
        device = None
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                device = dev
                break
                
        if device is None:
            raise RuntimeError(f"Camera with serial {self.serial_number} not found")
            
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        # Check for RGB sensor
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            raise RuntimeError(f"No RGB sensor found on camera {self.serial_number}")

        # Configure streams
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.fps)
        
        if device_product_line == "L500":
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.fps)
        else:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, self.fps)
            
    def start_streaming(self):
        """Start the camera streaming."""
        profile = self.pipeline.start(self.config)
        
        # Create align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.is_running = True
        
    def stop_streaming(self):
        """Stop the camera streaming."""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            
    def get_frames(self):
        """Get aligned color and depth frames."""
        if not self.is_running:
            return None, None
            
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            aligned_frames = self.align.process(frames)
            
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                return None, None
                
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except RuntimeError as e:
            print(f"Timeout getting frames from camera {self.serial_number}")
            return None, None


def discover_cameras():
    """Discover all connected RealSense cameras."""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    camera_serials = []
    for device in devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        product_name = device.get_info(rs.camera_info.name)
        print(f"Found camera: {product_name} (Serial: {serial_number})")
        camera_serials.append(serial_number)
        
    return camera_serials


def main():
    """Main function to visualize multiple RealSense cameras."""
    parser = argparse.ArgumentParser(description="Multi-Camera Realsense Visualization")
    parser.add_argument(
        "--fps",
        type=int,
        default=15,  # Lower FPS for stability with multiple cameras
        help="Specify the frames per second for capturing.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        help="Specify camera serial numbers to use (default: use all available).",
    )
    
    args = parser.parse_args()
    fps = args.fps
    
    # Discover cameras
    print("Discovering RealSense cameras...")
    available_cameras = discover_cameras()
    
    if not available_cameras:
        print("No RealSense cameras found!")
        return
        
    # Filter cameras if specific ones are requested
    if args.cameras:
        camera_serials = [serial for serial in args.cameras if serial in available_cameras]
        if not camera_serials:
            print("None of the specified cameras were found!")
            return
    else:
        camera_serials = available_cameras
        
    print(f"Using cameras: {camera_serials}")
    
    # Create camera objects
    cameras = []
    for serial in camera_serials:
        try:
            camera = RealSenseCamera(serial, fps)
            camera.configure_camera()
            cameras.append(camera)
            print(f"Camera {serial} configured successfully")
        except Exception as e:
            print(f"Failed to configure camera {serial}: {e}")
            
    if not cameras:
        print("No cameras could be configured!")
        return
        
    # Start all cameras
    print("Starting camera streams...")
    active_cameras = []
    for camera in cameras:
        try:
            camera.start_streaming()
            active_cameras.append(camera)
            print(f"Camera {camera.serial_number} started successfully")
            time.sleep(0.5)  # Small delay between starting cameras
        except Exception as e:
            print(f"Failed to start camera {camera.serial_number}: {e}")
            
    if not active_cameras:
        print("No cameras could be started!")
        return
        
    print(f"Successfully started {len(active_cameras)} cameras")
    print("Press 'q' or ESC in any window to stop...")
    
    frame_count = 0
    
    try:
        while True:
            all_frames_valid = True
            
            # Get frames from all cameras
            for i, camera in enumerate(active_cameras):
                color_image, depth_image = camera.get_frames()
                
                if color_image is not None:
                    # Resize for display if needed
                    display_color = cv2.resize(color_image, (640, 480))
                    
                    # Create depth colormap for visualization
                    if depth_image is not None:
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03), 
                            cv2.COLORMAP_JET
                        )
                        depth_display = cv2.resize(depth_colormap, (640, 480))
                        
                        # Create side-by-side display
                        combined = np.hstack((display_color, depth_display))
                        window_name = f"Camera {camera.serial_number} - Color | Depth"
                    else:
                        combined = display_color
                        window_name = f"Camera {camera.serial_number} - Color Only"
                        
                    cv2.imshow(window_name, combined)
                else:
                    all_frames_valid = False
                    
            # Check for exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC key
                break
                
            frame_count += 1
            """
            if frame_count % 30 == 0:  # Print status every 30 frames
                print(f"Frames processed: {frame_count}")
                """
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        print("Stopping cameras...")
        for camera in active_cameras:
            camera.stop_streaming()
            
        cv2.destroyAllWindows()
        print("All cameras stopped")


if __name__ == "__main__":
    main()