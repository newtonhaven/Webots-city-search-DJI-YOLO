from controller import Robot
import math, sys
import numpy as np            
from ultralytics import YOLO

#DELETE THESE 
# from controller import Keyboard
# import cv2
# import time
#-------------------------------
def clamp(v, mn, mx):
    return max(mn, min(v, mx))

class Drone(Robot):
    K_VERTICAL_THRUST = 68.5 # hover engine thrust
    K_VERTICAL_OFFSET = 0.6 # 
    K_VERTICAL_P      = 10.0 # altitude
    K_ROLL_P          = 15.0 # front-back roll
    K_PITCH_P         = 1.0 # left-right pitch
    target_precision  = 1.0 # waypoint proximity, error in meters
    
    # City boundaries   # Default values
    CITY_MIN_X = -310.0 #-317.0
    CITY_MAX_X =  210.0 #317.0
    CITY_MIN_Y = -220.0 #-325.5
    CITY_MAX_Y =  313.0 #325.5

    # -----------------------------------------
    # INITIALIZATION DRONE CONFIG
    # -----------------------------------------
    def __init__(self, ID):
        super().__init__()
        self.ID = ID
        self.ts = int(self.getBasicTimeStep())
        print(f"[DRONE {ID}] INITIALIZING...")

        # --- KEYBOARD SETUP --- DELETEEEEEEEEEEEEEEEEEEEE
        # self.keyboard = Keyboard()
        # self.keyboard.enable(self.ts)  # <--- 2. Enable Keyboard
        
        # --- SENSORS SETUP ---
        # YOLO model load
        try:
            self.yolo_model = YOLO("yolov8_trained.pt")
            self.vision_ready = True
            print(f"[DRONE {ID}] YOLO Model Loaded.")
        except Exception as e:
            print(f"[DRONE {ID}] WARNING: YOLO failed to load: {e}")
            self.vision_ready = False
        self.step_counter = 0

        # radio
        self.receiver = self.getDevice("receiver_from_supervisor")
        self.receiver.enable(self.ts)
        self.emitter = self.getDevice("emitter_to_supervisor")

        # sensors
        self.imu = self.getDevice("inertial unit"); self.imu.enable(self.ts)
        self.gps = self.getDevice("gps");           self.gps.enable(self.ts)
        self.gyro = self.getDevice("gyro");         self.gyro.enable(self.ts)
        self.camera = self.getDevice("camera");     self.camera.enable(self.ts)

        # motors
        self.fl = self.getDevice("front left propeller")
        self.fr = self.getDevice("front right propeller")
        self.rl = self.getDevice("rear left propeller")
        self.rr = self.getDevice("rear right propeller")
        for m in [self.fl, self.fr, self.rl, self.rr]:
            m.setPosition(float('inf'))
            m.setVelocity(1.0)

        # state
        self.target_alt    = 25.0
        self.pose          = [0]*6
        self.wp_index      = 0
        self.control_mode  = "SEARCH"
        self.last_nav_update = 0.0
        
        # persistent inputs
        self.yaw_d   = 0.0
        self.pitch_d = 0.0
        
        self.has_taken_off = False 

        # generate zig-zag pattern
        self.search_wp = self.generate_zigzag(ID)
        print(f"[DRONE {ID}] READY.")

    def scan_for_target(self):
        # --- SCREENSHOT IMAGE FROM CAMERA --- DELEEETEEEEEEEEEEEEEEEE
        # delete later
        # key = self.keyboard.getKey()
        #if key == Keyboard.UP:
        #    print(f"[DRONE {self.ID}] SNAPSHOT TAKEN!")
        #    self.camera.saveImage(f"drone_{self.ID}_{int(time.time())}.jpg", 100) # Built-in Webots function
            
        # YOLO INFERENCE
        if not self.vision_ready: return False
        
        raw_img = self.camera.getImage()
        if raw_img is None: return False
        
        w, h = self.camera.getWidth(), self.camera.getHeight()
        img_arr = np.frombuffer(raw_img, dtype=np.uint8).reshape((h, w, 4))
        img_bgr = img_arr[:, :, :3]
        results = self.yolo_model(img_bgr, verbose=False, classes=[0], conf=0.65)
        
        
        for r in results:
            for box in r.boxes:
                # 4. ADDITIONAL COLOR CHECK
                # Calculate center of the detected box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Clamp to ensure we don't check pixels outside the image
                cx = max(0, min(cx, w-1))
                cy = max(0, min(cy, h-1))

                # Get the actual color of the pixel at the center
                # Webots image is BGR: [Blue, Green, Red]
                b_val, g_val, r_val = img_bgr[cy, cx]

                # --- TARGET COLOR LOGIC ---
                #check for Hex: #0055ff
                target_r, target_g, target_b = 0, 85, 255
                
                # Allow some wiggle room (tolerance) because lighting changes colors
                tolerance = 60 

                # Check difference
                diff = abs(int(r_val) - target_r) + \
                       abs(int(g_val) - target_g) + \
                       abs(int(b_val) - target_b)

                if diff < tolerance * 3:
                    # Confidence + Color Match = Success
                    print(f"[VISION] Target Confirmed! Color match score: {diff} Confidence: {float(box.conf):.2f}")
                    return True
                else:
                    print(f"[VISION] Saw target, but color was wrong. (Diff: {diff})")
        return False

    def generate_zigzag(self, drone_id):
        MARGIN        = 10.0
        STRIP_SPACING = 30.0
        
        # This angle tilts the path. 
        # POSITIVE = Counter-Clockwise, NEGATIVE = Clockwise
        ROTATION_ANGLE_DEG = -25.235
        theta = math.radians(ROTATION_ANGLE_DEG)
        c, s = math.cos(theta), math.sin(theta)

        if drone_id == 1:   # top-left
            xmin, xmax = self.CITY_MIN_X + MARGIN, -MARGIN
            ymin, ymax = MARGIN, self.CITY_MAX_Y - MARGIN
        elif drone_id == 2: # top-right
            xmin, xmax = MARGIN, self.CITY_MAX_X - MARGIN
            ymin, ymax = MARGIN, self.CITY_MAX_Y - MARGIN
        elif drone_id == 3: # bottom-left
            xmin, xmax = self.CITY_MIN_X + MARGIN, -MARGIN
            ymin, ymax = self.CITY_MIN_Y + MARGIN, -MARGIN
        else:               # 4 bottom-right
            xmin, xmax = MARGIN, self.CITY_MAX_X - MARGIN
            ymin, ymax = self.CITY_MIN_Y + MARGIN, -MARGIN

        waypoints = []
        y = ymin
        direction = 0
        
        while y <= ymax:
            # 2. Determine the "straight" point
            if direction == 0: 
                px, py = xmax, y
            else:              
                px, py = xmin, y
            
            # 3. ROTATE the point to match the city tilt
            # Formula: x' = x*cos - y*sin, y' = x*sin + y*cos
            rx = px * c - py * s
            ry = px * s + py * c
            
            waypoints.append((rx, ry))
            
            y += STRIP_SPACING
            direction = 1 - direction
            
        return waypoints

    # -----------------------------------------
    # NAVIGATION LOGIC
    # -----------------------------------------
    def compute_heading(self, tx, ty):
        x, y = self.pose[0], self.pose[1]
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < self.target_precision:
            return True, 0.0, 0.0

        yaw = self.pose[5]
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        yaw_d = clamp(1.0 * yaw_err, -1.3, 1.3)


        if dist > 30: 
            pitch_d = -0.5 
        elif dist > 10: 
            pitch_d = -0.2
        else:           
            pitch_d = -0.1

        pitch_d = clamp(pitch_d, -0.5, 0.0)
        
        return False, yaw_d, pitch_d

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------
    def run(self):
        while self.step(self.ts) != -1:
            self.step_counter += 1
            
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x, y, z = self.gps.getValues()
            ra, pa, _ = self.gyro.getValues()
            self.pose = [x, y, z, roll, pitch, yaw]

            # --- RECEIVE MESSAGES & SET FORMATION ---
            while self.receiver.getQueueLength() > 0:
                msg = self.receiver.getString()
                self.receiver.nextPacket()
                if msg.startswith("FORMATION"):
                    parts = msg.split()
                    did, sx, sy = parts[1], float(parts[2]), float(parts[3])
                    
                    if int(did) == self.ID:
                        self.control_mode = "FORMATION"
                        if self.ID == 1:
                            self.fx, self.fy = sx, sy + 1.0
                        elif self.ID == 2:
                            self.fx, self.fy = sx, sy - 1.0
                        elif self.ID == 3:
                            self.fx, self.fy = sx - 1.0, sy
                        elif self.ID == 4:
                            self.fx, self.fy = sx + 1.0, sy
                        else:
                            self.fx, self.fy = sx, sy

            # --- VISION CHECK ---
            if self.control_mode == "SEARCH" and self.step_counter % 15 == 0:
                found = self.scan_for_target()
                if found:
                    print(f"[DRONE {self.ID}] !!! TARGET CAR FOUND !!!")
                    msg = f"FOUND {x:.2f} {y:.2f}"
                    self.emitter.send(msg.encode())

            # --- NAVIGATION UPDATE ---
            t = self.getTime()
            if t - self.last_nav_update > 0.2:
                
                # === TAKEOFF CHECK ===
                if not self.has_taken_off:
                    if z < self.target_alt - 0.5:
                        self.yaw_d = 0.0
                        self.pitch_d = 0.0
                    else:
                        self.has_taken_off = True
                        print(f"[DRONE {self.ID}] Reached {self.target_alt}m. Moving.")

                if self.has_taken_off:
                    if self.control_mode == "SEARCH":
                        tx, ty = self.search_wp[self.wp_index]
                        reached, self.yaw_d, self.pitch_d = self.compute_heading(tx, ty)
                        if reached:
                            self.wp_index = (self.wp_index + 1) % len(self.search_wp)
                    
                    elif self.control_mode == "FORMATION":
                        reached, self.yaw_d, self.pitch_d = self.compute_heading(self.fx, self.fy)
                        if reached:
                            self.target_alt = 3.0
                            self.yaw_d = 0.0
                            self.pitch_d = 0.0
                
                self.last_nav_update = t

            # --- MOTOR MIXING ---
            alt_err = clamp(self.target_alt - z + self.K_VERTICAL_OFFSET, -1.0, 1.0)
            vertical = self.K_VERTICAL_P * (alt_err ** 3)
            roll_input  = self.K_ROLL_P  * clamp(roll,  -1, 1) + ra
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pa + self.pitch_d
            yaw_input   = self.yaw_d

            fl = self.K_VERTICAL_THRUST + vertical - roll_input + pitch_input - yaw_input
            fr = self.K_VERTICAL_THRUST + vertical + roll_input + pitch_input + yaw_input
            rl = self.K_VERTICAL_THRUST + vertical - roll_input - pitch_input + yaw_input
            rr = self.K_VERTICAL_THRUST + vertical + roll_input - pitch_input - yaw_input

            self.fl.setVelocity(fl)
            self.fr.setVelocity(-fr)
            self.rl.setVelocity(-rl)
            self.rr.setVelocity(rr)
            
ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
Drone(ID).run()