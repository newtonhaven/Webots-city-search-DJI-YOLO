from controller import Supervisor

class CitySupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.ts = int(self.getBasicTimeStep())

        self.emitter = self.getDevice("emitter_to_drones")
        self.receiver = self.getDevice("receiver_from_drones")
        self.receiver.enable(self.ts)

        print("Supervisor initialized.")
        print("Standing by for 'FOUND' signals...")

        # --- TRAIL DRAWING SETUP ---
        self.root = self.getRoot()
        self.root_children = self.root.getField("children")
        
        # Color map for drones
        self.colors = {
            '1': "1 0 0", # Red
            '2': "0 0 1", # Blue
            '3': "0 1 0", # Green
            '4': "1 1 0"  # Yellow
        }
        self.drone_last_pos = {}

    # -----------------------------------------
    # RUN LOOP
    # -----------------------------------------
    def run(self):
        while self.step(self.ts) != -1:
            # Check for incoming messages from drones
            while self.receiver.getQueueLength() > 0:
                msg = self.receiver.getString()
                self.receiver.nextPacket()

                # If a drone finds the target
                if msg.startswith("FOUND"):
                    # Parse the location sent by the drone
                    parts = msg.split()
                    sx = str(float(parts[1])+20.0)
                    sy = str(float(parts[2])-29.0)
                    
                    print(f"Supervisor received FOUND at: {sx}, {sy}")
                    print("Broadcasting FORMATION command to all drones.")

                    # Command all 4 drones to converge on that location
                    for d in range(1, 5):
                        command = f"FORMATION {d} {sx} {sy}"
                        self.emitter.send(command.encode())
                
                elif msg.startswith("TRAIL"):
                    # Format: TRAIL <ID> <X> <Y> <Z>
                    try:
                        parts = msg.split()
                        did = parts[1]
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        
                        if did in self.drone_last_pos:
                            lx, ly, lz = self.drone_last_pos[did]
                            
                            # Draw Segment
                            color = self.colors.get(did, "1 1 1") # Default white
                            
                            # Create a LineSet Shape
                            shape_str = (
                                f"Shape {{ "
                                f"  appearance Appearance {{ material Material {{ diffuseColor {color} emissiveColor {color} }} }} "
                                f"  geometry IndexedLineSet {{ "
                                f"    coord Coordinate {{ point [ {lx} {ly} {lz}, {x} {y} {z} ] }} "
                                f"    coordIndex [ 0, 1, -1 ] "
                                f"  }} "
                                f"}}"
                            )
                            self.root_children.importMFNodeFromString(-1, shape_str)
                            
                        # Update last pos
                        self.drone_last_pos[did] = (x, y, z)
                    except Exception as e:
                        print(f"Supervisor TRAIL error: {e}")

controller = CitySupervisor()
controller.run()