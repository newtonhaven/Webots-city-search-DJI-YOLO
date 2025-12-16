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
                    sx = parts[1]
                    sy = parts[2]
                    
                    print(f"Supervisor received FOUND at: {sx}, {sy}")
                    print("Broadcasting FORMATION command to all drones.")

                    # Command all 4 drones to converge on that location
                    for d in range(1, 5):
                        command = f"FORMATION {d} {sx} {sy}"
                        self.emitter.send(command.encode())

controller = CitySupervisor()
controller.run()