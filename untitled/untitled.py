import numpy as np

class Bone:
    def __init__(self):
        self.name = "bone"
        self.parent = None
        self.children = []
        self.length = 1.0
        # origin of all muscles is 0,0,0. Bones rotate around this point. 
        # The location of the pivot point (origin) in the parent space is stored.
        self.origin_in_parent_space = np.zeros(3)
        self.rotation = np.zeros(3)
        self.mass = 1.0
        self.inertia = np.zeros(3)
        self.rot_acc = np.zeros(3)
        self.rot_vel = np.zeros(3)
        
    def update(self):
        
        self.inertia = (self.mass * (self.length**2))/3.0
        torque = 9.81*self.length # torque = Fr
        ang_acc = torque / self.inertia
        
        
        
class Muscle:
    def __init__(self):
        self.name = "muscle"
        
class Joint:
    def __init__(self):
        self.name = "joint"
        
