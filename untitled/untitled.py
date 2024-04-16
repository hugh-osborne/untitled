import numpy as np

class Bone:
    def __init__(self, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, name="bone", parent=None, mass=1.0, location_in_parent=np.zeros(3)):
        self.name = name
        self.parent = parent
        self.children = []
        self.origin_in_parent_space = location_in_parent
        self.mass = mass
        self.local_I = np.matrix([[I_xx, -I_xy, -I_xz],[-I_xy, I_yy, -I_yz],[-I_xz, -I_yz, I_zz]]) # Inertia Matrix
        self.local_I_i = np.linalg.inv(self.local_I) # Inverse Inertia Matrix
        self.x = np.zeros(0) # position
        self.R = np.identity(3) # orientation
        self.P = np.zeros(3) # linear momentum
        self.L = np.zeros(3) # angular momentum
        self.mat_I_i = np.identity(3)
        self.v = np.zeros(3) # linear velocity
        self.omega = np.zeros(3) # angular velocity
        self.force = np.zeros(3) # net linear force
        self.torque = np.zeros(3) # new angular force
        
        # Each force is defined with a (force vector, point of contact)
        self.forces = []
        # Linear forces impart no spin and act through the centre of mass (like gravity)
        # so are defined only with a force vector
        self.linear_forces = []
        
    def addForce(self, force, contact):
        self.forces += [(force, contact)]
        
    def addLinearForce(self, force):
        self.linear_forces += [force]
        
    def calculate_v(self):
        self.v = self.P / self.mass
        
    def calculate_mat_I_i(self):
        self.mat_I_i = self.R * self.local_I_i * np.transpose(self.R)
        
    def calculate_omega(self):
        self.omega = self.mat_I_i * self.L
        
    def calculateNetForceTorque(self):
        self.force = np.sum(self.linear_forces) + np.sum([f[0] for f in self.forces]) # linear force is just the sum of all forces
        self.torque = np.sum([np.cross(f[1] - self.x, f[0]) for f in self.forces]) # torques are calculated with the cross product
        
    def euler(self, dt):
        # update the position
        self.x += dt*self.v
        
        # update the orientation
        R_prime = np.matrix([[0,-self.omega[2],self.omega[1]],[self.omega[2], 0, -self.omega[0]],[-self.omega[1], self.omega[0], 0]]) * self.R
        self.R += dt * R_prime
        
        # update linear momentum
        self.P += dt * self.force
        
        # update angular momentum
        self.L += dt * self.torque

        # Recalculate Forces etc
        self.calculateNetForceTorque()
        self.calaculate_v()
        self.calculate_mat_I_i()
        self.calculate_omega()
           
        
        
class Muscle:
    def __init__(self):
        self.name = "muscle"
        
class Joint:
    def __init__(self):
        self.name = "joint"
        
