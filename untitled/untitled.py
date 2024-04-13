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
        
class Muscle:
    def __init__(self):
        self.name = "muscle"
        
class Joint:
    def __init__(self):
        self.name = "joint"
        
