import folium
import geopandas as gpd
from geopandas import GeoSeries

from matplotlib.colors import to_rgba, to_hex
from shapely.geometry import LineString
from shapely.geometry import Point

from collections.abc import MutableMapping

class Arm():
    def __init__(self, label, arm_set):
        self.label = label
        self.context = None
        self.arm_set = arm_set

    def __getitem__(self, key):
        a = self.arm_set[key]
        return self.arm_set._arm_weight_fun(a)
    
    def set_context(self, context):
        self.context = context

    def set_reward(self, reward):
        self.reward = reward

class ArmDict(MutableMapping):
    
    def __init__(self):
        self.arm_set = dict()
        self.set_arm_weight_function()

    def __setitem__(self, key, val):
        self.arm_set[key] = val
    
    def __getitem__(self, key):
        return self.arm_set[key]
            
    def __delitem__(self, key):
        del self.arm_set[key]
            
    def __iter__(self):
        return iter(self.arm_set)
        
    def __len__(self):
        return len(self.arm_set)
        
    def __repr__(self):
        return '{}, ArmDict({})'.format( \
               super(ArmDict, self).__repr__(), self.arm_set)
            
    def set_arm_weight_function(self, fun=None):
        if fun is None:
            self._arm_weight_fun = lambda e : e.weight
        elif callable(fun):
            self._arm_weight_fun = fun
        else:
            raise ValueError("Arm weight function must be " + 
                             "'weight' or a function!")
    
    def get_arm_weight_function(self):
        return self._arm_weight_fun
