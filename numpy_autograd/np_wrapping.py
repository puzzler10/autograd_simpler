#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as _np
import types 
from functools import wraps

def primitive(f, keepgrad=True): 
    @wraps(f)
    def inner(*args, **kwargs):
        ## Code to add operation/primitive to computation graph
        # We need to separate out the integer/non node case. Sometimes you are adding 
        # constants to nodes. 
        def getval(o):      return o.value if type(o) == Node else o
        if len(args):       argvals = [getval(o) for o in args]
        else:               argvals = args
        if len(kwargs):     kwargvals = dict([(k,getval(o)) for k,o in kwargs.items()])
        else:               kwargvals =  kwargs
         
        # get parents 
        l = list(args) + list(kwargs.values())
        parents = [o for o in l if type(o) == Node ]
        
        value = f(*argvals, **kwargvals)
        print("add", "'" + f.__name__ + "'", "to graph with value",value)
        return Node(value, f, parents, keepgrad)
    return inner

# def notrace_primitive(f): 
#     """Doesn't really do much. Evaluates functions, doesn't add them to the computation graph. """
#     @wraps(f)
#     def inner(*args, **kwargs): 
#         def getval(o):      return o.value if type(o) == Node else o
#         if len(args):       argvals = [getval(o) for o in args]
#         else:               argvals = args
#         if len(kwargs):     kwargvals = dict([(k,getval(o)) for k,o in kwargs.items()])
#         else:               kwargvals =  kwargs
            
#         return f(*argvals, **kwargvals)
#     return inner


# In[10]:


class Node:
    """A node in a computation graph."""
    def __init__(self, value, fun, parents, keepgrad):
        self.parents = parents
        self.value = value
        self.fun = fun 
        self.keepgrad = keepgrad
        
    def __repr__(self): 
        """A (very) basic string representation"""
        if self.value is None: str_val = 'None'
        else:                  str_val = str(self.value)
        return   "\n" + "Fun: " + str(self.fun) +                " Value: "+ str_val +                 " Parents: " + str(self.parents) 
    
    def start_node(value = None, keepgrad=True): 
        """A function to create an empty node to start off the graph"""
        fun,parents = lambda x: x, []
        return Node(value, fun, parents, keepgrad=True)
    
    


# In[4]:


# class Node:     
#     def __init__(self, value, fun,  parents): 
#         self.value,self.fun,self.parents = value,fun,parents
    
#     @staticmethod
#     def new_root(value = None): 
#         fun,parents = lambda x: x, []
#         return Node(value, fun, parents)
    
#     def __repr__(self): 
#         if self.value is None: str_val = 'None'
#         else:                  str_val = str(round(self.recipe[1],3))
#         return "\n" +  "\t" + "Fun: " + str(self.recipe[0])+\
#                 " Value: "+ str_val + \
#                 " Parents: " + str(self.parents)  
#     #(Node.curr_max -self.depth) *

    
#     ## Overwrite operators to use our functions 
#     # Don't put self.value or other.value in the arguments of these functions, 
#     # otherwise you won't be able to access the Node object to create the 
#     # computational graph. 
#     # Instead, pass the whole node through. And to prevent recursion errors, 
#     # extract the value inside the `primitive` function. 
# #     def __add__(self, other): return add_new(self, other)
# #     def __radd__(self, other): return add_new(other, self)
# #     def __sub__(self, other): return sub_new(self, other)
# #     def __rsub__(self, other): return sub_new(other, self)
# #     def __truediv__(self, other): return div_new(self, other)
# #     def __rtruediv__(self, other): return div_new(other, self)
# #     def __mul__(self, other): return mul_new(self, other)
# #     def __rmul__(self, other): return mul_new(other, self)
# #     def __neg__(self): return neg_new(self)
# #     def __exp__(self): return exp_new(self)
    


# In[5]:


def wrap_namespace(old, new):
    """Performs triage on objects from numpy, copying them from old to new namespace. 
       old: __dict__ from original numpy
       new: dict to copy old into 
       """
    # Taken from here: 
    # https://github.com/mattjj/autodidact/blob/b3b6e0c16863e6c7750b0fc067076c51f34fe271/autograd/numpy/numpy_wrapper.py#L8 
    nograd_functions = [
        _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type, _np.zeros_like,
        _np.ones_like, _np.floor, _np.ceil, _np.round, _np.rint, _np.around,
        _np.fix, _np.trunc, _np.all, _np.any, _np.argmax, _np.argmin,
        _np.argpartition, _np.argsort, _np.argwhere, _np.nonzero, _np.flatnonzero,
        _np.count_nonzero, _np.searchsorted, _np.sign, _np.ndim, _np.shape,
        _np.floor_divide, _np.logical_and, _np.logical_or, _np.logical_not,
        _np.logical_xor, _np.isfinite, _np.isinf, _np.isnan, _np.isneginf,
        _np.isposinf, _np.allclose, _np.isclose, _np.array_equal, _np.array_equiv,
        _np.greater, _np.greater_equal, _np.less, _np.less_equal, _np.equal,
        _np.not_equal, _np.iscomplexobj, _np.iscomplex, _np.size, _np.isscalar,
        _np.isreal, _np.zeros_like, _np.ones_like, _np.result_type
    ]
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name,obj in old.items(): 
        if obj in nograd_functions:  
            # non-differentiable functions 
            new[name] = primitive(obj, keepgrad=False)
        elif type(obj) in function_types:  # functions with gradients 
            # differentiable functions
            new[name] = primitive(obj)
        else: 
            # just copy over 
            new[name] = obj
        


# In[6]:


# using globals() here means we can access each np function like np.add: 
# it means it is available to the global space. 
anp = globals()
wrap_namespace(_np.__dict__, anp)


# In[7]:


# Constants w.r.t float data just pass though
setattr(Node, 'ndim', property(lambda self: self.value.ndim))
setattr(Node, 'size', property(lambda self: self.value.size))
setattr(Node, 'dtype',property(lambda self: self.value.dtype))
setattr(Node, 'T', property(lambda self: anp['transpose'](self)))
setattr(Node, 'shape', property(lambda self: self.value.shape))
setattr(Node,'__len__', lambda self, other: len(self._value))
setattr(Node,'astype', lambda self,*args,**kwargs: anp['_astype'](self, *args, **kwargs))
setattr(Node,'__neg__', lambda self: anp['negative'](self))
setattr(Node,'__add__', lambda self, other: anp['add'](     self, other))
setattr(Node,'__sub__', lambda self, other: anp['subtract'](self, other))
setattr(Node,'__mul__', lambda self, other: anp['multiply'](self, other))
setattr(Node,'__pow__', lambda self, other: anp['power'](self, other))
setattr(Node,'__div__', lambda self, other: anp['divide'](  self, other))
setattr(Node,'__mod__', lambda self, other: anp['mod'](     self, other))
setattr(Node,'__truediv__', lambda self, other: anp['true_divide'](self, other))
setattr(Node,'__matmul__', lambda self, other: anp['matmul'](self, other))
setattr(Node,'__radd__', lambda self, other: anp['add'](     other, self))
setattr(Node,'__rsub__', lambda self, other: anp['subtract'](other, self))
setattr(Node,'__rmul__', lambda self, other: anp['multiply'](other, self))
setattr(Node,'__rpow__', lambda self, other: anp['power'](   other, self))
setattr(Node,'__rdiv__', lambda self, other: anp['divide'](  other, self))
setattr(Node,'__rmod__', lambda self, other: anp['mod'](     other, self))
setattr(Node,'__rtruediv__', lambda self, other: anp['true_divide'](other, self))
setattr(Node,'__rmatmul__', lambda self, other: anp['matmul'](other, self))
setattr(Node,'__eq__', lambda self, other: anp['equal'](self, other))
setattr(Node,'__ne__', lambda self, other: anp['not_equal'](self, other))
setattr(Node,'__gt__', lambda self, other: anp['greater'](self, other))
setattr(Node,'__ge__', lambda self, other: anp['greater_equal'](self, other))
setattr(Node,'__lt__', lambda self, other: anp['less'](self, other))
setattr(Node,'__le__', lambda self, other: anp['less_equal'](self, other))
setattr(Node,'__abs__', lambda self: anp['abs'](self))
setattr(Node,'__hash__', lambda self: id(self))


# In[13]:


# convert to python script 
#!jupyter nbconvert --to script np_wrapping.ipynb


# In[ ]:




