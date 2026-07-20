from femtotorch.backend import xp # xp is the convention for both numpy and cupy compatibility


class Node:
    """
    Minimal class to save the necessary informations related to one tensor to be able to backprogate.
    It's the object that stays in RAM until its gradients are passed in the backprobation.
    """

    def __init__(self, function, inputs):
        self.function = function # the operation that generated the data of of the consumer
        self.inputs = inputs     # the inputs to which the gradient will be backpassed
        self.saved = ()          # raw arrays or values stashed for backward; never Tensors (to avoid storing unnecessary informations)

    def save(self, *values, **parameters):
        self.saved = values # tuple of values


# helper function to allow backward pass on operations with numpy broadcasting
def unbroadcast(outGrad, shape):
    """
    Sum over outGrad axis backdown shape of the broadcasted tensor.
    """

    if shape is None: # nothing to unbroadcast
        return outGrad 

    # handle the case of an operation between arrays of different dimensional space
    # typically a bias vector (n,)
    while outGrad.ndim > len(shape):
        outGrad = outGrad.sum(axis=0)
    # handle the case of different dimensional object in the same space
    # typically a bias vector (1, n)
    for i, dim in enumerate(shape):
        if dim == 1:
            outGrad = outGrad.sum(axis=i, keepdims=True)

    return outGrad # Returns the unbroadcasted gradient array to use in the chain rule


# helper function to allow backward pass on reduction operations (sum, max, mean)
def broadcast_back(grad, shape, axis, keepdims):
    """
    Expand a reduced gradient back to 'shape'.
    Reduction operations and broadcast are transposes of each other;
    broadcast_back is unbroadcast, reversed
    """
    if axis is not None and not keepdims:
        grad = xp.expand_dims(grad, axis) # reinsert the collapsed slot(s) first
    return xp.broadcast_to(grad, shape)

# Construction of the computation graph and gradient descent
def graph_backward(root_graph):

    # Build topological (oldest node to youngest) ordering of all nodes in the computation graph
    topo = []
    visited = set()

    def build_topo(v):

        if id(v) not in visited:
            visited.add(id(v))

            if v.grad_node is not None: # if v.grad_none is None it's a leaf Node

                for child in v.grad_node.inputs:
                    build_topo(child)

            topo.append(v)

    build_topo(root_graph)

    root_graph.grad = xp.ones_like(root_graph.data) # base case of the recurrence dL/dL = array of ones
    
    # backpropagation
    for v in reversed(topo): # consumers of t before t, so t.grad is conmplete once visited

        if v.grad_node is not None: # if v.grad_none is None it's a leaf Node so there's no backpass to do 
            grad_node = v.grad_node # saved infos about the relations with and between the input nodes
            grads = grad_node.function.backward(grad_node, v.grad) # compute the gradient that is going to be backpassed
            
            # update the gradient of each input nodes
            for input_tensor, g in zip(grad_node.inputs, grads):
                input_tensor._accumulate_grad(g) 

            # once the gradients of v are backpassed no need to keep infos about them until the end of the current backpropagation
            v.grad = None 
            v.grad_node = None
