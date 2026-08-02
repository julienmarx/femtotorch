from femtotorch.backend import xp # xp is the convention for both numpy and cupy compatibility


class Node:
    """
    Minimal class to save the necessary informations related to one tensor to be able to backprogate.
    It's the object that stays in RAM until its gradients are passed to Tensors in the backprobation.

    In the comments 'consumer' refers to the output of input nodes and an operation during the forward pass.
    """

    def __init__(self, function, inputs):
        self.function = function # the operation that generated the data of of the consumer 
        self.inputs = inputs     # the inputs to which the gradient will be backpassed

        # These are raw arrays or values thare are stashed for backward;
        # never Tensors (to avoid storing unnecessary informations).
        # These raw arrays/ values never have outgoing references to Node or Tensors,
        # so there can't be circular reference by design
        self.saved = () 

    def save(self, *values):
        self.saved = values # tuple of values


# helper function to allow backward pass on operations with numpy broadcasting
def unbroadcast(outGrad, shape):
    """
    Sum over outGrad axes to have the input 'shape' of the broadcasted tensor.
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
    broadcast_back is unbroadcast but reversed
    """
    if axis is not None and not keepdims:
        grad = xp.expand_dims(grad, axis) # reinsert the collapsed slots first
    return xp.broadcast_to(grad, shape)

# Construction of the computation graph and gradient descent
def graph_backward_recursive(root_graph):
    """
    Historic version using recurrence which borrows the interpreter's call stack as its data structure,
    """

    # Build topological (oldest node to youngest) ordering of all nodes in the computation graph
    topo = []
    visited = set()

    def build_topo(v):

        if id(v) not in visited:
            visited.add(id(v))

            if v.grad_node is not None: # if v.grad_node is None it's a leaf Node

                for child in v.grad_node.inputs:
                    build_topo(child)

            topo.append(v)

    build_topo(root_graph)

    root_graph.grad = xp.ones_like(root_graph.data) # base case of the recurrence dL/dL = array of ones
    
    # backpropagation
    for v in reversed(topo): # consumers of t before t, so t.grad is complete once visited

        if v.grad_node is not None: # if v.grad_node is None it's a leaf Node so there's no backpass to do 
            grad_node = v.grad_node # saved infos about the relations with and between the input nodes
            grads = grad_node.function.backward(grad_node, v.grad) # compute the gradient that is going to be backpassed
            
            # update the gradient of each input nodes
            for input_tensor, g in zip(grad_node.inputs, grads):
                input_tensor._accumulate_grad(g) 

            # once the gradients of v are backpassed no need to keep infos about them until the end of the current backpropagation
            v.grad = None 
            v.grad_node = None

def graph_backward(root_graph):
    """
    visited : tag to know if the node has been added to topo list

    """
    # Iterative version to avoid cycle reference build_topo -> closure -> build_topo
    # Build topological (oldest node to youngest) ordering of all nodes in the computation graph
    topo = []
    visited = set()
    stack = [(root_graph, False)]
    while stack : # while stack is not empty (so until (root_graph, False) or (root_graph, True) is in it)
        v, children_are_revealed = stack.pop()

        if children_are_revealed:
            topo.append(v)
            # and we dont append back v to the stack

        elif id(v) not in visited:
            visited.add(id(v))
            stack.append((v, True)) # tag as node with children discovered
            if v.grad_node is not None:
                for child in v.grad_node.inputs:
                    stack.append((child, False))

    root_graph.grad = xp.ones_like(root_graph.data) # base case of the recurrence dL/dL = array of ones
    
    # backpropagation
    for v in reversed(topo): # consumers of t before t, so t.grad is complete once visited

        if v.grad_node is not None: # if v.grad_node is None it's a leaf Node so there's no backpass to do 
            grad_node = v.grad_node # saved infos about the relations with and between the input nodes
            grads = grad_node.function.backward(grad_node, v.grad) # compute the gradient that is going to be backpassed
            
            # update the gradient of each input nodes
            for input_tensor, g in zip(grad_node.inputs, grads):
                input_tensor._accumulate_grad(g) 

            # once the gradients of v are backpassed no need to keep infos about them until the end of the current backpropagation
            v.grad = None 
            v.grad_node = None