#!/usr/bin/env python
import tensorflow as tf
import math; log2 = lambda x: math.log(x)/math.log(2.0)

# Model definition and parameters: note non-complex types
# H = sum_{i=1..N} s_z(i) + J s_x(i) s_x(i+1)
N = 30
J = 0.1

internal_bond_dim = 100
tensor_precision = tf.float64
log = "output"

# Pauli basis matrices
zero = [[ 0, 0],[ 0, 0]]
one  = [[ 1, 0],[ 0, 1]]
sz   = [[ 1, 0],[ 0,-1]]
sx   = [[ 0, 1],[ 1, 0]]
# jsx = J*sx
jsx  = [[ 0, J],[ J, 0]]

# MPO tensor
t_mpo_bulk = tf.constant([
    [  one,  jsx,   sz],
    [ zero, zero,   sx],
    [ zero, zero,  one],
], dtype = tensor_precision, name = "H")
assert t_mpo_bulk.shape == (3,3,2,2)

# MPO fixed boundaries tensors
t_mpo_l = tf.constant([1,0,0], dtype = tensor_precision, name = "trunc_l")
t_mpo_r = tf.constant([0,0,1], dtype = tensor_precision, name = "trunc_r")
assert t_mpo_l.shape == t_mpo_r.shape == (3,)

# MPS tensors
def bond_dim(i):
    m = log2(internal_bond_dim)
    left = min(1.0*N/2, m-1)
    right = max(1.0*N/2, N-1-m)
    return 2**(i+1) if i<left else 2**(N-i-1) if i>right else internal_bond_dim
assert bond_dim(-1) == bond_dim(N-1) == 1
assert bond_dim(N/2) <= internal_bond_dim

t_mps_array = list(
    tf.Variable(
        tf.random_uniform(
            (bond_dim(i-1),2,bond_dim(i)),
            maxval = 1.0/min(bond_dim(i-1),bond_dim(i)),
            dtype = tensor_precision,
        ),
        name = "psi_"+str(i),
    ) for i in range(N)
)

# Calculate norm
t_norm = tf.constant([[1.0]], dtype = tensor_precision)
for i, t_mps in enumerate(t_mps_array):
    t_norm = tf.tensordot(t_norm,t_mps,[[0],[0]]) # bra
    t_norm = tf.tensordot(t_norm,t_mps,[[0,1],[0,1]]) # ket
t_norm = tf.squeeze(t_norm)
assert t_norm.shape == tuple()

# Calculate value
t_value = tf.reshape(t_mpo_l, (1,3,1))
for i, t_mps in enumerate(t_mps_array):
    t_value = tf.tensordot(t_value,t_mps,[[0],[0]]) # bra
    t_value = tf.tensordot(t_value,t_mpo_bulk,[[0,2],[0,2]]) # H
    t_value = tf.tensordot(t_value,t_mps,[[0,3],[0,1]]) # ket
t_value = tf.squeeze(tf.tensordot(t_value,t_mpo_r,[[1],[0]]))
assert t_value.shape == tuple()
    
# Calculate energy
t_energy = t_value / t_norm

# Initialize variables
op_init_model = tf.global_variables_initializer()

# Normalize MPS
t_n = t_norm**(1./2/N)
op_norm = list(tf.assign(i,i/t_n) for i in t_mps_array)

# Execute TF
with tf.Session() as sess:
    if not log is None:
        writer = tf.summary.FileWriter(log, sess.graph)
    sess.run(op_init_model)
    sess.run(op_norm)
    v,n,e = sess.run([t_value, t_norm, t_energy])
    print("Init:\t{:.8e} = {:.8e} / {:.8e}".format(e,v,n))
    tf.contrib.opt.ScipyOptimizerInterface(
        t_energy,
        #method='CG',
        options = dict(disp = True),
    ).minimize(sess)
    v,n,e = sess.run([t_value, t_norm, t_energy])
    print("Result:\t{:.8e} = {:.8e} / {:.8e}".format(e,v,n))
    if not log is None:
        writer.close()
