from test_tf_setup import *

U_final = rechenknecht.propagation(U0, X_gate, params)
gate_error = rechenknecht.gate_err(U0, X_gate, params)
gate_error_grad = tf.gradients(gate_error, params)

print("Computing fidelity error:")
print(
    sess.run(gate_error,
                       feed_dict={
                           params: X_gate.parameters['initial']
                           }
                       )
)
print("Computing gradients:")
print(
    sess.run(gate_error_grad,
                       feed_dict={
                           params: X_gate.parameters['initial']
                           }
                       )
)
