from test_tf_setup import *

#sess = tf_debug.LocalCLIDebugWrapperSession(sess) # Enable this to debug
params = tf.placeholder(
    tf.float64,
    shape=X_gate.parameters['initial'].shape
    )


U_final = rechenknecht.propagation(U0, X_gate, params)
gate_error = rechenknecht.gate_err(U0, X_gate, params)
gate_error_grad = tf.gradients(gate_error, params)

print("Propagating to U_final with Tensorflow:")
sess.run(U_final,
                   feed_dict={
                       params: X_gate.parameters['initial']
                       }
                   )
