from test_tf_setup import *

params = tf.placeholder(
    tf.float64,
    shape=X_gate.parameters['initial'].shape
    )

def plot_dynamics(u_list, ts, states):
    pop = []
    for si in states:
        for ti in range(len(ts)):
            pop.append(abs(u_list[ti][si][0] ** 2))
#        plt.plot(ts, pop)
    return pop

U_of_t, ts = rechenknecht.propagation(U0, X_gate, params, history=True)

print("Propagating U(t) with Tensorflow:")
out = sess.run(U_of_t,
                   feed_dict={
                       params: X_gate.parameters['initial']
                       }
                   )

Ts = sess.run(ts,
                   feed_dict={
                       params: X_gate.parameters['initial']
                       }
                   )


u_list = []
for i in range(0, len(out)):
    tmp = Qobj(out[i])
    u_list.append(tmp)

pop1 = plot_dynamics(u_list, Ts,[0])

# plt.plot(ts, pop1)
# plt.title("pwc_tf_1e4")
# plt.show()

fig = plt.figure(1)
sp1 = plt.subplot(211)
plt.plot(Ts, pop1)

plt.show()
