from test_tf_setup import *

U_final = rechenknecht.propagation(U0, X_gate, params)


u_list = []
for i in range(0, len(out)):
    tmp = Qobj(out[i])
    u_list.append(tmp)

u_list2 = []
for i in range(0, len(out2)):
    tmp = Qobj(out2[i])
    u_list2.append(tmp)


pop1 = plot_dynamics(u_list, ts,[0])

# plt.plot(ts, pop1)
# plt.title("pwc_tf_1e4")
# plt.show()

pop2 = plot_dynamics(u_list2, ts, [0])

# plt.plot(ts, pop2)
# plt.title("pwc_tf_1e4")
# plt.show()



fig = plt.figure(1)
sp1 = plt.subplot(211)
name_str = "pwc_tf_%.2g" % n
sp1.title.set_text(name_str)
plt.plot(ts, pop1)

sp2 = plt.subplot(212)
name_str = "pwc_no_tf_%.2g" % n
sp2.title.set_text(name_str)
plt.plot(ts, pop2)

plt.show()
