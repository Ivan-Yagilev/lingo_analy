import matplotlib.pyplot as plt
plt.style.use('ggplot')

loss = [9.270808, 0.731134, 0.166938, 0.060590, 0.013362, 0.011553, 0.009126, 0.008366, 0.007208, 0.006062]

x = [i for i in range(len(loss))]

plt.plot(x, loss)
ax = plt.gca()
ax.set_ylabel("Loss")
plt.savefig("mygraph.png")