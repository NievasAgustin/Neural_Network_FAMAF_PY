import numpy as np
import matplotlib.pyplot as plt


N = 10
x = np.linspace(0, 1, N)

r = 0.1*np.random.normal(0, 0.3, N)
y = x**3 + x + 1 + r

def model(x,par):
    return par[0]*x**3 + par[1]*x + par[2]

    
ye = model(x,[1,1,1]) #ye = x**3 + x + 1
ym = model(x,[0.5,1,1]) #ym = x**3/2 x + 1 Probably wrong


plt.hist(r)
plt.show()

plt.scatter(x, y)
plt.plot(x, ye, color='r')
plt.plot(x, ym, color='g')
plt.show()

plt.scatter(x, np.abs(y-ye), label="y-ye", alpha=0.5)
plt.scatter(x, np.abs(y-ym), label="y-ym", alpha=0.5)
for i in range(0, len(x)):
    plt.vlines(x[i], 0, np.abs(y[i]-ye[i]), color='blue', alpha=0.5, lw=3)
    plt.vlines(x[i], 0, np.abs(y[i]-ym[i]), color='orange', alpha=0.5, linestyle='--', lw=3)

plt.title('Errors')
plt.legend()
plt.show()


def mse(y, y_hat):
    return np.sum((y-y_hat)**2)/len(y)

print("MSE(y, ye) = ", mse(y, ye))
print("MSE(y, ym) = ", mse(y, ym))


par_0 = np.linspace(0, 2, 100)

mse_vals = np.zeros(len(par_0))

for i in range(len(par_0)):
    mse_vals[i] = mse(y, model(x,[par_0[i],1,1]))

plt.plot(par_0, mse_vals, label="MSE(y, model)")


min_val = np.min(mse_vals)
min_idx = np.argmin(mse_vals)
print("Minimum value of MSE(y, model) = ", min_val)
print("Minimum value of par_0 = ", par_0[min_idx])


y_alt = y.copy()
y_alt[4] = 5

plt.scatter(x, y_alt, s=15, label="y_alt")
plt.scatter(x, y, s=5, label="y")
plt.show()


mse_vals_alt = np.zeros(len(par_0))
for i in range(len(par_0)):
    mse_vals_alt[i] = mse(y_alt, model(x,[par_0[i],1,1]))

plt.plot(par_0, mse_vals_alt, label="MSE(y_alt, model)")



min_val = np.min(mse_vals_alt)
min_idx = np.argmin(mse_vals_alt)
print("Minimum value of MSE(y, model) = ", min_val)
print("Minimum value of par_0 = ", par_0[min_idx])



def mae(y, y_hat):
    return np.sum(np.abs(y-y_hat))/len(y) 

print("MAE(y, ye) = ", mae(y, ye))  



mae_vals = np.zeros(len(par_0))

for i in range(len(par_0)):
    mae_vals[i] = mae(y, model(x,[par_0[i],1,1]))

plt.plot(par_0, mae_vals, label="MAE(y, model)")
plt.plot(par_0, mse_vals, label="MSE(y, model)")
plt.legend()
plt.show()


mae_vals_alt = np.zeros(len(par_0))
for i in range(len(par_0)):
    mae_vals_alt[i] = mae(y_alt, model(x,[par_0[i],1,1]))

plt.plot(par_0, mse_vals_alt, label="MSE(y_alt, model)")
plt.plot(par_0, mae_vals_alt, label="MAE(y_alt, model)")
plt.legend()
plt.show()


def huber_loss(y, y_hat, delta=1.0):
    huber_mse = 0.5*(y-y_hat)**2
    huber_mae = delta * (np.abs(y - y_hat) - 0.5 * (delta**2))
    return np.where(np.abs(y - y_hat) <= delta, huber_mse, huber_mae)


huber_loss_vals = np.zeros(len(par_0))
for i in range(len(par_0)):
    huber_loss_vals[i] = np.sum(huber_loss(y_alt, model(x,[par_0[i],1,1])))/len(y_alt)

plt.plot(par_0, huber_loss_vals, label="Huber loss")
plt.plot(par_0, mse_vals_alt, label="MSE(y_alt, model)")
plt.plot(par_0, mae_vals_alt, label="MAE(y_alt, model)")
plt.legend()
plt.show()