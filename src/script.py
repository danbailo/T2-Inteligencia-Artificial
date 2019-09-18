import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True
# plt.rcParams['text.usetex'] = True

def linear_regression(num_it, alpha, theta):
    J = np.zeros(num_it)
    for i in range(num_it):
        H = np.dot(X,theta)
        E = H-y
        J[i] = (np.dot(E.T,E)/(2*m))
        theta = theta - ((alpha/m)*np.dot(X.T,E))
    return J, theta

def normalize(X):
    return np.divide(X-np.mean(X, axis=0), np.std(X, axis=0))

if __name__ == "__main__":
	wine_red = pd.read_csv('../inputs/winequality-red.csv', sep=';')
	x = wine_red[wine_red.columns[:-1]].to_numpy()
	y = wine_red[wine_red.columns[-1:]].to_numpy()
	X = normalize(x)
	X = np.insert(X, obj=0, values=1, axis=1)
	m = len(wine_red.columns[:-1])

	initial_theta = np.zeros(m+1).reshape(m+1,1) 


	NUM_IT = 100
	plt.figure()
	for alpha in np.arange(0.001, 0.01, 0.001):
		J, _ = linear_regression(NUM_IT, alpha, initial_theta)
		plt.plot(range(NUM_IT), J, label=fr'$\alpha$ = {alpha}')
	plt.axis(ymin=25, ymax=10e3)
	plt.yscale('log')
	plt.title('Plotando valores de J para o intervalo [0.001, 0.01)')
	plt.xlabel('Iterações')
	plt.ylabel(r'J($\theta$)')
	plt.legend()
	plt.savefig(os.path.join('..','imgs/')+'first_loss.pdf', bbox_inches='tight')

	NUM_IT = 100
	plt.figure()
	for alpha in np.arange(0.001, 0.01, 0.001):
		J, _ = linear_regression(NUM_IT, alpha, initial_theta)
		if max(J) > 10e4: continue
		else:
			plt.plot(range(NUM_IT), J, label=fr'$\alpha$ = {alpha}')
	plt.axis(ymin=25, ymax=10e3)
	plt.title('Valores que estouraram foram removidos')
	plt.yscale('log')
	plt.xlabel('Iterações')
	plt.ylabel(r'J($\theta$)')
	plt.legend()	
	plt.savefig(os.path.join('..','imgs/')+'without_exp_loss.pdf', bbox_inches='tight')

	 
	NUM_IT = 1000
	alpha = 0.004
	plt.figure()
	J, new_theta = linear_regression(NUM_IT, alpha, initial_theta)
	plt.plot(range(NUM_IT), J, label=fr'$\alpha$ = {alpha}')
	plt.title(f'Alpha definido como {alpha} para {NUM_IT} iterações.')
	plt.xlabel('Iterações')
	plt.ylabel(r'J($\theta$)')
	plt.legend()
	plt.savefig(os.path.join('..','imgs/')+'final.pdf', bbox_inches='tight')
	
	plt.show()