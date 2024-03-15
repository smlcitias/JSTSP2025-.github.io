import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import binom

def getPrior(a, b):
  #x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
  x = np.linspace(0.0, 1.0, 200)
  y = beta.pdf(x, a, b)
  return x, y

def getPosterior(a, b, n, c):
  pa = n + a
  pb = c - n + b
  #px = np.linspace(beta.ppf(0.01, pa, pb), beta.ppf(0.99, pa, pb), 100)
  px = np.linspace(0.0, 1.0, 200)
  py = beta.pdf(px, pa, pb)
  return px, py

def getLikelihood(n, c):
  lx = np.linspace(0.0, 1.0, 200)
  ly = binom.pmf(n, c, lx)
  return lx, ly


a = 20.0
b = 20.0
n = 4
c = 10

x1, y1 = getPrior(a, b)
l1 = 'Prior = Beta(%.1f, %.1f)' % (a, b)
x2, y2 = getPosterior(a, b, n, c)
l2 = 'Posterior = Beta(%d+%.1f, %d-%d+%.1f)' % (n, a, c, n, b)
#x3, y3 = getLikelihood(n, c)

plt.figure(figsize=(7, 7))
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 7.0)

plt.plot(x1, y1, 'red')
plt.plot(x2, y2, 'blue')
#plt.plot(x3, y3, 'green')

plt.legend([l1, l2], loc='upper left')
plt.text(0.6, 3.0, r'$p(\mathbf{X}|\theta)$', color='blue',fontsize='15')
plt.text(0.7, 2.0, r'$p(\theta)$', color='red',fontsize='15')

plt.title('With informative prior', fontsize='15')
plt.xlabel(r'$\theta$', fontsize='15')
#plt.ylabel('Probability', fontsize='15')

#plt.show()
plt.savefig('Beta2.pdf', format='pdf', bbox_inches='tight')


