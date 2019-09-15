from __future__ import print_function
a1, b1, c1 = 10, 172, -3
from math import sqrt
from math import exp
def gcd(a, b):
	if b > a:
		return gcd(b, a)
	if a%b == 0:
		return b
	else:
		return gcd(b, a%b)
def tovect(begin, end):
	vect = []
	for el in zip(begin, end):
		vect.append(el[1] - el[0])
	return vect

def summ(ls1, ls2):
	ls = []
	for el in zip(ls1, ls2):
		ls.append(el[0] + el[1])
	return ls

def mul(n, ls1):
	ls2 = []
	for el in ls1:
		ls2.append(n*el)
	return ls2

def skal(a, b):
	sk = 0
	for el in zip(a,b):
		sk += el[0]*el[1]
		#print ("%.3f*%.3f+" % (el[0], el[1]), end = '')
	#print ('\n')
	return sk

def combin(coeffs, vectors):
	suma = [0]*len(vectors[0])
	for el in zip(coeffs, vectors):
		suma = summ(suma, mul(el[0],el[1]))
		#print (mul(el[0],el[1]))
	return suma


def vect_pts(pt1, pt2):
	return (summ(pt2, mul(-1, pt1)))

def length(vect):
	length = 0
	for el in vect:
		length += el * el
	#length = sqrt(length)
	return length

def coss(vect1, vect2):
	return (skal(vect1, vect2))/ (length(vect1) * length(vect2))

def gram(*vectors):
	skals = []
	for v1 in vectors:
		for v2 in vectors:
			skals.append(skal(v1, v2))
	return skals

def det_2(a,b,c,d):
	return a * d - b * c


def cross(vect1, vect2):
	x = vect1[1] * vect2[2] - vect1[2] * vect2[1]
	y = -(vect1[0] * vect2[2] - vect1[2] * vect2[0])
	z = vect1[0] * vect2[1] - vect1[1] * vect2[0]
	return [x, y, z]

def f1(x,y,z):
	return (x*x+y*y+z*z)

def prn(lst, otstup=1):
	print ('\n')
	for ls in lst:
		counter = 0
		for el in ls:
			if el > 0:
				print ("%.2f" % el, end = '  ')
			else:
				print ("%.2f" % el, end = ' ')
			if counter == otstup:
				print ("    ", end = '')
			counter += 1 
		print ('\n')
#vydelenie poln_quadr, naxodim tretiy coefficient po dvum
def poln_quadr(coeff1, coeff2):
	return (coeff2/( 2.0 * sqrt(coeff1)))**2

def ratio(begin, end, r):
	answ = []
	for i in zip(begin, end):
		answ.append((i[0] + r * i[1])/(r + 1))
	return answ


def factorial(n):
	facto = 1
	i = 1
	for i in range(1, n+1):
		facto = facto * i
	return facto


def Poisson(p, lambd, k):
	return (exp(-lambd) * (lambd**k)/factorial(k))


def poly(x):
	return x**4-5*x**3-4*x**2-3*x+12
def pr_poly(x):
	return 4*x**3-15*x**2-8*x-3
def sec_poly(x):
	return 12*x**2-30*x-8
'''
print ((30 - sqrt(30*30+4*12*8))/24, (30 + sqrt(30*30+4*12*8))/24)
print ('rorni')
print ("poly(%f)=%f, poly(%f)= %f, poly(%f)poly(%f)=%f " % (0, poly(0), 2.5, poly(2.5),0, 2.5, poly(0)*poly(2.5)))
print ("poly(%f)=%f, poly(%f)= %f, poly(%f)poly(%f)=%f " % (1, poly(1), 2, poly(2),1, 2, poly(1)*poly(2)))
'''
def pol_del(a,b, delta):
	if b-a < delta:
		print ('x = %f' % ((a + b)/2.0) )
		return
	c = (a+b)/2.0
	print(' ')
	print("f(%f)=%f" % (c, poly(c)), end = ' ')
	if poly(c) < 0:
		print('< 0')
	if poly(c) > 0:
		print('> 0')
	print ('delta %f = (%f - %f)/2.0  = %f' % (c, b, a, (b-a)/2.0))
	print("f(%f)=%f" % (a, poly(a)), end = ' ')
	if poly(a) < 0:
		print('< 0')
	if poly(a) > 0:
		print('> 0')
	print("f(%f)=%f" % (b, poly(b)), end = ' ')
	if poly(b) < 0:
		print('< 0')
	if poly(b) > 0:
		print('> 0')
	if poly(a)*poly(c) < 0:
		print ('[a, b] = [%f, %f] ' % (a, c))
		pol_del(a,c,delta) 
	else:
		if poly(b)*poly(c) < 0:
			print ('[c, b] = [%f, %f] ' % (c, b))
			pol_del(c,b,delta)
'''
pol_del(0, 2.5, 0.6)
pol_del(1, 2, 0.6)

print ('\n')
print (pr_poly(1))
a = 2.5
print ('tangents')
print (a, a - poly(a)/pr_poly(a))
a = a - poly(a)/pr_poly(a)
print (a, a - poly(a)/pr_poly(a))
a = a - poly(a)/pr_poly(a)
print (a, a - poly(a)/pr_poly(a))
print (poly(a), poly(a)/3.0)
print ('completed!')
'''
def kasat(a,a1,i,delta):
	print ('\n')
	print (a, a1)
	print ('i=%d, pogreshn = %f' % (i, abs(a1-a)))
	if abs(a1 - a) < delta:
		return
	a = a- (poly(a)*(a1-a)/(poly(a1)-poly(a)))
	a1 = a1 - (poly(a1)/pr_poly(a1))
	kasat(a,a1,i+1, delta)
	
#kasat(0, 2.5, 0, 0.00001)

def be_rational(x):
	if x  % 1 == 0.000:
		status = 1 #status of ints 1, rationals 2, irrationals 0
		return [status, int(x)]
	for i in range(2,50000):
		if abs(round(x, 5) * i) % 1 == 0.00000:
			 c = int(round(x, 8) * i)
			 #return [c, i]
			 status = 2
			 return [status, c//gcd(abs(c), i), i//gcd(abs(c), i)]
	status = 0
	return ([status, x])

def ad_w_coef(upper_row, lower_row, coef):
	return combin([coef, 1], [upper_row, lower_row])

def print_as_rational(x):
	if be_rational(x)[0] < 2:
		print(be_rational(x)[1], end = ' ')
	if be_rational(x)[0] == 2:
		print ("%d/%d" % (be_rational(x)[1], be_rational(x)[2]), end = ' ')
def prn1(lst, otstup):
	print ('\n')
	for ls in lst:
		counter = 0
		for el in ls:
			print_as_rational(el)
			if counter == otstup:
				print ("  ", end = '')
			counter += 1 
		print ('\n')
#"DONT DELETE IT!"
def swap( arr, k, j):
	arr[j], arr[k] = arr[k], arr[j]
def Gauss(arr):
	n = len(arr)
	for i in range(0, n):
		for j in range(i+1, n):
			if arr[i][i] != 0:
				print ('pribavim k %d  stroke %d stroku c coef = ' % (j, i), end='')
				print("- %.2f/%.2f" % (arr[j][i], arr[i][i]))
				arr[j] = ad_w_coef(arr[i], arr[j], -arr[j][i]/arr[i][i])
				prn (arr)
			else:
				k = i
				while k < n and arr[k][i] == 0:
						k += 1
				if k < n:
					tmp = arr[k]
					arr[k]=arr[i]
					arr[i]=tmp
					print ('swap %d and %d' % (i, j))
					prn (arr)
					print("- %.2f/%.2f" % (arr[j][i], arr[i][i]))
					arr[j] = ad_w_coef(arr[i], arr[j], -arr[j][i]/arr[i][i])
					prn (arr)
#Currently is not working properly
def smart_Gauss(arr):
	n = len(arr)
	for i in range(0, n):
		for j in range(i+1, n):
			if arr[i][i] != 0:
				print ('pribavim k %d  stroke %d stroku c coef = ' % (j, i), end='')
				print("- %d/%d" % (arr[j][i], arr[i][i]))
				arr[j] = ad_w_coef(arr[i], arr[j], -((arr[j][i] -arr[j][i]%arr[i][i] )/arr[i][i]))

			else:
				k = i
				while k < n and arr[k][i] == 0:
						k += 1
				if k < n:
					tmp = arr[k]
					arr[k]=arr[i]
					arr[i]=tmp
					print ('swap %d and %d' % (i, j))
					prn (arr)
					print("- %.2f/%.2f" % (arr[j][i], arr[i][i]))
					arr[j] = ad_w_coef(arr[i], arr[j], -arr[j][i]/arr[i][i])
					prn (arr)
def Gram_Smith(lst):
	for i in range(1, len(lst)):
		for j in range(i):
			lst[i] = lst[i]-(skal(lst[j], lst[i])/skal(lst[j], lst[j]))
	print ('check')
	print (gram(lst))
def divi(ar, num):
	ls2 = []
	for el in ar:
		ls2.append(Fraction(el, num))
	return ls2
def nullify (Array, Position, Number_of_rows ):
	for i in range(Position + 1, Number_of_rows):
		#print('Added %d row to %d row with coeff = -%.3f/%.3f = %.3f' % (Position, i, Array[i][Position], Array[Position][Position], -Array[i][Position]/Array[Position][Position]))
		Array[i] = ad_w_coef(Array[Position], Array[i], -Array[i][Position]/Array[Position][Position])
		

from sympy import symbols, diff, sin, sinh, cos 
import math
import numpy as np 
from fractions import Fraction

a, c, u, v = symbols('a c u v', real = 'True') 
y = a*sinh(u)*sin(v) 
x = a*sinh(u)*cos(v) 
z = c*cos(u) 

fu = [diff(x, u), diff(y, u), diff(z, u)]
fv = [diff(x, v), diff(y, v), diff(z, v)]
print (skal(fu, fu), skal(fu, fv))
#
'''
#Pierson corellation index here
def sum(Array, length):
	s = 0
	for x in Array:
		s += x
	return s
def sr_arithm(Array, length):
	return sum(Array, length)/length
def sq(Array, length):
	for i in range(length):
		Array[i] *= Array[i]
def PlNtoAr(Array, length, number):
	for i in range(length):
		Array[i] += number
def minusAr(Array, length):
	for i in range(length):
		Array[i] = -Array[i]
arr = [[2,3,4,5,6], [20, 18, 15, 8, 2]]
ln = 5
sr1 = sr_arithm(arr[0], ln)
sr2 = sr_arithm(arr[1], ln)
print (sr1, sr2)
PlNtoAr(arr[0], ln, -sr1)
PlNtoAr(arr[1], ln, -sr2)
minusAr(arr[0], ln)
minusAr(arr[1], ln)
print ('hei')
prn (arr)
s1 = 0
for i in range(ln):
	skk = arr[0][i]*arr[1][i]
	print (skk, end = ' ')
	s1 += skk
sq(arr[0], ln)
sq(arr[1], ln)
print ('aa')
prn(arr)
s2 = sum(arr[0], ln)
s3 = sum(arr[1], ln)
print (s1, s2, s3)
r = s1/sqrt(s2*s3)
print (r)
'''


#DONT DELETE
'''  Using of those functions
nullify(a, 1, na)
a[5] = ad_w_coef(a[2], a[5], -1)
swap(a, 4,5)
prn(a)
'''
u1 = [0, 0, -1, 0, 0, 1]
u2 = [-1, -1, 0, 1, 1, 0]
v1 = [1, 0, 0, -1, 0, 2]
v2 = [1, 0, 0, -1, 2, 0]
w = [0, 0, 0, 2, 0, -2]
vectors = (v1, v2, u1, u2, w)
g1 = gram(v1, v2, u1, u2, w)
def div_long_lst(g1):
	i = 0
	j = 0
	n = len(g1)**0.5
	g_list = [ ]
	lst = []
	for el in g1:
		print (el, end = ' ')
		lst.append(el)
		i += 1
		if i%n== 0:
			i = 0
			j += 1
			g_list.append(lst)
			lst = [ ]
			print ('')
	return g_list
g_list = div_long_lst(g1)
for i in range(len(g_list)):
	nullify(g_list, i, len(g_list))
print("\n g_list")
prn(g_list)
D = 1.0
for i in range(len(g_list)):
		D *= g_list[i][i]
g2 = gram(v1, v2, u1, u2)
g_list2 = div_long_lst(g2)
print ("\n glist_2")
print (g_list2)
for i in range(len(g_list2)):
	nullify(g_list2, i, len(g_list2))
D1 = 1.0
for i in range(len(g_list2)):
		D1 *= g_list2[i][i]
print(D, D1)
print ((D/D1)**0.5)

def f11(x):
	while True:
		try:
			return ((5*(4**x) + 2*(6**x))/(4*(2**x) + 7**x))
			break
		except OverflowError:
			return 10**30
def integ(a = 2, b = 1000, n = 1000):
	I = 0
	ai = a
	for i in range(n):
		ai += round((b - a)*(1.0/n), 2)
		I += f11(ai)*(1.0/n)
	return I
for i in range(50, 5000000, 1000):
	print (i, integ(2, i, 100))	
