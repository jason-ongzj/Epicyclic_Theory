import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from scipy.interpolate import interp1d, interp2d
from CalculateCoefficients import calcDampingMoment, calcTau, calcDampingExponents, \
	calcFrequencies

# PREFACE
# -----------------------------------------------------------------------------------------------
# This is a comparison with the 6DOF calculations in Astos for the analysis of the complex plane.
# For tricyclic theory calculations, the following are required:
# M_pBeta - Magnus moment due to Beta.
# M_q - Pitch moment due to q.
# M_alphaDot - Pitch moment due to rate of change of angle of attk.
# M_delta - Pitch moment due to control deflection angle
# M_alpha - Pitch moment due to angle of attk.
# Ix - Moment of inertia about x axis
# I (or Iyy) - Moment of inertia about y axis
# p - roll rate

# M_pBeta = C_mpBeta * q S d^2 / (2V)
# M_q = C_mq * q S d^2 / (2V)
# M_alphaDot = C_mAlphaDot * q S d^2 / (2V)
# M_alpha = C_mAlpha * q S d
# M_delta = C_mDelta * q S d
# q = 0.5 * rho * V^2

# Solution procedure for each time step
# 1. Find the density and the dynamic pressure.
# 2. Interpolate the aerodynamic coefficients based on the Mach number.
# 3. Then calculate the individual aerodynamic moments.
# 4. Calculate the coefficients A1 A2 A3 B1 B2 and B3 based on the
#    simplified equations in the paper by Nikolaides. Then find the
#    point on the complex plane for the particular timestep.
# 5. The outputs of the solution go into the inputs for the next time step.
# 6. For each time step, t value is assumed to be the difference between current and
#	 previous time steps. Use AoA_rate and sideslip_rate values at the particular
#	 time step to generate values for AoA and sideslip.

# Data to be extracted from Missle Datcom
# 1. C_mAlpha

# Strategy
# 1. Assume epicyclic motion, so C_mDelta is ignored.
# 2. Assume no magnus moment, so C_mpBeta is ignored.
# 3. Assume no damping, so C_mAlphaDot and C_mq are ignored.
# 3. Run a separate Astos calculation with C_mAlpha included.
#    C_mAlpha in Astos - moment coefficient wrt total AoA
# 4. As a result of 1 and 2, consider only A1 B1 A2 B2.
# 5. Gather data from Astos calculations: Ix, I, V, p, alt, q, AoA_rate, sideslip_rate
# Initial data: AoA[0], sideslip[0]
# Calculate: AoA and sideslip values for all iterations using generated using AoA_rate
#			 and sideslip_rate values.
# Validate against Astos data: AoA, sideslip

# Small angle restriction
# 1. True angle of attacks normally associated with aerodynamic coefficients
# 	 have a cosine component.
#	 alpha_t = alpha cos beta
# 	 beta_t = beta cos alpha
# -----------------------------------------------------------------------------------------------

# Define density function for interpolation
# alt = np.loadtxt("USA_AirProperties.txt", delimiter='\t', skiprows=1,
# 	usecols=[0])
# rho_data = np.loadtxt("USA_AirProperties.txt", delimiter='\t', skiprows=1,
# 	usecols=[1])

# rho_function = interp1d(alt, rho_data, kind='linear')

file_input = "Aeroballistics.csv"

# Define C_mq function for interpolation
cmq_x = np.loadtxt("CMq_deg.txt", delimiter='\t', max_rows=1)
cmq_y = np.loadtxt("CMq_deg.txt", delimiter='\t', skiprows=1, max_rows=1)
cmq_z = np.loadtxt("CMq_deg.txt", delimiter='\t', skiprows=2, usecols=[0,1])
cmq_function = interp2d(cmq_x, cmq_y, cmq_z, kind='linear')

# Define C_ma and C_mad functions for interpolations
mach_num = np.loadtxt("AeroCoeffs.txt", delimiter='\t', skiprows=1, usecols=[0])
cma = np.loadtxt("AeroCoeffs.txt", delimiter='\t', skiprows=1, usecols=[1])
cmad = np.loadtxt("AeroCoeffs.txt", delimiter='\t', skiprows=1, usecols=[2])
cma_function = interp1d(mach_num, cma, kind='linear')
cmad_function = interp1d(mach_num, cmad, kind='linear')

# Read input file
t = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[0])
Ixx = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[1])
Iyy = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[2])
p = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[3])/57.2958
V = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[4])*1000
mach = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[5])
alt = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[6])
q = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[7])
aoa_rate = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[8])
sideslip_rate = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[9])
aoa_astos = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[10])
sideslip_astos = np.loadtxt(file_input, delimiter=',', skiprows=1, usecols=[11])

S = 0.0755
d = 0.31
start = 0
aoa_initial = aoa_astos[start]
sideslip_initial = sideslip_astos[start]

print(aoa_initial, sideslip_initial)

real_array = np.zeros(1000)
imag_array = np.zeros(1000)

sideslip_rates = sideslip_rate[start]
aoa_rates = aoa_rate[start]

for i in range(start+1, 300):
	cm_ad = cmad_function(mach[i])*57.2958
	cm_alpha = cma_function(mach[i])*57.2958
	m_alpha = cm_alpha * q[i] * S * d
	m_damping = calcDampingMoment(0, 0, q[i], S, d, V[i])
	tau = calcTau(Ixx[i], Iyy[i], p[i], m_alpha)
	damp_exponents = calcDampingExponents(m_damping, Iyy[i], tau)
	frequencies = calcFrequencies(Ixx[i], Iyy[i], p[i], m_alpha)

	A1 = -(sideslip_rate[i] + frequencies[1] * aoa_initial)/(frequencies[0] - frequencies[1])
	A2 = -(sideslip_rate[i] + frequencies[0] * aoa_initial)/(frequencies[1] - frequencies[0])
	B1 = (aoa_rate[i] - frequencies[1] * sideslip_initial)/(frequencies[0] - frequencies[1])
	B2 = (aoa_rate[i] - frequencies[0] * sideslip_initial)/(frequencies[1] - frequencies[0])

	# Compute k1, k2, m1, m2
	k1 = complex(B1, A1)
	k2 = complex(B2, A2)
	m1 = complex(damp_exponents[0], frequencies[0])
	m2 = complex(damp_exponents[1], frequencies[1])

	print(i, "k1:", k1)
	print(i, "k2:", k2)

	complex_num = k1*cmath.exp(m1 * (t[i] - t[i-1])) + k2*cmath.exp(m2 * (t[i] - t[i-1]))
	sideslip_initial = complex_num.real
	aoa_initial = complex_num.imag

	real_array[i] = complex_num.real
	imag_array[i] = complex_num.imag

plt.figure()
fig, ax = plt.subplots(figsize=(12, 10))
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.set_aspect(1)
plt.plot(imag_array, real_array, color='green', marker='x', linestyle='none', markersize=4, label='Epicyclic Theory')
plt.plot(aoa_astos[0:300], sideslip_astos[0:300],label='ASTOS')
plt.axis([-0.3,0.5,-0.3,0.3])
plt.xlabel("Angle of Attack (\N{DEGREE SIGN})", fontsize=12)
plt.ylabel("Sideslip Angle (\N{DEGREE SIGN})", fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.savefig("Output",bbox_inches = 'tight', pad_inches = 0.3)
