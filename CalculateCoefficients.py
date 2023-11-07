import numpy as np
import math

def calcDampingMoment(cmq, cmad, q, S, d, V):
	m_damping = (cmq + cmad) * 0.5*q*S*math.pow(d, 2)/V
	return m_damping

def calcTau(Ix, I, p, m_alpha):
	numerator = p * Ix * 0.5 / I
	denominator = abs(math.sqrt(math.pow(numerator, 2) - (m_alpha/I)))
	return numerator/denominator

def calcDampingExponents(m_damping, I, tau):
	lambda_1 = m_damping * 0.5 * (1 + tau) / I
	lambda_2 = m_damping * 0.5 * (1 - tau) / I
	damping_exponents = np.array([lambda_1, lambda_2])
	return damping_exponents

def calcFrequencies(Ix, I, p, m_alpha):
	A = p * Ix * 0.5 / I
	omega_1 = A + math.sqrt(math.pow(A,2) - (m_alpha/I))
	omega_2 = A - math.sqrt(math.pow(A,2) - (m_alpha/I))
	frequencies = np.array([omega_1, omega_2])
	return frequencies
