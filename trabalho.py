import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

""" (a) Dab """
R = 8.314              # J/(mol*K)
P_atm = 1.0            # atm
P_Pa = P_atm * 101325  # Pa
T_C = 35.2             # °C
T_K = 273.15 + T_C     # 308.35 K

V_A = 77.99 * 1e-6     # m^3
V_B = 78.63 * 1e-6     # m^3
L = 85.9 * 1e-3        # m
d = 2.08 * 1e-3        # m
A = np.pi * d**2 / 4   # Área (m^2)

c = P_Pa / (R * T_K)   # Concentração total molar (mol/m^3)

# Dados dos Componentes (1: H2, 2: N2, 3: CO2)
M = {1: 2.016, 2: 28.013, 3: 44.01}
sigma = {1: 2.968, 2: 3.681, 3: 3.996}
eps_k = {1: 33.3, 2: 91.5, 3: 190.0}

# Parâmetros da Integral de Colisão Omega_D
A_omega, B_omega, C_omega, D_omega = 1.06036, 0.15610, 0.19300, 0.47635
E_omega, F_omega, G_omega, H_omega = 1.03587, 1.52996, 1.76474, 3.89411

def integral_colisao(T_asterisco):
  return (A_omega / (T_asterisco**B_omega) + C_omega / np.exp(D_omega * T_asterisco) +
          E_omega / np.exp(F_omega * T_asterisco) + G_omega / np.exp(H_omega * T_asterisco))

def chapman_enskog(i, j):
  """Calcula a difusividade binária D_ij e converte para m^2/s."""
  sigma_ij = (sigma[i] + sigma[j]) / 2
  eps_k_ij = np.sqrt(eps_k[i] * eps_k[j])

  T_asterisco = T_K / eps_k_ij
  Omega_D = integral_colisao(T_asterisco)

  D_ij_cm2s = (0.001858 * T_K**1.5 / (P_atm * sigma_ij**2 * Omega_D) * (1/M[i] + 1/M[j])**0.5)
  return D_ij_cm2s * 1e-4

D12 = chapman_enskog(1, 2)
D13 = chapman_enskog(1, 3)
D23 = chapman_enskog(2, 3)

# Ajeitando a matriz com 1-2 = 2-1
D = { (1, 2): D12, (2, 1): D12, (1, 3): D13, (3, 1): D13, (2, 3): D23, (3, 2): D23 }

print("Difusividades Binárias")
print(f"D12 (H2-N2): {D12:.4e} m^2/s")
print(f"D13 (H2-CO2): {D13:.4e} m^2/s")
print(f"D23 (N2-CO2): {D23:.4e} m^2/s\n")

""" (b) e (c) Maxwell-Stefan """
def fluxo_maxwell_stefan(x_A, x_B):
  """Calcula os fluxos J_i (mol/(m^2*s)) resolvendo o sistema 2x2 reduzido"""

  eps_stab = 1e-12 # critério de estabilidade
  x_A = np.clip(x_A, eps_stab, 1.0) # garantindo soma 1
  x_B = np.clip(x_B, eps_stab, 1.0) # garantindo soma 1
  x_A /= np.sum(x_A)
  x_B /= np.sum(x_B)

  # Média x e ∆x
  x_medio = (x_A + x_B) / 2
  delta_x = x_B - x_A

  d12 = c * D[(1, 2)]
  d13 = c * D[(1, 3)]
  d23 = c * D[(2, 3)]

  x1, x2, x3 = x_medio
  dx1, dx2, dx3 = delta_x

  # Matriz M (M * [J1, J2]^T = B) - Sistema 2x2
  M_mat = np.zeros((2, 2))
  B_vet = np.zeros(2)

  # Equacao 1 (H2)
  M_mat[0, 0] = x2/d12 + x3/d13 + x1/d13
  M_mat[0, 1] = x1/d13 - x1/d12
  B_vet[0] = -dx1 / L

  # Equacao 2 (N2)
  M_mat[1, 0] = x2/d23 - x2/d12
  M_mat[1, 1] = x1/d12 + x3/d23 + x2/d23
  B_vet[1] = -dx2 / L

  try:
    fluxonA_fluxonB = np.linalg.solve(M_mat, B_vet)
    fluxonA, fluxonB = fluxonA_fluxonB
    fluxonC = -fluxonA - fluxonB  # linha de amarração
  except np.linalg.LinAlgError:
    fluxonA, fluxonB, fluxonC = 0.0, 0.0, 0.0

  return np.array([fluxonA, fluxonB, fluxonC])

def derivadas_composicao(t, y):
  """ dna/dt = -na"S """
  x_A = y[0:3]
  x_B = y[3:6]

  J = fluxo_maxwell_stefan(x_A, x_B)  # fluxo de A para B

  dxA_dt = -J * A / (c * V_A)
  dxB_dt = J * A / (c * V_B)

  dydt = np.concatenate((dxA_dt, dxB_dt))

  return dydt

# Valores iniciais + plotagem
eps = 1e-10 # substitui zeros iniciais exatos

# Bulbo A
x1A_0 = eps             # H2 (0.0)
x2A_0 = 0.50086 - eps   # N2
x3A_0 = 0.49914         # CO2

# Bulbo B
x1B_0 = 0.50121         # H2
x2B_0 = 0.49879 - eps   # N2
x3B_0 = eps             # CO2 (0.0)

y0 = np.array([x1A_0, x2A_0, x3A_0, x1B_0, x2B_0, x3B_0])

t_span = [0, 90000] # 90000 segundos (~25 horas)

# Resolvendo pelo BDF
sol = solve_ivp(
  derivadas_composicao,
  t_span,
  y0,
  method='BDF', # Backward Differentiation Formula como método
  rtol=1e-7,
  atol=1e-10
)

tempo = sol.t
x1A, x2A, x3A = sol.y[0], sol.y[1], sol.y[2]
x1B, x2B, x3B = sol.y[3], sol.y[4], sol.y[5]

print("Tempo de simulação: {:.2f} h".format(tempo[-1]/3600))
print("\nComposições Finais (equilíbrio):")
print(f"Bulbo A: H₂={x1A[-1]:.4f}, N₂={x2A[-1]:.4f}, CO₂={x3A[-1]:.4f}")
print(f"Bulbo B: H₂={x1B[-1]:.4f}, N₂={x2B[-1]:.4f}, CO₂={x3B[-1]:.4f}")

# Plotando para entender a evolução temporal e comparar com os
# dados experimentais de Duncan & Torr
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(tempo/3600, x1A, label='$H_2$ (A: X, -)', color='red', linestyle='-', marker='x', markevery=500)
plt.plot(tempo/3600, x1B, label='$H_2$ (B: o, --)', color='red', linestyle='--', marker='o', markevery=500)

plt.plot(tempo/3600, x2A, label='$N_2$ (A: X, -)', color='blue', linestyle='-', marker='x', markevery=500)
plt.plot(tempo/3600, x2B, label='$N_2$ (B: o, --)', color='blue', linestyle='--', marker='o', markevery=500)

plt.plot(tempo/3600, x3A, label='$CO_2$ (A: X, -)', color='green', linestyle='-', marker='x', markevery=500)
plt.plot(tempo/3600, x3B, label='$CO_2$ (B: o, --)', color='green', linestyle='--', marker='o', markevery=500)

# Pontos experimentais
t_exp_co2_bulboB = [0., 3.961178364839883, 7.928021945256125, 12.020842029936189, 16.969556920508076]
x_exp_co2_bulboB = [0, 0.09658536585365862, 0.1580487804878049, 0.18292682926829268, 0.2063414634146341]

t_exp_h2_bulboA = [0, 3.953873218438785, 7.922058560438902, 11.95375395074244, 16.90455602600035]
x_exp_h2_bulboA = [0, 0.1682926829268293, 0.2165853658536585, 0.2414634146341464, 0.24439024390243902]

t_exp_h2_bulboB = [0, 3.937175740950563, 7.915498837139959, 12.012791460432938, 17.025612737789963]
x_exp_h2_bulboB = [0.5019512195121951, 0.33219512195121953, 0.2809756097560976, 0.26195121951219513, 0.25609756097560976]

t_exp_co2_bulboA = [0, 3.9292742560677443, 7.96961655435625, 11.945107042757465, 16.89874172580356]
x_exp_co2_bulboA = [0.5034146341463415, 0.40975609756097564, 0.34975609756097564, 0.32634146341463416, 0.30146341463414633]

t_exp_n2_bulboA = [0, 4.050181883236926, 7.961118730991709, 11.933627526984315, 16.944212535034886]
x_exp_n2_bulboA = [0.5019512195121951, 0.4229268292682927, 0.4331707317073171, 0.4390243902439025, 0.4551219512195122]

t_exp_n2_bulboB = [0, 3.9738505575764798, 7.94799928439382, 11.92170075734987, 16.93541654242948]
x_exp_n2_bulboB = [0.5034146341463415, 0.5721951219512196, 0.5619512195121952, 0.5560975609756098, 0.5414634146341464]

# Plotando pontos experimentais com mesma cor dos componentes
plt.scatter(t_exp_h2_bulboA, x_exp_h2_bulboA, color='red')
plt.scatter(t_exp_h2_bulboB, x_exp_h2_bulboB, color='red', marker='x')
plt.scatter(t_exp_n2_bulboA, x_exp_n2_bulboA, color='blue')
plt.scatter(t_exp_n2_bulboB, x_exp_n2_bulboB, color='blue', marker='x')
plt.scatter(t_exp_co2_bulboA, x_exp_co2_bulboA, color='green')
plt.scatter(t_exp_co2_bulboB, x_exp_co2_bulboB, color='green', marker='x')

plt.xlabel('Tempo (s)', fontsize=16, labelpad=10)
plt.ylabel('Fração Molar', fontsize=16, labelpad=10)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()