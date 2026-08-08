# 1. Gere amostras de distribuições de Poisson com diferentes valores de λ. Para cada amostra, calcule a média e o desvio-padrão. Em seguida, faça um gráfico da média em função do desvio-padrão. O que você observa?

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Definindo os valores de lambda (taxa média de ocorrência)
lambdas = np.linspace(1, 100, 50) # 50 valores de lambda entre 1 e 100
tamanho_amostra = 1000

medias = []
desvios_padrao = []

# Gerando amostras e calculando média e desvio-padrão para cada lambda
np.random.seed(42)

for lam in lambdas:
    amostra = np.random.poisson(lam=lam, size=tamanho_amostra) # Amostra de Poisson
    # Calculando média e desvio-padrão amostrais
    medias.append(np.mean(amostra))
    desvios_padrao.append(np.std(amostra, ddof=1))

# Gráfico: média vs desvio-padrão
plt.figure(figsize=(9, 5))
plt.scatter(desvios_padrao, medias, color='darkblue', alpha=0.7, label='Amostras simuladas')

# Linha de tendência teórica
std_teorico = np.sqrt(lambdas)
plt.plot(std_teorico, lambdas, color='red', linestyle='--', label=r'Teórico: $\mu = \sigma^2$')

plt.title('Relação entre Média e Desvio-Padrão na Distribuição de Poisson', fontsize=12)
plt.xlabel('Desvio-Padrão ($s$)', fontsize=11)
plt.ylabel('Média ($\overline{x}$)', fontsize=11)
plt.legend()
plt.tight_layout()

plt.show()

#Observações: a média cresce de forma quadrática em relação do desvio-padrão. Conforme a média aumenta, a variabilidade dos dados também cresce.