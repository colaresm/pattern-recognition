import runs
import results
from train_test_split import train_test_split
from dataset import load_data
import time

start = time.time()
X, y = load_data()
end = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y)
p_values = [0.5, 2/3, 1, 3/2, 2, 5/2]
test_sizes = [0.2, 0.3, 0.5, 0.7, 0.8]

#### Classificador Vizinho Mais Próximo #####
#results_nn = runs.NN(X, y, test_sizes, p_values, rounds=10)
#results.show_results(results_nn)

#### Classificador Distância Mínima ao Centróide #####
#results_mdc = runs.MDC(X, y, test_sizes, rounds=100)
#results.show_results(results_mdc)

#### Classificador Distância Mínima ao Centróide (versão robusta a outliers) #####
#results_mdc = runs.MDC(X, y, test_sizes, rounds=100,is_robust_version=True)
#results.show_results(results_mdc)

#### Classificador de Máxima Correlação #####
results_mc = runs.mc(X, y, test_sizes, rounds=100)
results.show_results(results_mc)