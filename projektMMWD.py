import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib


# WARTOŚCI STAŁE
ingrid_storehouse = np.array([[431, 1063, 379, 517, 1115, 442, 172, 609, 402, 1035]])  # magazyn składników
recipe = np.array([[3, 1, 0, 3, 0.5, 0],  # przepisy, ilosc składników potrzebna do
                   [4, 3, 3, 6, 1, 1.5],  # wyprodukowania poszczególnych produktów
                   [2, 1, 1.5, 1, 0.5, 0.6],
                   [2, 0, 1, 5, 1, 0],
                   [4, 10, 0, 5, 0.2, 0.2],
                   [0, 3, 2, 2, 0, 0.7],
                   [1, 0, 2, 0, 0, 0],
                   [3, 1, 4, 2, 0, 0.6],
                   [0, 4, 2, 0, 0, 1],
                   [2, 1, 3, 2, 0.5, 0.5]])

ingrid_price = np.array([[0.17, 0.12, 0.08, 0.13, 0.09, 0.5, 0.6, 0.27, 0.21, 0.01]])  # ceny składników
prod_price = np.array([[3.9, 5.2, 4.9, 4.8, 0.5, 1.8]])  # ceny produktów


# GENEROWANIE PRZYKŁADOWEJ LISTY ILOŚCI PRODUKTÓW
def random_vec():  # funkcja generująca przykładowy wektor ilosci produktów
    random_list = []
    for i in range(0, 6):
        if i == 0 or i == 1 or i == 2:
            n = random.randint(0, 115)
            random_list.append(n)
        else:
            n = random.randint(0, 115)
            random_list.append(n)
    return np.array(random_list)

# FUNKCJA CELU
def obj_func(n, w=recipe, p=ingrid_price, pi=prod_price):  # funkcj celu
    n = np.array(n)
    
    rec_price = np.dot(w.T, np.diagflat(p.T))  # macierz cen składników potrzebnych na wyprodukowanie 1 porcji każdego produktu
    price_ingrid = np.dot(rec_price.T,
                          n.T)  # wektor cen składników potrzebnych na wyprodukowanie n porcji każdego produktu
    
    sum_price_ingrid = 0
    for i in range(0, np.size(price_ingrid) - 1):  # sumowanie cen poszczególnych składników
        sum_price_ingrid = sum_price_ingrid + price_ingrid[i]
        
    price_prod = np.dot(np.diagflat(pi), n.T)  # wektor cen n ilosci produktów
    
    sum_price_prod = 0
    for i in range(0, np.size(price_prod) - 1):  # sumowanie cen produktów
        sum_price_prod += price_prod[i]
        
    gain = sum_price_prod - sum_price_ingrid  # zysk końcowy (funkcja celu)
    return gain

def fit_fun(parent, w=recipe, d=ingrid_storehouse):
    # LIMIT 1
    obj_value = obj_func(parent)
    over1 = []
    over2 = []
    over3 = 0
    sklad = np.dot(w, parent.T)
    d = d.T
    for i in range(0, len(sklad)):
        if sklad[i] > d[i]:
            over1.append(sklad[i] - d[i])
        else:
            over1.append(0)
    penalty1 = np.array([0.17, 0.12, 0.08, 0.13, 0.09, 0.5, 0.6, 0.27, 0.21, 0.01]).T
    obj_value = obj_value - np.dot(over1, penalty1)
    # LIMIT 2
    limits = np.array([[15, 15, 30, 15, 30, 30]]).T
    for i in range(0, len(parent)):
        if np.any(parent[i] < limits[i]):
            e = (limits[i]-parent[i])
            over2.append(e)
        else:
            over2.append(0)
    penalty2 = np.array([0.2,0.2,0.2,0.2,0.2,0.2]).T
    obj_value = obj_value - np.dot(over2, penalty2)
    # LIMIT 3
    sum = 0
    parent = parent.T
    for e in range(0, np.size(parent)):
        sum = sum + parent[e]
    if sum > 250:
        over3 = sum - 250
    obj_value = obj_value - over3 * 0.1
    return obj_value

# OGRANICZENIA

def cond1(n, w=recipe, d=ingrid_storehouse):  # warunek 1 - ilosć składników potrzebna do
    #c = np.zeros((10, 1))                    # wytworzenia produktów nie przekracza
    s = np.dot(w, n.T)                        # ilosci składników w magazynie
    d = d.T
    for i in range(0, len(s)):
        if np.any(s[i] > d[i]):
            return False
    return True

def cond2(n):  # warunek 2 - minimalna ilosć poszczególnych produtków
    limits = np.array([[15, 15, 30, 15, 30, 30]])
    if np.any(n < limits):
        return False
    return True

def cond3(n):  # warunek 3 - suma produktów nie przekracza 250
    sum = 0
    n = n.T
    for e in range(0, np.size(n)):
        sum = sum + n[e]
    if sum > 250:
        return False
    else:
        return True
    
# FUNKCJE POTRZEBNE DO ALGORYTMU
        
def generate_vector():  # funkcja generująca wektor produktów spełniający wszystkie warunki
    actual_prod = np.array(random_vec())
    stop = 0
    while stop < 1:
        if cond3(actual_prod):
            stop = 1
        else:
            actual_prod = np.array(random_vec())
    return actual_prod

def is_acceptable(parent):
    if cond3(parent): #cond1(parent, recipe, ingrid_storehouse): #and and cond3(parent):
        return True
    else:
        return False

def select_acceptable(parent_gen):
    parent_gen = [parent for parent in parent_gen if is_acceptable(parent)]
    return parent_gen


def create_parent_generation(n_gen_parents):
    ret = []
    for i in range(n_gen_parents):
        ret.append(generate_vector())
    return ret

#SELEKCJE

def select_tournament(parents, n_pop):
    rand = [0,0,0,0]
    parent_gen = []
    for e in range(n_pop):
        rand[0] = np.random.randint(low=0, high=len(parents)-1)
        rand[1] = np.random.randint(low=0, high=len(parents)-1)
        rand[2] = np.random.randint(low=0, high=len(parents)-1)
        rand[3] = np.random.randint(low=0, high=len(parents)-1)
        best = fit_fun(parents[rand[0]])
        best_rand = 0
        for i in range(3):
            if best <= fit_fun(parents[rand[i+1]]):
                best = fit_fun(parents[rand[i+1]])
                best_rand = i+1
        random = parents[rand[best_rand]]
        parent_gen.append(random)
    return parent_gen


def select_ruletka(population,n_population):
    parent_gen = []
    fitness_sum = 0
    act_fitnes_sum = 0
    probability_vec = []
    j_vec = []
    for i in range(len(population)):
        if fit_fun(population[i]) > 0:
            fitness_sum += fit_fun(population[i])
    for idx, parent in enumerate(population):
        if fit_fun(population[i]) > 0:
            act_fitnes_sum += fit_fun(parent)
            s_probability = act_fitnes_sum/fitness_sum
            probability_vec.append(s_probability)
        else:
            del population[idx]
            
    for i in range(n_population):
        rand = np.random.random_sample()
        for j in range(len(probability_vec)):
            if probability_vec[j] >= rand:
                parent_gen.append(population[j])
                j_vec.append(j)
                break
    return parent_gen
        
# MUTACJA

def mutate_parent(parent, n_mutations):
    parent_size = len(parent)
    mutated_parent = parent
    for i in range(n_mutations):
        rand = np.random.randint(0, parent_size)
        mutated_parent[rand] = np.random.randint(0, 115)
    return mutated_parent


def mutate_gen(parent_g, n_mutations, pm):
    mutated_parent_gen = []
    pc = pm
    for parent in parent_g:
        e = np.random.random_sample(1)
        if e<=pc:
            mutated_parent_gen.append(mutate_parent(parent, n_mutations))
        else:
            mutated_parent_gen.append(parent)
    return mutated_parent_gen

# KRZYŻOWANIE

def crossover(parents, n_offspring, pc):
    offspring = []
    zapas = parents[:]
    pc = pc
    while len(parents) > 2:
        rand_mom = np.random.randint(low=0, high=len(parents)-1)
        rand_dad = np.random.randint(low=0, high=len(parents)-1)
        if rand_mom == rand_dad:
            rand_dad = np.random.randint(low=0, high=len(parents)-1)
        random_mom = parents.pop(rand_mom)
        random_dad = parents.pop(rand_dad)
        e1 = np.random.random_sample(1)
        if e1 <= pc:
            dad_mask1 = np.random.randint(0, 2, size=np.array(random_dad.shape))
            mom_mask1 = np.logical_not(dad_mask1)
            mom_mask2 = np.logical_not(mom_mask1)
            dad_mask2 = np.logical_not(dad_mask1)
            child1 = np.add(np.multiply(random_dad, dad_mask1), np.multiply(random_mom, mom_mask1))
            child2 = np.add(np.multiply(random_dad, dad_mask2), np.multiply(random_mom, mom_mask2))
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(random_dad)
            offspring.append(random_mom)
        offspring.append(parents.pop())
    return zapas, offspring

def crossover2(parents, n_offspring, pc):
    offspring = []
    zapas = parents[:]
    pc = pc
    while len(parents) > 2:
        rand_mom = np.random.randint(low=0, high=len(parents)-1)
        rand_dad = np.random.randint(low=0, high=len(parents)-1)
        if rand_mom == rand_dad:
            rand_dad = np.random.randint(low=0, high=len(parents)-1)
        random_mom = parents.pop(rand_mom)
        random_dad = parents.pop(rand_dad)
        e1 = np.random.random_sample(1)
        if e1 <= pc:
            dad_mask1 = np.array([1,1,1,0,0,0])
            mom_mask1 = np.logical_not(dad_mask1)
            mom_mask2 = np.logical_not(mom_mask1)
            dad_mask2 = np.logical_not(dad_mask1)
            child1 = np.add(np.multiply(random_dad, dad_mask1), np.multiply(random_mom, mom_mask1))
            child2 = np.add(np.multiply(random_dad, dad_mask2), np.multiply(random_mom, mom_mask2))
            offspring.append(child1)
            offspring.append(child2)          
        offspring.append(parents.pop())
    return zapas, offspring


###############################################################################
#FUNKCJE POMOCNICZE

def create_half_zero(n):
    random_list = np.array([0,0,0,0,0,0])
    rand = []
    for e in range(0,n):
        for i in range(0, 5):
            #if i == 0 or i == 1 or i == 2:
            random_list[i] = np.random.randint(low=0, high=5)
        rand.append(random_list)
        random_list = np.array([0,0,0,0,0,0])
    return rand

def select_best(parent_gen, n_best):
    costs = []
    for idx, parent in enumerate(parent_gen):
        parent_cost = fit_fun(parent)
        costs.append([idx, parent_cost])
    costs_tmp = pd.DataFrame(costs).sort_values(by=1, ascending=False).reset_index(drop=True)
    selected_parent_idx = list(costs_tmp.iloc[:n_best, 0])
    if n_best>1:
        selected_parents = [parent for idx, parent in enumerate(parent_gen) if idx in selected_parent_idx]
    else:
        selected_parents = [parent for idx, parent in enumerate(parent_gen) if idx in selected_parent_idx]
        selected_parents = selected_parents[0]
    return selected_parents


def select_worst(parent_gen,n_best):
    costs = []
    for idx, parent in enumerate(parent_gen):
        parent_cost = fit_fun(parent)
        costs.append([idx, parent_cost])
    costs_tmp = pd.DataFrame(costs).sort_values(by=1, ascending=True).reset_index(drop=True)
    selected_parent_idx = list(costs_tmp.iloc[:n_best, 0])
    if n_best>1:
        selected_parents = [parent for idx, parent in enumerate(parent_gen) if idx in selected_parent_idx]
    else:
        selected_parents = [parent for idx, parent in enumerate(parent_gen) if idx in selected_parent_idx]
        selected_parents = selected_parents[0]
    return selected_parents

def calculate_average(pop):
    sum_pop = 0
    for idx, parent in enumerate(pop):
        sum_pop = sum_pop + fit_fun(parent)
    average = sum_pop/len(pop)
    return average

def best_of_the_best(parent):
    best = 0
    for i in range(len(parent)):
        if best <= parent[i]:
            best = parent[i]
        return best

def odchyl_proby(parents,avr):
    suma = 0
    for parent in parents:
        suma = suma + ((fit_fun(parent)-avr)*(fit_fun(parent)-avr))
    suma = np.sqrt(suma/(len(parents)-1))
    return suma

def find_max(parents,maximum):
    for i in range(len(parents)):
        if parents[i] == maximum:
            return i
        
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def gui_layout1():
    sg.theme('Topanga')

    layout1 = [
        [sg.Text('Please enter parameter value')],
        [sg.Text('LIKELIHOOD OF MUTATION = ', size=(25, 1)), sg.InputText()],
        [sg.Text('LIKELIHOOD OF CROSSING  = ', size=(25, 1)), sg.InputText()],
        [sg.Text('ITERATION NUMBER = ', size=(25, 1)), sg.InputText()],
        [sg.Text('POPULATION SIZE = ', size=(25, 1)), sg.InputText()],
        [sg.Submit(), sg.Cancel()]
    ]

    window1 = sg.Window('Simple data entry window', layout1)
    event, values = window1.read()
    window1.close()
    return event, values[0], values[1], values[2], values[3]

###############################################################################

norm = [np.array([31, 15, 51, 29, 84, 27]), np.array([ 30,  35, 114,  22,  14,  16]), np.array([52,  4, 11, 81, 25, 10]), np.array([ 4, 42, 32, 39, 74, 48]), np.array([31, 49, 68, 18, 47, 27]), np.array([  3,   2,   9, 105,  57,  50]), np.array([114,  23,   0,  20,  20,   2]), np.array([23,  6, 64, 44, 86,  4]), np.array([77, 16, 91, 13,  1, 36]), np.array([15, 55,  2, 73, 51, 41]), np.array([25,  3, 20,  8,  6,  8]), np.array([68, 33, 18,  3, 18, 79]), np.array([ 3,  5, 23, 75, 36, 42]), np.array([16, 29,  6, 51, 54, 14]), np.array([12,  5, 19, 97, 52, 29]), np.array([23, 20, 55, 78, 52, 19]), np.array([28, 75, 16, 14,  2, 71]), np.array([30,  8, 17, 97, 65, 20]), np.array([32,  5, 41, 41,  8, 29]), np.array([17, 39,  9, 55, 22, 53]), np.array([61, 13, 35,  3, 96, 19]), np.array([52, 13, 53, 32, 78, 13]), np.array([13, 55, 49, 42, 16, 24]), np.array([50, 17, 38,  9, 22, 43]), np.array([16, 27, 29, 65,  8, 66]), np.array([73,  9, 16,  3, 24, 20]), np.array([13, 10, 37, 32,  1, 34]), np.array([ 7, 13, 68,  4, 76, 30]), np.array([14, 88, 13, 20, 34, 75]), np.array([31, 17, 11, 74, 65, 40]), np.array([ 4, 25, 51,  7, 27,  6]), np.array([ 1, 43, 30, 41, 37, 75]), np.array([22, 28, 74, 19, 69, 36]), np.array([12, 66,  0, 45, 39, 37]), np.array([ 13,  37, 113,  30,  41,   1]), np.array([18, 53, 15,  4, 98, 45]), np.array([ 61,  16,   3,   8, 110,   7]), np.array([25,  8, 54, 51,  6, 43]), np.array([16, 43, 19, 56,  7, 53]), np.array([10, 40, 22, 58, 15, 99]), np.array([ 5, 40, 82, 33, 30, 48]), np.array([15, 29,  5,  9, 82, 56]), np.array([ 4, 55, 14,  4, 10, 65]), np.array([ 2, 45, 10, 85,  6, 53]), np.array([40, 10, 58, 73, 53, 14]), np.array([12,  6,  1, 30, 27, 14]), np.array([57, 23,  7, 37, 32, 63]), np.array([ 6, 37,  7, 86, 73, 17]), np.array([ 23, 103,  29,   2,  16,  71]), np.array([74, 47, 11, 19, 49, 25]), np.array([ 10, 112,  46,   1,  17,  44]), np.array([10,  8, 37, 43, 69, 15]), np.array([34, 38, 20, 57, 16, 39]), np.array([77, 17, 14, 49,  9, 58]), np.array([  3,   0, 107,  33,   1,  45]), np.array([  1, 104,  63,  36,  11,   8]), np.array([16, 54, 17, 40, 23, 69]), np.array([ 3, 36, 39, 37, 51, 39]), np.array([30, 39,  2, 36, 21, 57]), np.array([ 14,  10, 103,  13,   3,  11]), np.array([27, 45, 87, 38,  7, 14]), np.array([32, 19, 90, 42,  3,  8]), np.array([22, 44, 34, 11, 46,  0]), np.array([52, 52, 25, 28,  5,  9]), np.array([ 14,   4,  30,  79, 102,  18]), np.array([28,  3, 34, 56, 55, 42]), np.array([ 8, 15, 71, 35, 75, 43]), np.array([85, 30,  7, 38,  5, 56]), np.array([ 6, 15, 87, 12, 19, 27]), np.array([ 9, 41, 52, 13, 56, 48]), np.array([ 50,  15,   5, 112,  42,  24]), np.array([89, 42, 17,  8, 10,  4]), np.array([71, 37, 13, 20,  5, 22]), np.array([ 6, 19, 29, 30, 33, 60]), np.array([68, 28, 56, 25, 13, 23]), np.array([12, 71, 25,  6, 48, 24]), np.array([41,  8, 29, 40, 69,  8]), np.array([44,  1, 57, 16, 87, 23]), np.array([31, 52, 48, 11, 57, 50]), np.array([ 16,   1,  21,  21, 111,  32]), np.array([68,  0, 68, 14, 20, 18]), np.array([ 0, 96, 68,  8, 22, 36]), np.array([ 3, 21, 53, 25, 81, 23]), np.array([ 5, 57,  4,  0, 62,  1]), np.array([27, 44, 66, 47, 16, 30]), np.array([46, 10, 12, 23, 65, 47]), np.array([74,  0, 56, 36, 17,  1]), np.array([37, 78, 35, 14,  8, 20]), np.array([36, 33, 21, 50, 31, 19]), np.array([ 16, 102,  13,  21,   7,  37]), np.array([35, 10, 85, 71, 10, 27]), np.array([60, 30, 23, 26, 81,  4]), np.array([42, 49, 21, 17, 77, 38]), np.array([97,  7, 42,  8, 30,  1]), np.array([  9, 109,  47,  34,  47,   3]), np.array([ 8, 41,  3, 75, 11, 78]), np.array([62, 33, 57, 17, 34, 34]), np.array([ 30, 105,  27,  22,   0,  46]), np.array([ 65,  24,  23,  29, 102,   4]), np.array([36, 22, 19, 26, 45, 64])]
zlo = [np.array([ 22,   8,  11,  24, 100,  69]), np.array([31, 14, 99, 16,  3, 66]), np.array([39,  9,  8,  4, 53, 41]), np.array([ 1, 44, 23,  5, 15, 95]), np.array([11, 25, 44,  3, 88, 79]), np.array([39,  1,  1, 19, 30, 96]), np.array([48,  7, 47,  0, 19, 68]), np.array([88, 12,  6, 30, 16, 93]), np.array([29, 12, 31, 35, 57, 85]), np.array([19, 24, 12, 25, 89, 66]), np.array([20,  5, 10,  2, 40, 62]), np.array([16, 85,  9,  9, 12, 98]), np.array([  0,  19,  20,   4, 109,  56]), np.array([23,  4,  5,  3, 11, 29]), np.array([ 43,   2,   5,  69,   7, 102]), np.array([57, 26, 27,  0,  0, 91]), np.array([  5,  11,  33,   5, 111,  42]), np.array([  4,  11,  27,   0, 103,  98]), np.array([ 27,  26,  35,   7,  35, 110]), np.array([13, 43, 29,  6,  3, 98]), np.array([30, 19,  6,  4, 74, 46]), np.array([15,  5, 22, 25, 17, 90]), np.array([18, 24, 52, 19, 39, 86]), np.array([28,  8,  7,  1, 49, 58]), np.array([ 9,  9,  2, 10, 24, 58]), np.array([ 0, 12, 16, 16, 81, 40]), np.array([54, 33, 10,  5, 30, 85]), np.array([ 23,   6, 113,   7,  40,  45]), np.array([13,  6, 70, 25, 40, 96]), np.array([ 40,  14,  49,  16,  15, 113]), np.array([ 51,  34,  23,  24,   4, 104]), np.array([ 5, 66, 15,  4, 38, 93]), np.array([ 23,  10,  26,   3,  11, 114]), np.array([24, 10, 44, 46, 34, 91]), np.array([13,  6,  3, 20,  5, 85]), np.array([12,  2, 38, 47, 67, 72]), np.array([25,  9, 61,  2, 18, 78]), np.array([ 6,  0, 32,  3, 84, 64]), np.array([46,  7,  7, 23, 39, 91]), np.array([ 1, 15, 17, 20, 71, 89]), np.array([12, 26, 37, 10, 38, 62]), np.array([  9,  11,   9,  39,  64, 110]), np.array([ 46,  38,   6,  18,  31, 107]), np.array([13,  4, 45, 45,  6, 90]), np.array([ 0, 26,  5, 10, 68, 58]), np.array([ 7,  6, 10, 48, 91, 83]), np.array([ 45,  44,   9,  16,  24, 102]), np.array([ 31,  10,  57,   9,  10, 109]), np.array([63,  8, 15, 12, 62, 70]), np.array([ 3,  8, 60,  6, 23, 64]), np.array([ 3,  3, 41,  3, 14, 43]), np.array([  8,  37,  40,  45,  15, 101]), np.array([28, 10,  7,  5, 93, 78]), np.array([50,  9, 49,  3, 27, 90]), np.array([  9,  16,   0,  13, 102,  32]), np.array([25,  4, 16, 12, 62, 92]), np.array([ 2,  0, 90, 18,  1, 75]), np.array([ 15,   6,  85,   5,   4, 110]), np.array([30,  0, 62,  1, 20, 54]), np.array([ 5, 33, 22, 14, 81, 70]), np.array([17,  6, 22, 30, 20, 90]), np.array([ 28,   0,  57,   4,  25, 104]), np.array([ 29,  29,  26,  51,   6, 106]), np.array([ 22,  52,  15,  15,  36, 102]), np.array([  3,   2,  10,  54,  49, 109]), np.array([ 32,   0,  12,  68,   0, 106]), np.array([  3,   9,   4,  36, 115,  44]), np.array([ 11,  20,  20,  25,   4, 113]), np.array([ 16,  12,  15,   3, 113,  62]), np.array([26, 13, 11, 30, 25, 60]), np.array([  4,   9,   4,  36, 101,  47]), np.array([ 2,  5, 61, 21, 10, 90]), np.array([25, 35, 49,  5, 32, 93]), np.array([ 33,  37,  24,  10,  31, 113]), np.array([ 6, 43, 48, 18,  4, 90]), np.array([ 31,   7,  13,  10, 112,  54]), np.array([  7,   3,  29,  26,  49, 111]), np.array([10, 48, 18, 17, 39, 99]), np.array([31,  4,  9, 31, 39, 55]), np.array([24,  2, 24,  5, 69, 60]), np.array([ 25,  49,  43,  12,   4, 111]), np.array([45, 10, 19, 13, 54, 82]), np.array([12, 23,  3, 32, 85, 59]), np.array([ 12,   7, 114,  25,   8,  53]), np.array([32, 47, 35,  2,  4, 89]), np.array([47, 24,  3, 16,  5, 97]), np.array([ 13,  16,  74,  20,   6, 108]), np.array([23, 13, 21, 10,  7, 52]), np.array([ 6, 18, 11,  9, 94, 72]), np.array([ 9,  9,  5, 55, 68, 86]), np.array([ 31,   0,  11,  18,  53, 114]), np.array([43,  3, 56, 40,  6, 93]), np.array([ 3, 35, 40, 29,  5, 79]), np.array([35, 13, 82,  1,  0, 64]), np.array([44, 18, 45,  6, 24, 81]), np.array([ 10,   0,   8,   1, 108,  63]), np.array([ 10,   8,   8,   7,  25, 107]), np.array([ 43,  42,  22,   2,  17, 103]), np.array([17, 30, 35,  5,  2, 59]), np.array([ 0, 16, 31, 45, 64, 89])]

#IMPLEMENTACJA ALGORYTMU

def gen_algo(n_iterations,pc=0.8,pm=0.2, gen_size=100):
    
    # ZMIENNE
    parent_gen = norm #create_parent_generation(gen_size)
    
    #WYKRESY
    best_child = []
    best_func_tab = []
    worst_func_tab = []
    average_tab = []
    outcome = []
    roznica = []
    
    #INFO DO SPRAWOZDANIA
    odchylenie = 0
    iteracje = 0
    fsnaj = 0
    
    #BŁĄD WZGLĘDNY
    relative_error = 0
    
    #NAJLEPSZY Z WYNIKOW
    wo = []
    best_of_the_best_child = 0
    
    fsnaj = select_best(parent_gen, n_best=1)
    
    # ALGPRYTM
    for i in range(n_iterations):
        parent_gen = select_tournament(parent_gen, gen_size)
        parent_gen, parent_gen1 = crossover(parent_gen, gen_size,pc)
        parent_gen2 = mutate_gen(parent_gen, n_mutations=1, pm=pm)
        parent_gen =  parent_gen2 + parent_gen1 + parent_gen
        parent_gen = select_acceptable(parent_gen)
        
        #TABLICE DO WYKRESÓW 
        
        # NAJLEPSZY
        best_child = select_best(parent_gen, n_best=1)
        best_func_tab.append(fit_fun(best_child))
        
        #NAJGORSZY
        worst_child = select_worst(parent_gen, n_best=1)
        worst_func_tab.append(fit_fun(worst_child))
        
        #SREDNIA
        average_tab.append(calculate_average(parent_gen))
        
        #DODATKOWE POMOCNE ZMIENNE
        if best_of_the_best_child <= (fit_fun(best_child)):
            wo = best_child
            outcome.append(fit_fun(best_child))
            best_of_the_best_child = fit_fun(best_child)
        else:
            outcome.append(best_of_the_best_child)
        if i == 299:
            odchylenie = odchyl_proby(parent_gen,calculate_average(parent_gen))
            
        print('iteracja nr:', i, ' Wynik: ', best_child, 'zysk outcome: ', outcome[i-1], 'zysk fit child:',fit_fun(best_child)) 
        
    #BŁĄD WZGLEDNY
    relative_error = (186 - outcome[-1])/186*100
    
    #SZYKANIE ITERACJI 
    iteracje = find_max(outcome,outcome[-1])
    
    for j in range(n_iterations):
        roznica.append(outcome[j] - average_tab[j])
    
    # CZAS
    y = []
    for i in range(0, n_iterations):
        y.append(i)
    time = np.array(y)
    
    # PLOT
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(time, outcome, label='champion')
    ax.plot(time, best_func_tab, label='best child')
    ax.grid(True)
    ax.set_xlabel("iteration number")
    ax.set_ylabel("gain")
    ax.set_title('evolutionary algorithm')
    ax.legend()
    ax.legend()
    
    #WYSWIETLENIE KONCOWYCH DANYCH
    print('Odchylenie standardowe ostatniego dziecka: ',odchylenie)
    print('Iteracja w której uzyskano najlepszy wynik: ',iteracje)
    print('Champion: ', outcome[-1])
    print('Najgorszy a ostatnich: ', worst_func_tab[-1])
    print('Srednia wartosc z ostatniej iteracji: ',average_tab[-1])
    print('Najlepszy wynik z pierwszej iteracji: ', fit_fun(fsnaj))
    print('Błąd względny: ', relative_error)
    print('Najlepszy z najlepszych: ', wo)
    return fig, wo

#WYWOLANIE ALGORYTMU

ev, var1, var2, var3, var4 = gui_layout1()
fig, tt = gen_algo(n_iterations=int(var3), pc=float(var2), pm=float(var1), gen_size=int(var4))
print('nasz super extra swietny wynik', tt)
print('oraz zyski które musza byc OGROOOMNE', fit_fun(tt))
print(is_acceptable(tt))
print(cond3(tt))


matplotlib.use("TkAgg")

layout = [
    [sg.Text("Wyniki")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Text("najlepsza wartość:  " + str(fit_fun(tt)))],
    [sg.Text("najlepsze rozwiązanie:  " + str(tt))],
    [sg.Button("Zakończ")]
]

# STWORZENIE I POKAZANIE FORMY BEZ FIGURY

window = sg.Window(
    "Evolutionary algorithm - results",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)

# DODANIE WYKRESU DO OKNA

draw_figure(window["-CANVAS-"].TKCanvas, fig)
event, values = window.read()
window.close()
