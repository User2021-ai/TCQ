


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from ast import literal_eval
import os
import shutil
import csv
import concurrent.futures
import copy
from tqdm import tqdm
import pickle
import time

 

def simulateTWIC(G, seedNode,Target):
    newActive = True
    currentActiveNodes = copy.deepcopy(seedNode)
    newActiveNodes = set()
    activatedNodes = copy.deepcopy(seedNode) 
    seedNode=list(set(seedNode))

    influenceSpread = 0

    while (newActive):
        for node in currentActiveNodes:
            for neighbor in G.neighbors(node):  
                if (neighbor not in activatedNodes):
                    propProbability=1/nx.degree(G, neighbor)
                    if ( random.random() < propProbability): #
                        newActiveNodes.add(neighbor)
                        activatedNodes.append(neighbor)
        influenceSpread += len([node for node in newActiveNodes if G.nodes[node]['label'] in Target and node not in seedNode])
        if newActiveNodes:
            currentActiveNodes = list(newActiveNodes)
            newActiveNodes = set()
        else:
            newActive = False
    return influenceSpread


def TWIC_cascade(G,S,iterations,Target):
    avgSize = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
        results = [executor.submit(simulateTWIC, G, S,Target) for _ in range(iterations)]
        for f in concurrent.futures.as_completed(results):
            avgSize += f.result()
    return avgSize/iterations


def get_set_influence(graph,Seed,D_model,iterations,num_nodes_list,Target):
  influence_probabilities=[]
  for rangSeed in num_nodes_list:
    if D_model=='TWIC':
      influence_prob =  TWIC_cascade(graph, Seed[:rangSeed], iterations,Target)
      influence_probabilities.append(influence_prob)

    print(f"Influence for range {rangSeed} nodes - Influence Probability: {influence_prob}")
  return influence_probabilities


import heapq


def get_neighbors_with_smaller_kcore(G, node, node_kcore,k_core):
    neighbors = G.neighbors(node)
    return [neighbor for neighbor in neighbors if k_core[neighbor] <= node_kcore]

def get_Candidate_seeds_TCQ(G, k, Target, n_C):
    np.random.seed(42)
    k_core = nx.core_number(G)
    degree_dict = dict(G.degree())
    target_set = set(Target)

    L_values = {}
    for node in G.nodes():
        neighbors = G.neighbors(node)
        neighbors_smaller = [n for n in neighbors if k_core[n] <= k_core[node]]
        count = sum(1 for n in neighbors_smaller if G.nodes[n]['label'] in target_set)
        degree = degree_dict[node]
        L_values[node] = count * (1 - 1/degree) if neighbors_smaller and degree else -1

    top_k = n_C * k + 1 if n_C == 1 else n_C * k
    top_nodes = heapq.nlargest(top_k, L_values, key=L_values.get)
    return top_nodes, L_values


 
 
def P_w_Values_G_TCQ(graph, node, S_set, target_set):
    if graph.nodes[node]['label'] not in target_set:
        return 0
    r_v = sum(1 for n in graph[node] if n in S_set)
    degree = graph.degree[node]
    return (1 - (1 - 1/degree) ** r_v) if degree else 0
    
def sigma_S_TCQ(graph, S, Target, prev_Gamma_S=None, prev_sigma=0):
    S_set = set(S)
    target_set = set(Target)
    
    if prev_Gamma_S is None:
        Gamma_S = set().union(*(graph.neighbors(node) for node in S_set))
    else:
        new_node = S[-1]
        Gamma_S = prev_Gamma_S.union(graph.neighbors(new_node))
    
    Gamma_S_ = {n for n in Gamma_S if graph.nodes[n]['label'] in target_set and n not in S_set}
    if prev_Gamma_S is None:
        new_contributions = sum(P_w_Values_G_TCQ(graph, n, S_set, target_set) for n in Gamma_S_)
    else:
        new_contributions = sum(P_w_Values_G_TCQ(graph, n, S_set, target_set) for n in (Gamma_S_ - prev_Gamma_S))
    
    return prev_sigma + new_contributions, Gamma_S

 
 
def select_k_nodes_TCQ(Q, K, candidate_set):
    visited = []

  
    current_state =  candidate_set.index(candidate_set[0])
    visited.append(candidate_set[current_state])

    while len(visited) < K:        
        action = argmax_Q_TCQ(Q,current_state,visited,candidate_set)
        if candidate_set[action] in visited:
          print('Error in argmax_Q  ', candidate_set[action])
        if action is None:
            break  
        
        current_state = action
        visited.append(candidate_set[current_state])

    return visited

  

def argmax_Q_TCQ(Q, state, visited, candidate_set):
    visited_indices = {candidate_set.index(node) for node in visited}
    valid_actions = [i for i in range(Q.shape[1]) if i not in visited_indices]
    return valid_actions[np.argmax(Q[state, valid_actions])] if valid_actions else None


def max_Q_TCQ(Q, action, visited ,candidate_set):
    max_value = float('-inf')
    for index, element in enumerate(Q[action, :]):
        if candidate_set[index] not in visited  and element > max_value:
            max_value = element
    return max_value



def TCQ(G,K,Taget,n_C=6,alpha = 0.1, gamma = 0.5, epsilon = 0.5):     
    
    candidate_set,L_values=get_Candidate_seeds_TCQ(G, K,Taget,n_C)
    
    

    initial_state = candidate_set.index(candidate_set[0])
    M=len(candidate_set)

  
    Q = np.zeros([M, M])
    for j in range(M):
        prev_Gamma_S=None
        prev_sigma=0
        q_value,_=sigma_S_TCQ(G,[ candidate_set[j],candidate_set[0]],Taget)
        Q[:, j] = q_value 


    episodes = 200
   
   

    for _ in range(episodes):
        prev_sigma = 0
        prev_Gamma_S = set()
    
        if _%10==0:
            print('episodes = ',_ ) 
      
        state = initial_state
        visited = [candidate_set[state]]

        r1=sigma_S_TCQ(G,visited,Taget)
        for _ in range(K-1):
            
            if np.random.uniform(0, 1) < epsilon:                
                
                possible_actions = [candidate_set.index(node) for node in candidate_set if node not in visited] 
                action = np.random.choice(possible_actions)
            else:
              
                action = argmax_Q_TCQ(Q,state,visited,candidate_set)

         
             

            current_sigma, current_Gamma = sigma_S_TCQ(G, visited + [candidate_set[action]], Target, prev_Gamma_S, prev_sigma)
            reward = current_sigma - prev_sigma
            prev_sigma = current_sigma
            prev_Gamma_S = current_Gamma

    

            
            maxQ=max_Q_TCQ(Q, action, visited+[candidate_set[action]] ,candidate_set)
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * maxQ- Q[state, action])
           
            state = action
            visited.append(candidate_set[state])

    
    selected_nodes = select_k_nodes_TCQ(Q, K, candidate_set)
   
    if len(selected_nodes) != len(set(selected_nodes)):
        print("Duplicate elements found in list!")
    else:
       
        return selected_nodes
 



def getSeedAlgo(G,Algo,Target,k=50):
    attribute_name = 'label'    
    Seed=[]
    if Algo=='TCQ':
        Seed=TCQ(G,k,Target,5,alpha = 0.1, gamma = 0.9, epsilon = 0.5)
     
    return Seed

 

dataList=[] 
data__list=[  'Politician','LastFM','Gnutella' , 'CiaoDVD', 'Sex','PGP',   'Hamster'  ,'PowerGrid' , 'GrQc' ,'DBLP', ] 

data__list=[  'GrQc' ] 
TargetDistribution=["Uniform"  ] 
attribute_name = 'label'
Target = [ 'Type 1']

 
num_nodes_list = [1,  5, 10,15,20,25,30,35,40,45,50]
k = 50
range_S=50
N_iterations = 10000
D_model='TWIC'
root_save='' 
Algo_list=['TCQ']
dataPath='/Datasets' 

savePatht=''
for Dataset in data__list:
    for Algo in Algo_list:
         print(f"++++++++++++++++++++++++++++++++++++++++++++")
         for TargetDistributionType in TargetDistribution:              
            with open(f"{dataPath}/{Dataset}.pickle", "rb") as f:
                G = pickle.load(f)

            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()

            print(f" Algorithm  : {Algo}")
            print(f"{TargetDistributionType} ,   {Dataset}")
            print(f"Number of nodes  : {num_nodes}")
            print(f"Number of edges  : {num_edges}")

           
            type1_count = 0           
            for node in G.nodes(data=True):
                if node[1]['label'] in Target:
                    type1_count += 1

            print(f"Number of nodes with 'Type 1' label in G: {type1_count}")
            print(f"---------------------------------------------------")
            start_time = time.time()             
            def set_seed(seed=5000):
                np.random.seed(seed)
                random.seed(seed)
            
            
            set_seed()
            Seed=getSeedAlgo(G.copy(),Algo,Target,k=50)             
            end_time = time.time()

            elapsed_time = end_time - start_time
            print('elapsed_time',elapsed_time)   
            Spread=get_set_influence(G,Seed,D_model,N_iterations,num_nodes_list,Target)
            print(Spread)
 
            
 
