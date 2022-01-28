# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:04:52 2022

@author: User
"""

#Problem: libiomp5md.dll in the numpy package installed by Anaconda conflicts 
#with libiomp5md.dll in pytoch -> leads to kernel crashing
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#from pgmpy.base import DAG

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx


bayesNet = BayesianModel()
bayesNet.add_node("E")
bayesNet.add_node("B")
bayesNet.add_node("R")
bayesNet.add_node("A")

bayesNet.add_edge("E", "R")
bayesNet.add_edge("E", "A")
bayesNet.add_edge("B", "A")

cpd_E = TabularCPD("E", 2, values=[[.999999], [.000001]],
                   state_names={"E": ["false", "true"]})
cpd_B = TabularCPD("B", 2, values=[[.99], [.01]],
                   state_names={"B": ["false", "true"]})

cpd_R = TabularCPD("R", 2, values=[[1,0], [0,1]],
                   evidence=["E"], evidence_card=[2],
                   state_names={"E": ["false", "true"], "R": ["false", "true"]})

cpd_A = TabularCPD("A", 2,
                   values=[[0.999, .59, .05, .02],[.001, .41, .95, .98]],
                   evidence=["B", "E"], evidence_card=[2, 2],
                   state_names={"B": ["false", "true"], "E": ["false", "true"],
                                "A": ["false", "true"]})
bayesNet.add_cpds(cpd_E, cpd_B, cpd_R, cpd_A)

#check if model is well defined
if bayesNet.check_model(): print("Model is true.")

#some checks to see if probabilities are assigned correctly
solver = VariableElimination(bayesNet)
print(solver.query(variables=["R"], evidence={"E":"false"}))
print(solver.query(variables=["B"]))
print(solver.query(variables=["A"], evidence={"E":"false", "B":"false"}))

#plot graph
nx.draw(bayesNet, with_labels=True)

#calulating (cond). probabilities
print(solver.query(variables=["A"]))
print(solver.query(variables=["A"], evidence={"R":"true"}))
print(solver.query(variables=["B"], evidence={"A":"true"}))
print(solver.query(variables=["B"], evidence={"A":"true", "R":"true"}))





