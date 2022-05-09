# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:04:52 2022

@author: User
"""
# Problem: libiomp5md.dll in the numpy package installed by Anaconda conflicts
# with libiomp5md.dll in pytoch -> leads to kernel crashing
import os

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run_exercise_2():

    # Defining network structure
    water_sprinkler_model = BayesianNetwork(
        [
            ("Cloudy", "Sprinkler"),
            ("Cloudy", "Rain"),
            ("Sprinkler", "WetGrass"),
            ("Rain", "WetGrass"),
        ]
    )

    # Defining the parameters using CPT
    cpd_cloudy = TabularCPD(variable="Cloudy", variable_card=2, values=[[0.5], [0.5]])
    cpd_sprinkler = TabularCPD(
        variable="Sprinkler",
        variable_card=2,
        values=[[0.5, 0.9], [0.5, 0.1]],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    cpd_rain = TabularCPD(
        variable="Rain",
        variable_card=2,
        values=[[0.8, 0.2], [0.2, 0.8]],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    cpd_wet_grass = TabularCPD(
        variable="WetGrass",
        variable_card=2,
        values=[[1.0, 0.1, 0.1, 0.01], [0.0, 0.9, 0.9, 0.99]],
        evidence=["Sprinkler", "Rain"],
        evidence_card=[2, 2],
    )

    # Associating the parameters with the model structure
    water_sprinkler_model.add_cpds(cpd_cloudy, cpd_sprinkler, cpd_rain, cpd_wet_grass)
    assert water_sprinkler_model.check_model(), "Model is not correct"

    # plot graph
    nx.draw(water_sprinkler_model, with_labels=True)
    plt.show()

    # now you can derive inference with it
    solver = VariableElimination(water_sprinkler_model)
    posterior_p = solver.query(["Sprinkler"], evidence={"WetGrass": 1.0})
    print(
        "Probability of sprinkler=True with WetGrass is True: ", posterior_p.values[1]
    )
    print(posterior_p)
    posterior_p = solver.query(["Sprinkler"], evidence={"WetGrass": 1.0, "Rain": 1.0})
    print(
        "Probability of sprinkler=True with WetGrass and Rain is True: ",
        posterior_p.values[1],
    )
    print(posterior_p)


def run_exercise_3():
    bayesNet = BayesianNetwork()
    bayesNet.add_node("E")
    bayesNet.add_node("B")
    bayesNet.add_node("R")
    bayesNet.add_node("A")

    bayesNet.add_edge("E", "R")
    bayesNet.add_edge("E", "A")
    bayesNet.add_edge("B", "A")

    cpd_E = TabularCPD(
        "E", 2, values=[[0.999999], [0.000001]], state_names={"E": ["false", "true"]}
    )
    cpd_B = TabularCPD(
        "B", 2, values=[[0.99], [0.01]], state_names={"B": ["false", "true"]}
    )

    cpd_R = TabularCPD(
        "R",
        2,
        values=[[1, 0], [0, 1]],
        evidence=["E"],
        evidence_card=[2],
        state_names={"E": ["false", "true"], "R": ["false", "true"]},
    )

    cpd_A = TabularCPD(
        "A",
        2,
        values=[[0.999, 0.59, 0.05, 0.02], [0.001, 0.41, 0.95, 0.98]],
        evidence=["B", "E"],
        evidence_card=[2, 2],
        state_names={
            "B": ["false", "true"],
            "E": ["false", "true"],
            "A": ["false", "true"],
        },
    )
    bayesNet.add_cpds(cpd_E, cpd_B, cpd_R, cpd_A)

    # check if model is well defined
    if bayesNet.check_model():
        print("Model is true.")

    # some checks to see if probabilities are assigned correctly
    solver = VariableElimination(bayesNet)
    print(solver.query(variables=["R"], evidence={"E": "false"}))
    print(solver.query(variables=["B"]))
    print(solver.query(variables=["A"], evidence={"E": "false", "B": "false"}))

    # plot graph
    nx.draw(bayesNet, with_labels=True)
    plt.show()

    # calulating (cond). probabilities
    print(solver.query(variables=["A"]))
    print(solver.query(variables=["A"], evidence={"R": "true"}))
    print(solver.query(variables=["B"], evidence={"A": "true"}))
    print(solver.query(variables=["B"], evidence={"A": "true", "R": "true"}))


if __name__ == "__main__":
    run_exercise_2()
    run_exercise_3()
