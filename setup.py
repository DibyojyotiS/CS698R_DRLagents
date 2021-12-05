from setuptools import setup

setup(
    name= "CS698R_DRLagents",
    version= "0.0.1",
    description= "A store for my DRL agents (from CS698R IITK)",
    packages= ["CS698R_DRLagents"],
    requires= ['gym', 'torch', 'numpy', 'time', 'copy']
)