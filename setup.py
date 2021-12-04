from setuptools import setup

setup(
    name= "DRLagents",
    version= "0.0.1",
    description= "A store for my DRL agents (from CS698R IITK)",
    packages= ["DRLagents"],
    requires= ['gym', 'torch', 'numpy', 'time', 'copy']
)