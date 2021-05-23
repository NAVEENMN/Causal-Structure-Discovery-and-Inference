# Causal Structure Discovery and Inference
## Abstract
Causal Structure Discovery (CSD) is the problem of identifying causal relationships from large quantities of data through computational methods. 
Causality or Causation is defined as a relationship between two events, states, process or objects such that changes in one event, state, process or object lead to changes in another. 
In general these events, states, process or objects are represented by variables such as X & Y. The key difference between association and causation lies in the potential of confounding. 
Suppose that no direct causal relationship exists between X and Y but rather a third variable Z causes both X and Y. 
In this case, even though X and Y are strongly associated, altering X will not lead to changes in Y. Z is called a con founder. 
In an experimental setup if intervening on a variable X leads changes to Y then causality can be established as X causes Y. 
However in the real world we cannot intervening on certain systems due to risk, cost or ethical concerns. 
Such systems include but not limited to climate, environment, biology, social etc. 
In systems like these studies are conducted purely based on observational data. 
Extracting causal structure through these observational data while adjusting for confounding has been a challenge for many scientific disciplines. 
Many machine learning methods have been proposed for this problem but they are largely based on associations. 
In this article we will explore the methods and challenges involved in CSD and also experiment regression based methods to extract underlying causal structure and infer future states of a given system.

## Introduction
Causation is a direct effect between variable X and Y that remains after adjusting for confounding. Confounding can be observed or unobserved. 
Since the 17th century modern science, we have had two kinds of scientific methods for discovering causes. 
The first method involved manipulating and varying features of a system to see what other features do or do not change. 
While there are many experiments that fit this methods perhaps the most famous one is Pavlov's classical conditioning experiment, where he established a stimulus-response connection. 
These methods shone brightly but manipulating systems like live animals or environments are bounded by ethics and costs. The notorious experiments like Tuskegee study and prison experiments among others have shown us why the intervention methods are dangerous. 
The other scientific methods for discovering causes involved observing the variation of features of system without manipulation. 
In these methods observational data will be collected for a system and just by observing who different attributes of a system changes causal connections can be established between different parts of the system. 
Some examples, include discovering astronomical objects through observational data or connecting weather patterns through remote sensing data.

## Acknowledgement and References
* [Introduction to foundations of Causal Discovery](https://link.springer.com/article/10.1007/s41060-016-0038-6)
* [Review of Causal Discovery Methods Based on Graphical Models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6558187/)
* [Dynamic Relational Inference in Multi-Agent Trajectories](https://arxiv.org/abs/2007.13524)
* [Amortized Causal Discovery](https://arxiv.org/abs/2006.10833)