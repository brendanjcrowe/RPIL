# RPIL
## Robust Probabilistic Imitation Learning

Robust Probabilistic Imiation Learning is a method that I conceived where in we model a set of expert demonstrations are having two sources, a true expert and an adversary. Using logistic regression we can pose the problem as a mixutre of multinomial logistic regression models. From there we can solve the non-convex optimization using a Expectation Maximization like algorithm. Experimentally I show that this algorithm can detect and remove bad demonstrations from the training set and thus perform much better that if the demonstrations are considered to be correct. A more detailed and formal discuss of this work is in the paper.
