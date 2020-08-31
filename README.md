# Batched-Bandit
This repository aims to reproduce and expend the   ["Batched Bandits Problems"](https://arxiv.org/pdf/1505.00369.pdf "Batched Bandits Problems") and extend it by adding the following experiments:
- Ploting the "optimal grid" as suggested in section 4.2
- PredTau algorithm which estimate delta and predict tau(\Delta) 
- Improved-UCB - I added this experiment for fair comparison
- Improved-UCB with go-to-broke policy
- Batched alimination algorithm based on 
***This is not the official code of the paper***

# Instalation
The code was implementedin python3 and uses scipy and numpy. Please install using pip:
```pip install numpy```
```pip install scipy```

# Runing the code
```python main_paralle --alg improved_ucb_gtb --d poisson```

## plot
```python ploter --sub_plot```

