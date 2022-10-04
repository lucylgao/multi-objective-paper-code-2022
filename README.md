# multi-objective-paper-code-2022

Code to reproduce all numerical results in "Necessary and sufficient conditions for multiple-objective optimal regression designs" by Lucy L. Gao, Jane J. Ye, Shangzhi Zeng, and Julie Zhou. 

## Required software 

To run the code in this repository, you will need to install [CVX](http://cvxr.com/cvx/), the [MATLAB Optimization Toolbox](https://www.mathworks.com/products/optimization.html), and the [MATLAB Symbolic Math Toolbox](https://www.mathworks.com/products/symbolic.html).  

## Organization 

Application1.m produces Figure 1, Table 3, and the first column of results in Table 4. To produce the second column of Table 4, change line 132 of Application1.m to `deal(0.90, 0.70)` and rerun the script. To produce the third column of Table 4, change line 132 of Application1.m to `deal(0.70, 0.70)` To produce the final column of Table 4, change line 132 of Application1.m to `deal(0.90, 0.90)` and re-run the script. 

Application2.m produces Figure 2 and Table 5.

Application3.m produces Table 6. 




