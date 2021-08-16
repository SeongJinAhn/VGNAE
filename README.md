# VGNAE
An implement of CIKM 2021 paper "Variational Graph Normalized Auto-Encoders" (CIKM 2021).
> Variational Graph Normalized Auto-Encoders.  
> Seong Jin Ahn, Myoung Ho Kim.  
> CIKM '21: The 30th ACM International Conference on Information and Knowledge Management Proceedings.  
> Short paper  

Thank you for your interest in our works!  
Paper is to be uploaded.

We find out that existing GAEs make embeddings of isolated nodes (nodes with no-observed links) zero vectors regardless of their feature information.  
Our works try to generate proper embeddings of isolated nodes (nodes with no-observed links).
![image](https://user-images.githubusercontent.com/37531907/129611067-0c4cb724-0bea-4b4b-a5b0-7afc56f87643.png)  
![image](https://user-images.githubusercontent.com/37531907/129611133-1dad1073-fcd2-4df8-a3df-1cca2cd2e090.png)

# Dependencies
Recent versions of the following packages for Python 3 are required:

* Anaconda3
* Python 3.8.0  
* Pytorch 1.8.1  
* torch_geometric 1.7.0  
* torch_scatter 2.0.6  

# Easy Run
> python main.py --dataset=Cora --training_rate=0.2 --epochs=300

# Citing
If you make advantage of our VGNAE in your research, please cite the following in your manuscript:
