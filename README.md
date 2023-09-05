# slurm-monitoring
Monitoring GPU, RAM and CPU usage for slurm partitions and users.

This is an app written in Python using flask. It gathers information using the standard slurm functions (squeue, control etc.)
The implementation assumes some fixed parameters such as maximum resource available to reduce  the number of requests sent to the slurm server.

You can quickly adapt this implementation to your own server.

![alt text](https://github.com/Ieremie/slurm-monitoring/blob/main/front-end-example-1.png)
![alt text](https://github.com/Ieremie/slurm-monitoring/blob/main/front-end-example-0.png)
