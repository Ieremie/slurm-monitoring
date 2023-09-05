# slurm-monitoring
Monitoring GPU, RAM and CPU usage for slurm partitions and users.

This is an app written in Python using flask. It gathers information using the standard slurm functions (squeue, control etc.)
The implementation assumes some fixed parameters such as maximum resource available to reduce  the number of requests sent to the slurm server.

You can quickly adapt this implementation to your own server. You can add as many partitions as you want and these will be displayed as a 2-column page.

In the case that resources can be locked by a user and not actually being used, this can be monitored too. This is when a job requests resources from a node containing GPU nodes, but does not actually use them.

![alt text](https://github.com/Ieremie/slurm-monitoring/blob/main/front-end-example-1.png)
![alt text](https://github.com/Ieremie/slurm-monitoring/blob/main/front-end-example-0.png)
