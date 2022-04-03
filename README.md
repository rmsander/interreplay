# Neighborhood Mixup Experience Replay (NMER)
---------------------------------------------------------------------------------------------------------------------------------------------------------
[Ryan Sander](https://scholar.google.com/citations?user=7B6apiIAAAAJ&hl=en)<sup>1</sup>, [Wilko Schwarting](https://scholar.google.com/citations?hl=en&user=YI1EqBoAAAAJ)<sup>1</sup>, [Tim Seyde](https://scholar.google.com/citations?hl=en&user=FJ7ILzkAAAAJ)<sup>1</sup>, [Igor Gilitschenski](https://scholar.google.com/citations?hl=en&user=Nuw1Y4oAAAAJ)<sup>1</sup>, [Sertac Karaman](https://scholar.google.com/citations?hl=en&user=Vu-Zb7EAAAAJ)<sup>2</sup>, [Daniela Rus](https://scholar.google.com/citations?hl=en&user=910z20QAAAAJ)<sup>1</sup>

1 - MIT CSAIL, 2 - MIT LIDS

**Paper (L4DC 2022)** | **[Technical Report](https://github.com/rmsander/rmsander.github.io/blob/master/projects/nmer_tech_report.pdf)** | **AirXv** | **[Website](https://sites.google.com/view/nmer-drl)**

---------------------------------------------------------------------------------------------------------------------------------------------------------

![nmer_diagram](img/diagram.png)
**What is NMER?** NMER is a novel replay buffer technique designed for improving continuous control tasks that recombines previous experiences of deep reinforcement learning agents linearly through a simple geometric heuristic.


## Code Release
Code release coming soon!

## Trained Agents
Videos coming soon.

## Paper
Please find our 2021 NeurIPS Deep RL Workshop **[paper](https://openreview.net/pdf?id=jp9NJIlTK-t)**, as well as our supplementary **[technical report](https://rmsander.github.io/projects/nmer_tech_report.pdf)**. 

If you find NMER useful, please consider citing our paper as:

```
@inproceedings{
  sander2021neighborhood,
  title={Neighborhood Mixup Experience Replay: Local Convex Interpolation for Improved Sample Efficiency in Continuous Control Tasks},
  author={Ryan Sander and Wilko Schwarting and Tim Seyde and Igor Gilitschenski and Sertac Karaman and Daniela Rus},
  booktitle={Deep RL Workshop NeurIPS 2021},
  year={2021},
  url={https://openreview.net/forum?id=jp9NJIlTK-t}
}
```

## Acknowledgements
This research was supported by the Toyota Research Institute (TRI). This article solely reflects the opinions and conclusions of its authors and not TRI,
Toyota, or any other entity. We thank TRI for their support. The authors thank the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC and consultation resources that have contributed to the research results reported within this publication. 
