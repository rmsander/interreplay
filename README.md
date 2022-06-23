# Neighborhood Mixup Experience Replay (NMER)
---------------------------------------------------------------------------------------------------------------------------------------------------------
[Ryan Sander](https://scholar.google.com/citations?user=7B6apiIAAAAJ&hl=en)<sup>1</sup>, [Wilko Schwarting](https://scholar.google.com/citations?hl=en&user=YI1EqBoAAAAJ)<sup>1</sup>, [Tim Seyde](https://scholar.google.com/citations?hl=en&user=FJ7ILzkAAAAJ)<sup>1</sup>, [Igor Gilitschenski](https://scholar.google.com/citations?hl=en&user=Nuw1Y4oAAAAJ)<sup>1</sup>, [Sertac Karaman](https://scholar.google.com/citations?hl=en&user=Vu-Zb7EAAAAJ)<sup>2</sup>, [Daniela Rus](https://scholar.google.com/citations?hl=en&user=910z20QAAAAJ)<sup>1</sup>

1 - MIT CSAIL, 2 - MIT LIDS

**[Paper (L4DC 2022)](https://proceedings.mlr.press/v168/sander22a/sander22a.pdf)** | **[Technical Report](https://github.com/rmsander/rmsander.github.io/blob/master/projects/nmer_tech_report.pdf)** | **[AirXv](https://arxiv.org/abs/2205.09117)** | **[Website](https://sites.google.com/view/nmer-drl)**

---------------------------------------------------------------------------------------------------------------------------------------------------------

![nmer_diagram](img/diagram.png)
**What is NMER?** NMER is a novel replay buffer technique designed for improving continuous control tasks that recombines previous experiences of deep reinforcement learning agents linearly through a simple geometric heuristic.


## Code Release
**Code release coming soon!**

## Trained Agents
Videos coming soon.

## Paper
Please find our 2022 L4DC **[paper]([https://openreview.net/pdf?id=jp9NJIlTK-t](https://proceedings.mlr.press/v168/sander22a/sander22a.pdf))**, as well as our supplementary **[technical report](https://rmsander.github.io/projects/nmer_tech_report.pdf)**. 

If you find NMER useful, please consider citing our paper as:

```
@InProceedings{pmlr-v168-sander22a,
  title = 	 {Neighborhood Mixup Experience Replay: Local Convex Interpolation for Improved Sample Efficiency in Continuous Control Tasks},
  author =       {Sander, Ryan and Schwarting, Wilko and Seyde, Tim and Gilitschenski, Igor and Karaman, Sertac and Rus, Daniela},
  booktitle = 	 {Proceedings of The 4th Annual Learning for Dynamics and Control Conference},
  pages = 	 {954--967},
  year = 	 {2022},
  editor = 	 {Firoozi, Roya and Mehr, Negar and Yel, Esen and Antonova, Rika and Bohg, Jeannette and Schwager, Mac and Kochenderfer, Mykel},
  volume = 	 {168},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--24 Jun},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v168/sander22a/sander22a.pdf},
  url = 	 {https://proceedings.mlr.press/v168/sander22a.html}
}
```

## Acknowledgements
This research was supported by the Toyota Research Institute (TRI). This article solely reflects the opinions and conclusions of its authors and not TRI,
Toyota, or any other entity. We thank TRI for their support. The authors thank the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC and consultation resources that have contributed to the research results reported within this publication. 
