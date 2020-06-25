## BIN-PACKING SIMULATION
This repository implements a software tool to assist development and evaluation of bin packing algorithms. This is based on the paper
#### Towards Robust Product Packing with a Minimalistic End-Effector ([pdf](https://arxiv.org/pdf/1703.03347.pdf))([website](http://paul.rutgers.edu/~cm1074/PHYSIM.html))
By Rahul Shome*, Wei N. Tang*, Changkyu Song, Chaitanya Mitash, Chris Kourtev, Jingjin Yu, Abdeslam Boularias, and Kostas E. Bekris (Rutgers University).

In proceedings of IEEE International Conference on Robotics and Automation (ICRA), 2019.

### Citing
To cite the work:

```
@inproceedings{shome2019towards,
  title={Towards robust product packing with a minimalistic end-effector},
  author={Shome, Rahul and Tang, Wei N and Song, Changkyu and Mitash, Chaitanya and Kourtev, Hristiyan and Yu, Jingjin and Boularias, Abdeslam and Bekris, Kostas E},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={9007--9013},
  year={2019},
  organization={IEEE}
}
```
### Setup
1. Download and extract [Blender](https://drive.google.com/file/d/1TqeLcV5nOvCZfRzfXyZTp8jS63MYlVbg/view?usp=sharing)
2. In ```~/.bashrc```, add line ```export BLENDER_PATH=/path/to/blender/blender```

### Demo
1. In ```~/.bashrc```, add line ```export PHYSIM_GENDATA=/path/to/repo```.
2. Run ```python generate_packing_data.py``` .
3. The generated data can be found in the folder ```rendered_images```
