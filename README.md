# pyhdc
A Python library to support long binary vectors

## Installation

The `pyhdc` library can be installed via

```
cd lib
sudo python3 setup.py install
```

**Note!** If you intend to work with event-based data, we recommend you also install our
[pydvs](https://github.com/better-flow/pydvs) library as well - it *is* a dependency for some tools and scripts in this
repository.

### Vector Length
The library supports 3 types of vector lengths - *128*, *3200* and *8160*. This is preset during installation by modifying
last lines of `lib/setup.py`.

### Permutations
Two chains of permutations are supported: `x` and `y`. Each chain is initialized with a random permutation, and then the
higher order permutations *(P(P(P...)))* up to a certain maximum order are generated. The permutations are generated
once with a separate `lib/header_gen/main.cpp` script. To change permutations, you will need to run the script and
rebuild the library.


## Data
The preprocessed version of MVSEC autonomous driving event-based dataset is available here:
 * [outdoor_day_1](https://drive.google.com/file/d/1cLks_SqnLbqSHRwPWOJn2tuRj4pasAay/view?usp=sharing) (1.1 GB)
 * [outdoor_day_2](https://drive.google.com/file/d/1rBj6gkgCSMTXO1rWs4--AF-4jwdXxFPq/view?usp=sharing) (4.8 GB)
 * [outdoor_night_1](https://drive.google.com/file/d/158ZHB1NsX3Al7_59BUPVfCIUKOb09kL3/view?usp=sharing) (0.9 GB)
 * [outdoor_night_2](https://drive.google.com/file/d/1Aq5SDZmQdA3GbN6lJiiJxVOQdHVpbIRW/view?usp=sharing) (1.6 GB)
 * [outdoor_night_3](https://drive.google.com/file/d/1nYHvRaLmhQkCaMo6Q7LMgDOlybIyKYII/view?usp=sharing) (1.1 GB)
 * [small_test_sequence](https://drive.google.com/file/d/1urIDRX1KF97tgqiXc8W62ucw35mdd6CI/view?usp=sharing) (112.5 MB)

Please find the main dataset webpage [here](https://daniilidis-group.github.io/mvsec/)

## Usage
The code is under heavy development and API may change!

### Library
Please consult `lib/sanity.py` for examples of usage.

### Egomotion estimation with HBVs
Please keep in mind that this code is a *work in progress* and has been changed multiple times; we always welcome collaboration
and merge requests! The basic usage is:

```
python3 image2vec/train.py --base_dir ./MVSEC/outdoor_day_1
```

Add `--use-direct-encoding` to the command if you wish to enable 'direct pixel encoding' - where each pixel of the image
is first converted to a vector and the final image encoding is a result of XOR, permutations and consensus sum
operations on these vectors. This option is slower mainly because the algorithm is implemented in Python, without C backend.

## Citation
If you use any of this code, please cite our publication in Science Robotics ([article](https://robotics.sciencemag.org/content/4/30/eaaw6736)).

```bibtex
@Article{Mitrokhineaaw6736,
	author = {Mitrokhin, A. and Sutor, P. and Ferm{\"u}ller, C. and Aloimonos, Y.},
	title = {Learning sensorimotor control with neuromorphic sensors: Toward hyperdimensional active perception},
	volume = {4},
	number = {30},
	elocation-id = {eaaw6736},
	year = {2019},
	doi = {10.1126/scirobotics.aaw6736},
	publisher = {Science Robotics},
	URL = {https://robotics.sciencemag.org/content/4/30/eaaw6736},
	eprint = {https://robotics.sciencemag.org/content/4/30/eaaw6736.full.pdf},
	journal = {Science Robotics}
}
```
