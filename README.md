# scatternet
scatternet: a library to explore applications of the scattering transform for radio galaxies

### Structure

- scatternet
    - kymatioex: some extensions of the kymatio library to modify the scattering transform. master branch only includes the angle-averaged scattering transform
    - utils
        - classifer.py: abstract class and wrapper classes for all classifiers
        - dataset.py: abtract class and classes for all datasets, including links/references
        - plotting.py: some plotting utils
- scripts: contains scripts for running different feature extraction / classification scenarios

### Install & Setup
Simply run

```
source setup.sh
```
