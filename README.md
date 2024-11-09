# Word Segmentation of Ancient Tamil Text extracted from inscriptions


## Requirements

- Python >= 3.8
- Pip

## Installation Steps

1. Install external dependency 'space-restorer'

    `pip install external-deps/space-restorer`

## Steps to run demo

1. Download model pickle from [zenodo](https://zenodo.org/records/14059256?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQxNDk1NzQ0LTM3MmUtNDdlOS1iNDJhLTQyOTUzMTFiM2NiOSIsImRhdGEiOnt9LCJyYW5kb20iOiI5MWRlMzc3MmQ0MmY5MDI5ZWM5NTI4MGZkMGU2MmM2ZSJ9.9wdR7Kbd2clQbCzTjeYEnNWOV3m1Es6ROfuRkavFF_-dtnH_0VnRepvRMifjq4TVyHRWp57waT1g425nL1Sn3Qto) 
2. Copy AncientTamilSegmenter-v1.0.pickle to `./pickle` directory
2. Run `demo/demo.ipynb` cells in jupyter notebook