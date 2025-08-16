# Fundraising_prediction documentation!

## Description

A simple prediction model of fundraising industry

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `Datasets/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `Datasets/data/` to `data/`.


