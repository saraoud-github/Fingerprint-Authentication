# Fingerprint Authentication

This repo contains the code for fingerprint classification and authentication. This code is part of our Capstone Design Project at the _Lebanese American University_.

## Required Datasets:

The NIST dataset of fingerprint images contains **2000 8-bit gray scale fingerprint image pairs**. Each image is 512-by-512 pixels with 32 rows of white space at the bottom and classified using one of the five following classes: 
1. A=Arch
2. L=Left Loop
3. R=Right Loop
4. T=Tented Arch
5. W=Whorl.

The dataset is evenly distributed over each of the five classifications with **400 fingerprint pairs from each class**.

**The dataset can be downloaded from: [NIST 8-Bit Gray Scale Images of Fingerprint Image Groups (FIGS)](https://academictorrents.com/details/d7e67e86f0f936773f217dbbb9c149c4d98748c6)**

**Additional datasets were manually labelled and can be downloaded from:**

1. [FVC 2000](http://bias.csr.unibo.it/fvc2000/)
2. [FVC 2002](http://bias.csr.unibo.it/fvc2002/)
3. [FVC 2004](http://bias.csr.unibo.it/fvc2004/download.asp)
4. [The Hong Kong Polytechnic University Contactless 2D to Contact-based 2D Fingerprint Images Database](http://www4.comp.polyu.edu.hk/~csajaykr/fingerprint.htm)

## How to use the repo:
1. Fingerprint Authentication.ipynb contains the code required to train, save, and download the ML model. Run each cell to obtain the model files.
2. Run FingerprintIdentification.py
