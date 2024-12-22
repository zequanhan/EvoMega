
# EvoMega  
**Prediction of Bacteriophage Transcription Factor Binding Sites**  

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Database Download](#database-download)
  - [Installing Dependencies](#installing-dependencies)
  - [Installing MEME Suite](#installing-meme-suite)
- [Usage](#usage)
  - [Running the Motif Analyzer](#running-the-motif-analyzer)
  - [Example](#example)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Introduction

EvoMega is a tool designed to predict transcription factor binding sites in bacteriophages. By analyzing genomic data, it identifies key motifs that are crucial for understanding phage biology and interactions with host organisms.

## Installation

### Prerequisites

Ensure that you have the following dependencies installed on your system:

- **Operating System:** Linux-based systems are recommended.
- **Python:** Version 3.6 or higher.
- **Build Tools:** `build-essential`, `gcc`, `g++`, `make`.
- **Compression Tools:** `7za` or `unzip`.

### Database Download

To download the required database, execute the following command in your terminal:

```bash
wget --save-cookies /tmp/cookies.txt --no-check-certificate "https://drive.usercontent.google.com/download?id=1569KsNmwhVuVNduQNfQ2_KLWx1v_fqGo&export=download&authuser=0&confirm=t&uuid=f30f18ad-3133-4dc9-bed3-cd95a448f69f&at=APvzH3rXKE6IjUztvq4HwPbot34Y:1734682290410" -O exclude_GPD_find_key_motif.zip && rm -f /tmp/cookies.txt
```

### Installing Dependencies

1. **Update Package Lists:**

    ```bash
    sudo apt update
    ```

2. **Install Essential Build Tools and Libraries:**

    ```bash
    sudo apt install build-essential gcc g++ make zlib1g-dev libbz2-dev liblzma-dev
    ```

3. **Install Compression Tools (if not already installed):**

    - **Using `7za`:**

        ```bash
        sudo apt install p7zip-full
        ```

    - **Or using `unzip`:**

        ```bash
        sudo apt install unzip
        ```

4. **Extract the Downloaded Database:**

    You can use either `7za` or `unzip` to extract the `exclude_GPD_find_key_motif.zip` file.

    - **Using `7za`:**

        ```bash
        7za x exclude_GPD_find_key_motif.zip -mmt=on
        ```

    - **Or using `unzip`:**

        ```bash
        unzip exclude_GPD_find_key_motif.zip
        ```

### Installing MEME Suite

The MEME Suite is essential for motif analysis. Follow these steps to install it:

1. **Download MEME Suite:**

    ```bash
    wget https://meme-suite.org/meme/meme-software/5.5.7/meme-5.5.7.tar.gz
    ```

2. **Extract the Archive:**

    ```bash
    tar -xzvf meme-5.5.7.tar.gz
    ```

3. **Navigate to the Extracted Directory:**

    ```bash
    cd meme-5.5.7
    ```

4. **Configure the Build:**

    ```bash
    ./configure
    ```

5. **Compile the Source Code:**

    ```bash
    make
    ```

6. **Install MEME Suite:**

    ```bash
    sudo make install
    ```

7. **Update Your PATH Environment Variable:**

    Add MEME Suite binaries to your PATH by editing the `.bashrc` file.

    ```bash
    nano ~/.bashrc
    ```

    Append the following line at the end of the file:

    ```bash
    export PATH=$PATH:/usr/local/meme/bin
    ```

    Save and exit the editor, then apply the changes:

    ```bash
    source ~/.bashrc
    ```

## Usage

### Running the Motif Analyzer

To use the motif analyzer, execute the following command:

```bash
python scripts/motif_analyzer.py -i INPUT_FILE -o OUTPUT_PATH
```

**Parameters:**

- `-i` or `--input_file`: Path to the input file (e.g., GenBank file).
- `-o` or `--output_path`: Directory where the output will be saved.

### Example

Here is an example of how to run the motif analyzer with a sample input file:

```bash
python scripts/motif_analyzer.py -i NC_002371.gbk -o scripts/output
```

This command analyzes the `NC_002371.gbk` GenBank file and saves the results in the `scripts/output` directory.

## Troubleshooting

- **MEME Suite Not Found:**

    Ensure that MEME Suite is correctly installed and that the path is added to your `PATH` environment variable. You can verify by running:

    ```bash
    meme --version
    ```

    If the command is not found, revisit the [Installing MEME Suite](#installing-meme-suite) section.

- **Permission Issues:**

    If you encounter permission errors during installation, ensure you have the necessary rights or use `sudo` where appropriate.

- **Missing Dependencies:**

    Make sure all required dependencies are installed. Refer to the [Installing Dependencies](#installing-dependencies) section.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or support, please contact [your.email@example.com](mailto:your.email@example.com).
