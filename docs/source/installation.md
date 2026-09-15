
[![PyPI version](https://badge.fury.io/py/qbiocode.svg)](https://badge.fury.io/py/qbiocode) [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.10-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.12-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
 
 This project requires **Python 3.10 or higher** and has been tested with Python versions 3.10, 3.11, and 3.12.

**Note:** Python 3.10+ is required for compatibility with the latest Qiskit ecosystem (qiskit-ibm-runtime 0.44.0+).

# Installation

QBioCode can be installed via PyPI, conda, or from source.

## Quick Install from PyPI (Recommended)

Install the latest stable version:

```bash
pip install qbiocode
```

Install with apps support (QProfiler, QSage):

```bash
pip install 'qbiocode[apps]'
```

Install with QuVINE graph embeddings (`quvine_rwr`, `quvine_dtqw`, `node2vec`, ...):

```bash
pip install 'qbiocode[quvine]'
```

Install with all optional dependencies:

```bash
pip install 'qbiocode[all]'
```

```{note}
**For zsh users:** The quotes around `'qbiocode[apps]'` are required because zsh interprets square brackets as glob patterns. Bash users can omit quotes, but using them works in both shells.
```

## Optional dependency extras

A plain `pip install qbiocode` gives you the full classical and quantum pipeline:
data generation, all scikit-learn embeddings, PQK, and the classical/quantum
models. Everything below is additive.

| Extra | Command | What it adds |
| --- | --- | --- |
| *(none)* | `pip install qbiocode` | Core library: embeddings (`pca`, `nmf`, `umap`, `tsne`, `spectral`, ...), PQK, classical + quantum models, `evaluate_graph`, `scale_train_test` |
| `apps` | `pip install 'qbiocode[apps]'` | Hydra-driven CLIs for the QProfiler and QSage apps (`hydra-core`, `joblib`) |
| `quvine` | `pip install 'qbiocode[quvine]'` | QuVINE quantum/classical graph embeddings — 83 methods reachable through `get_embeddings("quvine_*", ...)` |
| `tabpfn` | `pip install 'qbiocode[tabpfn]'` | The `tabpfn` classical model — a pretrained tabular transformer. No API key or license acceptance needed for the pinned default; see below |
| `docs` | `pip install 'qbiocode[docs]'` | Sphinx toolchain for building this documentation locally |
| `dev` | `pip install 'qbiocode[dev]'` | `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy` |
| `all` | `pip install 'qbiocode[all]'` | Union of every extra above |

Extras combine, so `pip install 'qbiocode[apps,quvine]'` is valid.

`catboost` is *not* an extra: it is a core dependency, so the `catboost` model
works from a plain `pip install qbiocode` exactly as `xgb` does.

### QuVINE graph embeddings

QuVINE ships as a single all-or-nothing extra rather than one extra per method
family. Its dependencies (`gensim`, `hiperwalk`, `node2vec`, `omegaconf`,
`python-louvain`, `ripser`, `torch-geometric`) overlap heavily across the walk,
spectral and neural methods, so a partial install would leave most method names
resolving and a handful raising at call time — harder to reason about than a
single yes/no.

Without the extra, `import qbiocode` and every classical embedding keep working;
only the QuVINE method names are affected, and they raise an actionable error
naming the missing module and the exact install command:

```python
>>> import qbiocode as qbc
>>> qbc.get_embeddings("quvine_rwr", X_train, X_test)
QuvineDependencyError: QuVINE method 'quvine_rwr' requires the optional 'quvine'
extra. Install it with: pip install "qbiocode[quvine]"
(missing: gensim, provided by gensim>=4.4)
```

`QuvineDependencyError` subclasses `ImportError`, so `except ImportError` already
catches it. `qbiocode.embeddings.QUVINE_METHODS` lists the method names, and
`qbiocode.apps.quvine.missing_dependencies()` reports which optional packages are
absent without raising.

```{note}
`torch` is already a base dependency, so the `quvine` extra does not pull in a
second deep-learning stack. On macOS, `pip install 'qbiocode[quvine]'` may need
`brew install cmake` first for `ripser`.
```

```{warning}
The `quvine` extra pins `setuptools<81` because `node2vec` imports
`pkg_resources`, which setuptools 81 removed. If you install QuVINE into an
environment that needs a newer setuptools, use a separate virtual environment.
```

### TabPFN

TabPFN is an extra because of its weight: it brings `mlx`, `lightgbm`, `huggingface-hub`,
`safetensors`, `einops` and `httpx` along with torch's ecosystem, and `import qbiocode` must
not pull torch in. That is the only reason — it needs **no API key and no license
acceptance**:

```bash
pip install 'qbiocode[tabpfn]'
```

QBioCode pins `model_version: v2`, whose weights are published under the Prior Labs License
(Apache 2.0 plus an attribution clause) and download anonymously on first fit. So a first
run needs network access, and nothing else.

```{warning}
**The model version is a licensing choice.** Only `v2` permits commercial use. The newer
`v2.5`, `v2.6` and `v3` checkpoints are under per-version **non-commercial and
non-production** licenses, and additionally require accepting that license against a Prior
Labs account — which upstream does interactively, so they cannot be fetched unattended and
an API key alone is not sufficient. Selecting one with `model_version` warns and names the
license. See the table in the [configuration guide](apps/config.md).

`v2` is the default because QBioCode is Apache-2.0 software whose users include companies.
```

```{note}
**macOS:** a TabPFN fit maps torch's OpenMP runtime into a process that already has
xgboost's, and the second one to open a parallel region dies below Python — exit 139, no
traceback, a notebook reporting only "kernel died". QBioCode sets `OMP_NUM_THREADS=1` before
importing TabPFN to prevent this, and warns that it has. Set the variable yourself before
importing `qbiocode` to choose differently. CatBoost is unaffected: it uses its own thread
pool.
```

### Supplying an API key (only for the restricted versions)

If you opt into `v2.5`, `v2.6` or `v3`, accept the license at <https://ux.priorlabs.ai>
(Licenses tab) and copy the API key from the Account page. Store it in a file **outside the
repository**, which is both convenient (it survives a new shell and a notebook kernel
started from a launcher) and the only way it cannot be committed:

```bash
python -c "from qbiocode.utils import write_token_template; print(write_token_template())"
# -> ~/.config/qbiocode/tabpfn.json, created mode 0600
# paste the key into its "token" field
```

QProfiler reads it automatically when `tabpfn` is in the model list, via the
`tabpfn_json_path` config key. Elsewhere, call `qbiocode.utils.load_tabpfn_token()` before
fitting. Exporting `TABPFN_TOKEN` still works and takes precedence.

After that it runs unattended like any other model. To stay fully offline, point
`model_path` at an already-downloaded checkpoint instead.

```{warning}
Keep the key out of the checkout. `~/.config/qbiocode/tabpfn.json` is the supported
location precisely because a gitignored file inside the tree is *unlikely* to be
committed, not unable to be. QBioCode never logs or prints the token;
`qbiocode.utils.describe_token_source()` reports only whether one is configured and where
it came from.
```

Without the extra, `import qbiocode` and every other model keep working; only `tabpfn`
is affected, and it raises an actionable error naming the extra:

```python
>>> import qbiocode as qbc
>>> qbc.compute_tabpfn(X_train, X_test, y_train, y_test, args)
ImportError: TabPFN is not installed. It is an optional extra rather than a core
dependency because of its weight: it brings torch's ecosystem along with
mlx, lightgbm, huggingface-hub and safetensors.

Install it with:
  pip install "qbiocode[tabpfn]"

No API key or license acceptance is needed: QBioCode pins model_version 'v2',
whose weights are under the Prior Labs License (Apache 2.0 plus attribution)
and download anonymously on first fit.
```

`qbiocode.learning.compute_tabpfn.tabpfn_is_available()` reports whether the extra is
present without importing it — and importantly without importing `torch`, which
QBioCode deliberately keeps out of processes that do not need it (see the OpenMP note
below).

```{note}
TabPFN supports at most **10 classes**. Unlike its row and feature limits, that ceiling
is fixed by the checkpoint and is not waived by `ignore_pretraining_limits`.
```

## Install with Conda

QBioCode will be available on conda-forge and bioconda channels after the initial release review process.

### Once Available on Conda-forge

```bash
# Install from conda-forge (recommended)
conda install -c conda-forge qbiocode

# Or from bioconda (includes bioinformatics dependencies)
conda install -c bioconda -c conda-forge qbiocode

# Create a new environment with qbiocode
conda create -n qbiocode -c conda-forge qbiocode
conda activate qbiocode
```

### Current Workaround

While conda packages are pending submission, you can use pip within a conda environment:

```bash
# Create a conda environment with Python 3.10+
conda create -n qbiocode python=3.10
conda activate qbiocode

# Install QBioCode from PyPI
pip install qbiocode

# Or with apps support
pip install 'qbiocode[apps]'
```

**Note**: See [docs/CONDA_SUBMISSION.md](https://github.com/qiskit-community/QBioCode/blob/main/docs/CONDA_SUBMISSION.md) for information about the conda submission process.

## Install from Source

### Option 1: Setting up a Python Virtual Environment (venv)

This is the standard way to create an isolated Python enviroment.

**Steps:**

1. **Install pip (if you don't have it):**
  ```bash
   python -m ensurepip --default-pip
  ```
  or on some systems:
 ```bash
  sudo apt update
  sudo apt install python3-pip
  ```
2. **Create a virtual enviroment:**
```bash
   python -m venv venv
  ```
This command creatas a new directory named `venv` (you can choose a different name if you prefer) containing a copy of the Python interpreter and necessary supporting files.

3. **Activate the virtual enviroment:**
* **On macOS and Linux:**
```bash
  source venv/bin/activate
  ```
* **On Windows (command promt):**
```bash
  venv\Scripts\activate
  ```
* **On Windows (PowerShell):**
```bash
  .\venv\Scripts\Activate.ps1
  ```
Once the activated, you'll see `(venv)` at the beginning of your terminal promt.

4. **Install QBioCode:**
   
   Once the virtual environment is activated, install QBioCode:
   
   ```bash
   # Clone the repository first
   git clone https://github.com/qiskit-community/QBioCode.git
   cd QBioCode
   
   # Install in editable mode
   pip install -e .
   
   # Or install with apps support
   pip install -e ".[apps]"

   # Or with QuVINE graph embeddings
   pip install -e ".[quvine]"

   # Or everything
   pip install -e ".[all]"
   ```

5. **macOS Users: Install OpenMP for XGBoost (Required)**
   
   On macOS, XGBoost requires the OpenMP library. Install it using Homebrew:
   ```bash
   brew install libomp
   ```
   
   If you don't have Homebrew installed, visit [https://brew.sh/](https://brew.sh/) for installation instructions.
   
   After installing OpenMP, you may need to reinstall XGBoost:
   ```bash
   pip install --force-reinstall xgboost
   ```

6. **Deactivate the virtual enviroment (when you are done):**
   ```bash
    deactivate
   ```
   This will return you to your base Python enviroment.

## Option 2: Setting up a Conda Enviroment

1. Create the environment from the `requirements.txt` file.  This can be done using anaconda, miniconda, miniforge, or any other environment manager.
```
conda create -n qbc python==3.12

```
* Note: if you receive the error `bash: conda: command not found...`, you need to install some form of anaconda to your development environment.
2. Activate the new environment:
```
conda activate qbc
pip install .
```

3. **macOS Users: Install OpenMP for XGBoost (Required)**
   
   On macOS, XGBoost requires the OpenMP library. Install it using Homebrew:
   ```bash
   brew install libomp
   ```
   
   If you don't have Homebrew installed, visit [https://brew.sh/](https://brew.sh/) for installation instructions.
   
   After installing OpenMP, you may need to reinstall XGBoost:
   ```bash
   pip install --force-reinstall xgboost
   ```

4. Verify that the new environment and packages were installed correctly:
```
conda env list
pip list
```
<!-- * Additional resources:
   * [Connect to computing cluster](http://ccc.pok.ibm.com:1313/gettingstarted/newusers/connecting/)
   * [Set up / install Anaconda on remote linux server](https://kengchichang.com/post/conda-linux/)
   * [Set up remote development environment using VSCode](https://code.visualstudio.com/docs/remote/ssh) -->

## Option 3: Using Galaxy (Cloud-Based, No Local Installation)

If you prefer not to install QBioCode on your local or personal machine, you can use [Galaxy](https://usegalaxy.org/), a free, web-based platform for data-intensive biomedical research.

```{admonition} Why Galaxy?
:class: tip
- **No installation required**: Run everything in your browser
- **Free computational resources**: Access to cloud computing
- **Jupyter notebook support**: Run QBioCode tutorials directly
- **Persistent workspace**: Your work is saved in the cloud
```

### Step 1: Register for a Galaxy Account

1. Go to [https://usegalaxy.org/](https://usegalaxy.org/)
2. Click **"Login or Register"** in the top menu
3. Select **"Register"** and fill in:
   - Email address
   - Password
   - Public name (username)
4. Click **"Create"** to complete registration
5. Verify your email address (check your inbox for confirmation link)

### Step 2: Launch a Qiskit Jupyter Notebook (Recommended - Pre-installed QBioCode)

1. **Log in** to your Galaxy account at [https://usegalaxy.org/](https://usegalaxy.org/)

2. From the left menu, select **"Interactive Tools"**

3. Search for **"Qiskit Jupyter notebook"**

4. Click on the Qiskit Jupyter tool to launch it

5. Configure the notebook environment:
   - **Allocate resources**: Default settings are usually sufficient
   - Click **"Execute"** or **"Run Tool"**

6. Wait for the notebook server to start (this may take 1-2 minutes)

7. Once ready, click the **link** to open your Jupyter environment in a new tab

```{admonition} Pre-installed QBioCode
:class: tip
The **Qiskit Jupyter notebook** in Galaxy comes with QBioCode pre-installed! You can start using it immediately without any installation steps. However, note that this may not be the latest version of the code.
```

### Step 3: Using QBioCode (Pre-installed Version)

Once your Qiskit Jupyter notebook server is running:

1. **Verify QBioCode is available**:
   
   Open a new notebook or terminal and run:
   ```python
   import qbiocode as qbc
   print(f"QBioCode version: {qbc.__version__}")
   ```

2. **Access the tutorials**:
   
   The tutorials should be available in the file browser. Navigate to the QBioCode tutorial directory and open any notebook:
   - `Artificial_data_generation/example_data_generation.ipynb`
   - `QProfiler/example_qprofiler.ipynb`
   - `QProfiler/sc_binary_qprofiler.ipynb`
   - `QSage/qsage.ipynb`
   - `Quantum_Projection_Learning/QPL_example.ipynb`

### Step 3b: Install Latest Version (Optional)

If you need the latest version of QBioCode with the most recent features and bug fixes:

1. **Open a new terminal** in Jupyter:
   - Click **"File" → "New" → "Terminal"** (in JupyterLab)
   - Or use the **"New" → "Terminal"** button (in classic Jupyter)

2. **Install the latest QBioCode from GitHub**:

   ```bash
   # Clone the repository
   git clone https://github.com/qiskit-community/QBioCode.git
   cd QBioCode
   
   # Install QBioCode (this will upgrade the pre-installed version)
   pip install --upgrade .
   ```

3. **Verify the updated installation**:

   ```bash
   python -c "import qbiocode; print(f'QBioCode version: {qbiocode.__version__}')"
   ```

```{note}
After upgrading, you may need to restart your Jupyter kernel for changes to take effect:
**Kernel → Restart Kernel**
```

### Step 4: Run QBioCode Tutorials

1. **Open a tutorial notebook**:
   - In the Jupyter file browser, navigate to `QBioCode/tutorial/`
   - Click on any tutorial notebook to open it:
     - `Artificial_data_generation/example_data_generation.ipynb`
     - `QProfiler/example_qprofiler.ipynb`
     - `QProfiler/sc_binary_qprofiler.ipynb`
     - `QSage/qsage.ipynb`
     - `Quantum_Projection_Learning/QPL_example.ipynb`

2. **Run the tutorial**:
   - Execute cells sequentially using **Shift+Enter**
   - Follow the instructions in each notebook

### Tips for Using Galaxy

```{tip}
**Best Practices:**
- **Save your work frequently**: Use File → Save to preserve your progress
- **Download important results**: Export notebooks and data files to your local machine
- **Monitor resource usage**: Galaxy sessions have time limits; plan accordingly
- **Use version control**: Consider connecting to GitHub for better workflow management
```

```{warning}
**Important Limitations:**
- Galaxy sessions may timeout after inactivity (typically 1-2 hours)
- Computational resources are shared; quantum simulations may be slower
- Large datasets may require local installation for better performance
- IBM Quantum hardware access requires separate IBM Quantum account setup
```

### Troubleshooting Galaxy Installation

**Issue: pip install fails**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install .
```

**Issue: Import errors**
```bash
# Restart the kernel: Kernel → Restart Kernel
# Then re-import
import qbiocode
```

**Issue: a notebook kernel dies with no traceback on macOS**

`xgboost`, `torch` and `qiskit-aer` each vendor their own copy of `libomp.dylib` under
the same install name, and whichever OpenMP runtime initialises second can kill the
process below Python — so there is no traceback, only "kernel died". `qbiocode`
initialises xgboost's first (see `qbiocode.utils._openmp`), and `compute_tabpfn`
imports `tabpfn` lazily so that merely importing QBioCode never maps torch's runtime
in. If you hit it anyway, set `OMP_NUM_THREADS=1` **before** importing `qbiocode` or
`torch`. CatBoost is unaffected: it uses its own thread pool rather than OpenMP.

**Issue: XGBoost errors on macOS**
```bash
# Install OpenMP library
brew install libomp
pip install --force-reinstall xgboost
```

**Issue: Session timeout**
- Save your work regularly
- Download notebooks before closing
- Restart the interactive tool if needed

### Alternative: Google Colab

Another cloud-based option is [Google Colab](https://colab.research.google.com/):

```text
# In a Colab notebook cell:
!git clone https://github.com/qiskit-community/QBioCode.git
%cd QBioCode
!pip install .
```

Then upload tutorial notebooks from the `tutorial/` directory.

---

<a name="running_qbiocode"></a>
