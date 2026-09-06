# Getting Started with `clif`

A guide for new users — from a bare Chromebook to running examples and contributing code.

---

## 1. Enable Linux on Your Chromebook

ChromeOS ships with a built-in Debian Linux container called Crostini. No third-party tools are required.

1. Open **Settings** → **Advanced** → **Developers**
2. Next to *Linux development environment*, click **Turn on**
3. Follow the prompts (allocate at least 10 GB of disk space; 20 GB is comfortable)
4. A terminal window opens automatically when setup completes — this is your Linux shell

> **ChromeOS version note:** The menu path above reflects recent ChromeOS releases. If you don't see a *Developers* section, look under **Settings** → search for "Linux".

From this point on, all commands are entered in that Linux terminal.

---

## 2. Install Miniconda

**Why Miniconda and not Anaconda?**
Anaconda installs roughly 300 packages (~3 GB) whether you need them or not. Miniconda installs only the conda package manager and Python itself (~400 MB). You then add exactly what your project requires. For a focused numerical/research environment, Miniconda is the better starting point.

### Download and install

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Accept the license, accept the default install location (`~/miniconda3`), and answer **yes** when asked whether to initialize conda. Then reload your shell:

```bash
source ~/.bashrc
```

Verify:

```bash
conda --version
```

### Create the project environment

```bash
conda create -n clif python=3.11
conda activate clif
```

You will need to run `conda activate clif` each time you open a new terminal. To make it automatic, add it to your `~/.bashrc`:

```bash
echo "conda activate clif" >> ~/.bashrc
```

### Install required packages

```bash
pip install numpy numba clifford sphinx furo myst-nb
```

### Fix for Intel OpenMP (Crostini-specific)

Numba uses Intel's OpenMP runtime, which conflicts with itself inside the Crostini container and produces a warning or crash on first import. The fix is a single environment variable:

```bash
echo "export KMP_AFFINITY=disabled" >> ~/.bashrc
source ~/.bashrc
```

This is harmless on other platforms and safe to leave in permanently.

---

## 3. Obtain the `clif` Package

Clone the repository from GitHub and install it in editable mode so that any local changes you make are immediately reflected without reinstalling:

```bash
git clone https://github.com/[AUTHOR_GITHUB]/[REPO_NAME].git
cd [REPO_NAME]
pip install -e .
```

> **[PLACEHOLDER]** Replace `[AUTHOR_GITHUB]/[REPO_NAME]` with the actual repository path. The author will update this before the first public release.

Verify the install:

```bash
python -c "import clifford; print('OK')"
```

---

## 4. Run the Examples

Example notebooks live in `docs/notebooks/`. The fastest way to explore them is with Jupyter:

```bash
pip install jupyterlab
jupyter lab
```

ChromeOS will offer to open the browser automatically. If it doesn't, copy the URL printed in the terminal (beginning with `http://127.0.0.1:8888/...`) and paste it into Chrome.

Suggested starting order:

| Notebook | What it shows |
|---|---|
| `api_examples.ipynb` | Basic multivector operations via the public API |
| `fls_sri_walkthrough.ipynb` | The core inversion algorithm, step by step |
| `benchmarks.ipynb` | Timing comparisons across dimension and signature |

---

## 5. Frequently Asked Questions

**`conda: command not found` after install**
Run `source ~/.bashrc` or close and reopen the terminal. If the problem persists, check that the conda initializer block was added to `~/.bashrc` (the installer does this when you answer *yes* to the init question).

**Numba prints an OpenMP warning on first import**
You have not yet set `KMP_AFFINITY=disabled`. Follow the step in Section 2 above.

**Numba is slow on the first call**
Expected — Numba compiles JIT kernels on first use and caches them in `__pycache__`. Subsequent calls are fast.

**The Linux terminal closes when I close the lid**
Crostini suspends with the Chromebook. Long-running computations will be interrupted. For extended jobs, keep the lid open or adjust power settings under **Settings** → **Device** → **Power**.

**How much RAM/CPU does the Linux container get?**
By default, Crostini uses up to half of available RAM and all CPU cores. No tuning is needed for `clif` at typical dimensions (up to 12D); 16D and above benefit from more RAM.

**Can I use VS Code?**
Yes. Install the Linux `.deb` from the VS Code website and open it from the ChromeOS launcher. The Python and Jupyter extensions work normally inside Crostini.

---

## 6. Contact the Author

For questions, bug reports, or discussion about the mathematics:

**Email:** `[AUTHOR_EMAIL]`

Please include your ChromeOS version (`Settings` → `About ChromeOS`), the output of `conda list | grep -E "numpy|numba|clifford"`, and a minimal reproducing example where relevant.

---

## 7. Extend the Package (Branching Workflow)

`clif` uses a straightforward feature-branch workflow. The `main` branch is stable; all new work goes on a named branch.

### One-time git setup (if you haven't done this before)

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### Start a new feature

```bash
git checkout main
git pull                          # make sure you're up to date
git checkout -b feature/my-topic  # create and switch to your branch
```

### Work, commit, push

```bash
# ... edit files ...
git add .
git commit -m "Brief description of what changed and why"
git push -u origin feature/my-topic
```

### Open a pull request

Go to the repository on GitHub, where your branch will appear with an offer to open a pull request. Describe what the change does and, for algorithmic changes, include operation counts or timing comparisons with the existing code.

### Keep your branch current

If `main` moves forward while you are working:

```bash
git fetch origin
git rebase origin/main
```

---

## 8. Further Reading

| Resource | Location |
|---|---|
| Package API reference | `docs/_build/html/index.html` (build locally with `cd docs && make html`) |
| Algorithm derivation | `clifford/scratch/FLS-SRI-to-11D.md` |
| Sign table architecture | `clifford/inverse/Signature_and_Reference.md` |
| Changelog | `HANDOFF.md` |

To build the docs locally:

```bash
cd docs
make html
```

Then open `docs/_build/html/index.html` in Chrome.

---

*This document lives at `docs/Getting_Started.md`. If you find a step that doesn't work, please email the author — it is almost certainly a documentation bug, not a user error.*
