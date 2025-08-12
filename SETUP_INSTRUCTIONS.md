# Setting Up PDF2MD as a New GitHub Project

Follow these steps to move the PDF2MD project from this subdirectory to its own standalone GitHub repository.

## Step 1: Create New GitHub Repository

1. Go to GitHub and create a new repository:
   - Repository name: `pdf2md`
   - Description: "Convert PDF documents to Markdown using IBM Docling"
   - Make it public
   - Don't initialize with README (we already have one)

## Step 2: Move Files to New Directory

```bash
# Create a new directory outside this project
mkdir ~/pdf2md-standalone
cd ~/pdf2md-standalone

# Copy all PDF2MD files
cp /path/to/ai-town-board/pdf2md/* .

# Verify all files are present
ls -la
# You should see:
# - pdf2md.py
# - README.md
# - requirements.txt
# - setup.py
# - pyproject.toml
# - LICENSE
# - .gitignore
# - SETUP_INSTRUCTIONS.md (this file)
```

## Step 3: Initialize Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PDF2MD - PDF to Markdown converter

- Universal PDF conversion using IBM Docling
- Automatic segmentation for large PDFs
- Simple CLI interface
- Cross-platform support"

# Add remote origin (replace with your GitHub username)
git remote add origin https://github.com/YOURUSERNAME/pdf2md.git

# Create and switch to main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Update Repository URLs

1. Edit `setup.py` and `pyproject.toml` to update the GitHub URLs:
   - Replace `yourusername` with your actual GitHub username
   - Update homepage, repository, and issues URLs

## Step 5: Test Installation

```bash
# Install in development mode to test
pip install -e .

# Test the command
pdf2md --help

# Test with a sample PDF
pdf2md sample.pdf
```

## Step 6: Publish to PyPI (Optional)

Once you're satisfied with the project:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI (you'll need a PyPI account)
python -m twine upload dist/*
```

## Step 7: Update README

Update the README.md installation instructions once published:

```markdown
### Install from PyPI
```bash
pip install pdf2md
```

## Project Structure

Your final project structure should look like:

```
pdf2md/
├── pdf2md.py              # Main CLI script
├── README.md             # Documentation
├── requirements.txt      # Dependencies
├── setup.py             # Setup script
├── pyproject.toml       # Modern Python packaging
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
└── SETUP_INSTRUCTIONS.md # This file
```

## Next Steps

1. **Add GitHub Actions**: Set up automated testing and PyPI publishing
2. **Create Issues Templates**: Help users report bugs effectively
3. **Add Examples**: Include sample PDFs and expected outputs
4. **Documentation**: Consider adding a docs/ folder with detailed guides
5. **Testing**: Add unit tests and integration tests

## Maintenance

Remember to:
- Keep dependencies updated (especially docling)
- Monitor GitHub issues for bug reports
- Consider adding new features based on user feedback
- Maintain Python version compatibility

## Success!

You now have PDF2MD as a standalone project ready for the world to use! 🎉

The tool provides a simple, reliable way for anyone to convert PDFs to Markdown with just one command.