# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = pipenv run sphinx-build
SPHINXPROJ    = imagelab
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = ../imagelab
.DEFAULT_GOAL := html
SRC = $(shell find $(SOURCEDIR) -type f -name '*.rst') $(shell find $(PACKAGEDIR) -type f -name '*.py')

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

watch:  ## Recompile on any update
	@fswatch -o $(SRC) | xargs -n1 -I{} make html

open: html
	open build/html/index.html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)