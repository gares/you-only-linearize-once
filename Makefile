ELPI ?= elpi

# The paper's running example, f(u) = −sin(u), through J, U, T and L.
all:
	$(ELPI) -test run-example.elpi

# All the tests. Fails (exit code 1) if any check fails.
test:
	$(ELPI) -test run-tests.elpi

.PHONY: all test
