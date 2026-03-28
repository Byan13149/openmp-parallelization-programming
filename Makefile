# Makefile for MPhil DIS Cholesky Factorization
#
# Usage:
#   make                  — build everything (default: opt1)
#   make IMPL=baseline    — build using the baseline implementation
#   make IMPL=opt1        — build using the opt1 implementation
#   make test             — build and run tests
#   make bench            — build and run benchmark
#   make clean            — remove all build artifacts

# --- Configuration ---
CC       = gcc
CFLAGS   = -Wall -Wextra -O2
LDLIBS   = -lm

# Enable OpenMP (set OPENMP=0 to disable, e.g. on macOS with Apple Clang)
OPENMP   ?= 1
ifeq ($(OPENMP),1)
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
endif

# Select implementation: baseline or opt1 (default: opt1)
IMPL     ?= opt1

# --- Directories ---
SRCDIR   = src
INCDIR   = include
TESTDIR  = test
EXDIR    = example
BUILDDIR = build

# --- Source selection ---
LIB_SRC  = $(SRCDIR)/cholesky_$(IMPL).c
LIB_OBJ  = $(BUILDDIR)/cholesky_$(IMPL).o

# --- Targets ---
LIB      = $(BUILDDIR)/libcholesky.a
EXAMPLE  = $(BUILDDIR)/cholesky_example
TEST_BIN = $(BUILDDIR)/test_cholesky
BENCH    = $(BUILDDIR)/bench_$(IMPL)

.PHONY: all test bench clean

all: $(LIB) $(EXAMPLE) $(TEST_BIN) $(BENCH)

# --- Build directory ---
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- Library ---
$(LIB_OBJ): $(LIB_SRC) $(INCDIR)/mphil_dis_cholesky.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^

# --- Example ---
$(EXAMPLE): $(EXDIR)/main.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lcholesky $(LDLIBS) -o $@

# --- Tests ---
$(TEST_BIN): $(TESTDIR)/test_cholesky.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lcholesky $(LDLIBS) -o $@

# --- Benchmark ---
$(BENCH): $(TESTDIR)/benchmark.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lcholesky $(LDLIBS) -o $@

# --- Convenience targets ---
test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH)
	./$(BENCH)

clean:
	rm -rf $(BUILDDIR)
