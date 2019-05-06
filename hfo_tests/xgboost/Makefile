SRC=xgboost_test.cc
TGT=xgboost_test

cc=c++
CFLAGS ?=-O3
XGBOOST_ROOT ?=$(HOME)/xgboost/
INCLUDE_DIR=-I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include -I$(HOME)/robocin/HFO/hfo -I$(HOME)/robocin/HFO/src
LIB_DIR=-L$(XGBOOST_ROOT)/lib -L$(HOME)/robocin/HFO/lib

build: $(TGT)

$(TGT): $(SRC) Makefile
	$(cc) $(CFLAGS) $(INCLUDE_DIR) $(LIB_DIR) -o $(TGT) $(SRC) -lxgboost -lhfo

run: $(TGT)
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib:$(HOME)/robocin/HFO/hfo ./$(TGT)

clean:
	rm -f $(TGT)
