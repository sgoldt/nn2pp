CC=$(CXX) -std=c++11 -Igtest/include
CFLAGS = -Wall -pedantic -O3  -Xpreprocessor -fopenmp
MFLAGS = -larmadillo -lomp

.PHONY : clean distclean

all: nn2pp.exe

%.o: %.cpp libnn2pp.h
	$(CC) -c $(CFLAGS) $<

%.exe: %.o
	$(CC) -o $@ $(CFLAGS) $< $(MFLAGS)

# Google Test Library
lib/libgtest.a: gtest/src/gtest-all.cc
	$(CC) -isystem gtest/include -I gtest/ -pthread -c gtest/src/gtest-all.cc
	ar -rv lib/libgtest.a gtest-all.o
	rm gtest-all.o

test_nn2pp.exe : test_nn2pp.o lib/libgtest.a
		$(CC) -o test_nn2pp.exe $(CFLAGS) test_nn2pp.o $(lib_objects) -lgtest -lpthread $(MFLAGS) -I gtest/include -L./lib

# ============================================================
# PHONY targets

clean :
	rm -f *.o core gmon.out *.gcno

distclean : clean
	rm -f nn2pp.exe test_nn2pp.exe nn2pp_ode.exe



