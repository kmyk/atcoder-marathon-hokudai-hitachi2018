.PHONY: build run

CXX ?= g++
CXXFLAGS = -std=c++14 -Wall -O2 -g

build: a.out
a.out: main.cpp
	${CXX} $<

SEED := 0

run: a.out generator.out output_checker.out
	[ -e test ] || mkdir test
	[ -e test/A.case-0.in ] || ./generator.out test/A.case-0.in 1 ${SEED}
	[ -e test/B.case-0.in ] || ./generator.out test/B.case-0.in 2 ${SEED}
	[ -e test/C.case-0.in ] || ./generator.out test/C.case-0.in 3 ${SEED}
	./a.out < test/A.case-0.in > test/A.case-0.out
	./a.out < test/B.case-0.in > test/B.case-0.out
	./a.out < test/C.case-0.in > test/C.case-0.out
	-./output_checker.out test/A.case-0.in test/A.case-0.out ${SEED}
	-./output_checker.out test/B.case-0.in test/B.case-0.out ${SEED}
	-./output_checker.out test/C.case-0.in test/C.case-0.out ${SEED}

generator.out: toolkit/scripts/generator.cpp
	${CXX} -o $@ $<
output_checker.out: toolkit/scripts/output_checker.cpp
	${CXX} -o $@ $<

submit:
	oj s -y https://atcoder.jp/contests/hokudai-hitachi2018/tasks/hokudai_hitachi2018_a main.cpp --no-open
	oj s -y https://atcoder.jp/contests/hokudai-hitachi2018/tasks/hokudai_hitachi2018_b main.cpp --no-open
	oj s -y https://atcoder.jp/contests/hokudai-hitachi2018/tasks/hokudai_hitachi2018_c main.cpp
