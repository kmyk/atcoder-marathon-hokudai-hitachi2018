.PHONY: build run

CXX ?= g++
CXXFLAGS = -std=c++14 -Wall -O2 -g -DLOCAL

build: a.out generator.out output_checker.out

a.out: main.cpp
	${CXX} ${CXXFLAGS} $<

generator.out: toolkit/scripts/generator.cpp
	${CXX} ${CXXFLAGS} -o $@ $<
output_checker.out: toolkit/scripts/output_checker.cpp
	${CXX} ${CXXFLAGS} -o $@ $<

run: a.out generator.out output_checker.out
	[ -e test ] || mkdir test
	[ -e test/A.0.in ] || ./generator.out test/A.0.in 1 0
	[ -e test/B.0.in ] || ./generator.out test/B.0.in 2 0
	[ -e test/C.0.in ] || ./generator.out test/C.0.in 3 0
	./a.out < test/A.0.in > test/A.0.out
	./a.out < test/B.0.in > test/B.0.out
	./a.out < test/C.0.in > test/C.0.out
	-./output_checker.out test/A.0.in test/A.0.out 0
	-./output_checker.out test/B.0.in test/B.0.out 0
	-./output_checker.out test/C.0.in test/C.0.out 0

submit:
	oj s -y https://atcoder.jp/contests/hokudai-hitachi2018/tasks/hokudai_hitachi2018_a main.cpp --no-open
	oj s -y https://atcoder.jp/contests/hokudai-hitachi2018/tasks/hokudai_hitachi2018_b main.cpp --no-open
	oj s -y https://atcoder.jp/contests/hokudai-hitachi2018/tasks/hokudai_hitachi2018_c main.cpp

bench: a.out generator.out output_checker.out
	[ -e test ] || mkdir test
	[ -e log ] || mkdir log
	for e in A B C ; do for i in `seq 100` ; do [ -e test/$$e.$$i.in ] || ./generator.out test/$$e.$$i.in $$(echo $$e | tr ABC 123) $$i ; done ; done
	for e in A B C ; do for i in `seq 100` ; do echo test/$$e.$$i.in ; LOG=log/$$e.$$i.json ./a.out < test/$$e.$$i.in > test/$$e.$$i.out ; done ; done
	for e in A B C ; do for i in `seq 100` ; do cat log/$$e.$$i.json | jq .score ; done | awk '{ a += $$1 } END { printf "score '$$e' = %d\n", a }' ; done
