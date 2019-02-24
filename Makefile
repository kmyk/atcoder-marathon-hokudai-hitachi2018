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

BENCH=
NUMBER=150
bench/of: a.out generator.out output_checker.out
	[ "${BENCH}" ]
	[ -e test ] || mkdir test
	[ -e log ] || mkdir log
	for i in `seq ${NUMBER}` ; do [ -e test/${BENCH}.$$i.in ] || ./generator.out test/${BENCH}.$$i.in $$(echo ${BENCH} | tr ABC 123) $$i ; done
	for i in `seq ${NUMBER}` ; do echo test/${BENCH}.$$i.in ; LOG=log/${BENCH}.$$i.json ./a.out < test/${BENCH}.$$i.in > test/${BENCH}.$$i.out ; done
	for i in `seq ${NUMBER}` ; do cat log/${BENCH}.$$i.json | jq .score ; done | awk '{ a += $$1 } END { printf "score '${BENCH}' = %d\n", a }'

bench:
	${MAKE} bench/a
	${MAKE} bench/b
	${MAKE} bench/c
	echo ---
	for BENCH in A B C ; do for i in `seq 100` ; do cat log/$${BENCH}.$$i.json | jq .score ; done | awk '{ a += $$1 } END { printf "score '$${BENCH}' = %d\n", a }' ; done
bench/a:
	${MAKE} bench/of BENCH=A
bench/b:
	${MAKE} bench/of BENCH=B
bench/c:
	${MAKE} bench/of BENCH=C
