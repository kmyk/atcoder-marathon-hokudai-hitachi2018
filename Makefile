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

RUN=0
run: a.out generator.out output_checker.out
	[ -e test ] || mkdir test
	[ -e test/A.${RUN}.in ] || ./generator.out test/A.${RUN}.in 1 ${RUN}
	[ -e test/B.${RUN}.in ] || ./generator.out test/B.${RUN}.in 2 ${RUN}
	[ -e test/C.${RUN}.in ] || ./generator.out test/C.${RUN}.in 3 ${RUN}
	./a.out < test/A.${RUN}.in > test/A.${RUN}.out
	./a.out < test/B.${RUN}.in > test/B.${RUN}.out
	./a.out < test/C.${RUN}.in > test/C.${RUN}.out
	-./output_checker.out test/A.${RUN}.in test/A.${RUN}.out 0
	-./output_checker.out test/B.${RUN}.in test/B.${RUN}.out 0
	-./output_checker.out test/C.${RUN}.in test/C.${RUN}.out 0

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
bench/a:
	${MAKE} bench/of BENCH=A
bench/b:
	${MAKE} bench/of BENCH=B
bench/c:
	${MAKE} bench/of BENCH=C

score:
	for BENCH in A B C ; do for i in `seq ${NUMBER}` ; do cat log/$${BENCH}.$$i.json | jq .score ; done | awk '{ a += $$1 } END { printf "score '$${BENCH}' = %d\n", a }' ; done

check: output_checker.out
	set -e ; for BENCH in A B C ; do for i in `seq ${NUMBER}` ; do echo '[*]' $${BENCH} $$i ;./output_checker.out test/$${BENCH}.$$i.in test/$${BENCH}.$$i.out 0 ; echo ; echo ; done ; done
