#include "../common/constraints.hpp"
#include "../common/random_generator.hpp"
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>
#include <set>
#include <map>

const int BOUND_N = 15;
const int NUM_OF_PROBLEMS = 3;

const int COMMAND_ARG    = 0;
const int FILENAME_ARG   = 1;
const int PROBLEM_ID_ARG = 2;
const int SEED_ARG       = 3;
const int NUM_OF_ARGS    = 4;

int popcount(int bit) {
  int res = 0;
  for(; bit>0; bit>>=1) {
    if(bit & 1) res++;
  }
  return res;
}

void output_expression(FILE *fp,
		       int N,
		       int K,
		       std::vector<int> d,
		       std::vector<int> c,
		       std::vector< std::vector<int> > vars) {
  // variable id compression
  std::map<int, int> comp;
  for(auto &v : vars) for(auto &e : v) comp[e]++;
  int id = 1; // 1-indexed
  for(auto &e : comp) e.second = id++; 
  for(auto &v : vars) for(auto &e : v) e = comp[e];

  // output
  fprintf(fp, "%d %d\n", N, K);
  for(int i=0; i<K; i++) {
    fprintf(fp, "%d %d", d[i], c[i]);
    for(int k=0; k<d[i]; k++) {
      fprintf(fp, " %d", vars[i][k]);
    }
    fprintf(fp, "\n");
  }
}

unsigned long long int str2ull(char *str) {
  char* end_point;
  unsigned long long int ret_number = strtoull(str, &end_point, 10);
  if(*end_point != '\0') {
    fprintf(stderr, "Error: invalid input: \"%s\"\n", str);
    exit(1);
  }
  return ret_number;
}

int main(int argc, char **argv) {
  if(argc != NUM_OF_ARGS) {
    fprintf(stderr, "Usage: %s <filename> <problem-id> <seed>\n", argv[COMMAND_ARG]);
    fprintf(stderr, "<problem-id> \\in {1, 2, 3}\n");
    return 1;
  }

  unsigned long long int seed = str2ull(argv[SEED_ARG]);
  Rand rnd(seed);
  std::mt19937_64 shuffle_engine(seed);

  int case_type = (int)str2ull(argv[PROBLEM_ID_ARG]) - 1;

  if(case_type < 0 or case_type > NUM_OF_PROBLEMS) {
    fprintf(stderr, "Error: problem-id is invalid\n");
    fprintf(stderr, "    problem A => 1\n");
    fprintf(stderr, "    problem B => 2\n");
    fprintf(stderr, "    problem C => 3\n");
    return 1;
  }

  FILE *fp = fopen(argv[FILENAME_ARG], "w");
  if(fp == NULL) {
    fprintf(stderr, "Error: cannot open the file\n");
    return 1;
  }

  int MIN_N = DEF_MIN_N[case_type], MAX_N = DEF_MAX_N[case_type];
  int MIN_D = DEF_MIN_D[case_type], MAX_D = DEF_MAX_D[case_type];
  int MIN_W = DEF_MIN_W[case_type], MAX_W = DEF_MAX_W[case_type];
  int MIN_M = DEF_MIN_M[case_type], MAX_M = DEF_MAX_M[case_type];

  while(1) {
    int N = rnd.NextInt(MIN_N, MAX_N);

    // 項数と次数の制約を直す
    MAX_D = std::min(MAX_D, N);

    // 項は最大でいくつできるか？それによって項数最大を変えないといけないかも
    // N が小さい時に注意 (作れる項数を超えていたらまずいので)
    if(N <= BOUND_N) {
      int size_of_terms = 0;
      for(int bit=0; bit<(1<<N); bit++) {
	if(popcount(bit) <= MAX_D) size_of_terms++;
      }
      MAX_M = std::min(MAX_M, size_of_terms);
    }
      
    int M = rnd.NextInt(MIN_M, MAX_M);
    if(N <= BOUND_N) {
      // N が小さいときは、あり得る次数の項を全部作る
      // 変数の集合の重複は起こらない
      std::vector< std::vector<int> > terms;
      std::vector<int> degs, coeffs;
      for(long long int bit=0; bit<(1LL << N); bit++) {
	std::vector<int> vars;
	for(int i=0; i<N; i++) {
	  if(bit >> i & 1LL) {
	    vars.push_back(i + INDEX);
	  }
	}

	// 次数が制約を満たしてない
	int D = vars.size();
	if(D < MIN_D or D > MAX_D) continue;

	int C = (2 * rnd.NextInt(0, 1) - 1) * rnd.NextInt(MIN_W, MAX_W);
	degs.push_back(D);
	coeffs.push_back(C);
	terms.push_back(vars);
      }

      // 指定した項数がそもそもおかしい
      assert((int)terms.size() >= M);

      // 指定した項数だけランダムに選ぶ
      std::vector<int> random_choose(terms.size());
      for(int i=0; i<M; i++) {
	random_choose[i] = 1;
      }

      std::shuffle(random_choose.begin(), random_choose.end(), shuffle_engine);
      
      std::set<int> appear_id;
      std::vector< std::vector<int> > final_terms;
      std::vector<int> final_degs, final_coeffs;
      for(size_t i=0; i<terms.size(); i++) {
	if(!random_choose[i]) continue;
	for(auto e : terms[i]) appear_id.insert(e);
	final_terms.push_back(terms[i]);
	final_degs.push_back(degs[i]);
	final_coeffs.push_back(coeffs[i]);
      }

      if(appear_id.size() >= MIN_N) {
	N = appear_id.size();
	output_expression(fp, N, M, final_degs, final_coeffs, final_terms);
	break;
      }
    }
    else {
      // 適当に変数を選んで、追加していく
      // 選んだ変数の集合がかぶっていないかのチェックが必要
      std::set< std::vector<int> > used_variables_set;
      std::vector< std::vector<int> > terms;
      std::vector<int> degs, coeffs;
      std::set<int> appear_id;

      for(int i=0; i<M; i++) {
	while(1) {
	  std::vector<int> random_choose(N);

	  // 次数が制約を満たさないことはない
	  int D = rnd.NextInt(MIN_D, MAX_D);
	  for(int k=0; k<D; k++) random_choose[k] = 1;
	  std::shuffle(random_choose.begin(), random_choose.end(), shuffle_engine);
	    
	  // 重複チェック
	  if(used_variables_set.count(random_choose)) continue;
	  used_variables_set.insert(random_choose);
	  int C = (2 * rnd.NextInt(0, 1) - 1) * rnd.NextInt(MIN_W, MAX_W);

	  std::vector<int> vars;
	  for(int k=0; k<N; k++) {
	    if(!random_choose[k]) continue;
	    vars.push_back(k + INDEX);
	    appear_id.insert(k + INDEX);
	  }

	  degs.push_back(D);
	  coeffs.push_back(C);
	  terms.push_back(vars);
	  break;
	}
      }

      if(appear_id.size() >= MIN_N) {
	N = appear_id.size();
	output_expression(fp, N, M, degs, coeffs, terms);
	break;
      }
    }
  }
  fclose(fp);
  return 0;
}
